"""
SMS Spam Intelligence Dashboard
Support Vector Machines for SMS Spam Detection in Telecommunications
Machine Learning - Unit 2 Project (Flask backend)

All machine-learning work happens here on the server:
  * CSV upload, column detection and validation
  * text preprocessing and TF-IDF feature extraction
  * Support Vector Machine training (linear / RBF / polynomial kernels)
  * kernel experiment and a small grid search
  * Bayesian Logistic Regression (MAP estimate + Laplace approximation)
  * evaluation metrics, confusion matrices and ROC curves
  * live SMS prediction
  * Linear / Robust / Polynomial / Ridge / Bayesian linear regression demos

Every number shown on the website is computed from the dataset that is
currently loaded. Nothing is hard-coded.
"""

import html
import io
import json
import os
import re
import time
import threading
import warnings
from collections import Counter
from functools import wraps

import numpy as np
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import BayesianRidge, HuberRegressor, LinearRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_PATH = os.path.join(BASE_DIR, "data", "sample_sms.csv")

PROJECT_TITLE = "Support Vector Machines for SMS Spam Detection in Telecommunications"
DATASET_URL = "https://www.kaggle.com/uciml/sms-spam-collection-dataset"

RANDOM_STATE = 42
TEST_SIZE = 0.20
MIN_MESSAGES = 20          # smallest dataset we allow (after cleaning)
MIN_PER_CLASS = 5          # each class needs at least this many messages
MAX_UPLOAD_MB = 25
VALID_LABELS = ("ham", "spam")

DEFAULT_SVM = {"kernel": "linear", "C": 1.0, "gamma": "scale", "balanced": False}
DEFAULT_BLR = {"prior_variance": 10.0, "n_features": 1000}
BLR_BIAS_PRIOR_VARIANCE = 100.0   # weak prior for the intercept

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024
app.json.sort_keys = False


class UserError(Exception):
    """An error caused by the input (shown to the user as a friendly message)."""

    def __init__(self, message, status=400, extra=None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.extra = extra or {}


# ---------------------------------------------------------------------------
# Text preprocessing
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
CURRENCY_RE = re.compile(r"[£$€₹]")
NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")
SPACE_RE = re.compile(r"\s+")
STOP_WORDS = set(ENGLISH_STOP_WORDS)


def preprocess_text(text):
    """Clean one SMS. The same function is used by TF-IDF, so training,
    testing and live prediction are all processed identically."""
    t = html.unescape(str(text))                # 1. decode HTML codes (&lt; -> <, &amp; -> &)
    t = t.lower()                               # 2. lowercase
    t = URL_RE.sub(" url ", t)                  # 3a. web links -> 'url'
    t = CURRENCY_RE.sub(" money ", t)           # 3b. currency symbols -> 'money'
    t = NON_ALNUM_RE.sub(" ", t)                # 4. remove punctuation / symbols
    t = SPACE_RE.sub(" ", t).strip()            # 5. remove extra whitespace
    return t


def preprocessing_steps(text):
    """Return every intermediate step (used by the website to explain cleaning)."""
    s0 = html.unescape(str(text))
    s1 = s0.lower()
    s2 = CURRENCY_RE.sub(" money ", URL_RE.sub(" url ", s1))
    s3 = NON_ALNUM_RE.sub(" ", s2)
    s4 = SPACE_RE.sub(" ", s3).strip()
    tokens = [w for w in s4.split() if len(w) >= 2]
    kept = [w for w in tokens if w not in STOP_WORDS]
    removed = [w for w in tokens if w in STOP_WORDS]
    return {
        "original": str(text),
        "decode_html": s0,
        "lowercase": s1,
        "replace_links_money": s2,
        "remove_punctuation": s3,
        "remove_whitespace": s4,
        "tokens_after_stopwords": kept,
        "stopwords_removed": removed,
    }


def tokenize_clean(text):
    return [w for w in preprocess_text(text).split() if len(w) >= 2 and w not in STOP_WORDS]


def build_vectorizer(n_train_docs):
    """TF-IDF on unigrams + bigrams. min_df=2 removes one-off typos on large data."""
    return TfidfVectorizer(
        preprocessor=preprocess_text,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if n_train_docs >= 500 else 1,
        max_features=5000,
        sublinear_tf=True,
    )


# ---------------------------------------------------------------------------
# CSV loading, column detection and validation
# ---------------------------------------------------------------------------
LABEL_NAME_HINTS = {"label", "v1", "class", "category", "target", "type"}
TEXT_NAME_HINTS = {"message", "text", "v2", "sms", "content", "body"}


def read_csv_bytes(raw_bytes):
    """Read CSV bytes, trying common encodings (Kaggle's spam.csv is latin-1).
    Also accepts the original tab-separated UCI file and header-less files."""
    if not raw_bytes or not raw_bytes.strip():
        raise UserError("The uploaded file is empty.")
    last_error = None
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            head_lines = [ln for ln in raw_bytes[:20000].decode(enc).splitlines() if ln.strip()][:30]
        except UnicodeDecodeError as exc:
            last_error = exc
            continue
        sep = "\t" if head_lines and sum("\t" in ln for ln in head_lines) >= 0.8 * len(head_lines) else ","
        first_cell = head_lines[0].split(sep)[0].strip().strip('"').lower() if head_lines else ""
        header = None if first_cell in VALID_LABELS else "infer"
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes), encoding=enc, sep=sep, header=header,
                             on_bad_lines="skip", quoting=3 if sep == "\t" else 0)
        except Exception as exc:  # pandas parser errors
            last_error = exc
            continue
        if header is None:
            df.columns = [f"column_{i + 1}" for i in range(df.shape[1])]
        df.columns = [str(c).strip() for c in df.columns]
        df = df.dropna(axis=1, how="all")          # drop completely empty columns
        if df.empty or df.shape[1] < 2:
            raise UserError("The CSV must contain at least two columns: a label column and a message column.")
        return df
    raise UserError(f"Could not read the file as CSV ({type(last_error).__name__}). "
                    "Please upload a valid comma-separated file.")


def detect_columns(df):
    """Guess which column holds ham/spam labels and which holds the SMS text."""
    label_col, best_frac = None, 0.0
    for col in df.columns:
        values = df[col].dropna().astype(str).str.strip().str.lower()
        if values.empty:
            continue
        frac = values.isin(VALID_LABELS).mean()
        bonus = 0.01 if col.lower() in LABEL_NAME_HINTS else 0.0
        if frac + bonus > best_frac:
            label_col, best_frac = col, frac + bonus
    if best_frac < 0.8:
        label_col = None

    text_col, best_len = None, 0.0
    for col in df.columns:
        if col == label_col:
            continue
        series = df[col]
        if series.notna().mean() < 0.5:
            continue
        lengths = series.dropna().astype(str).str.len()
        score = lengths.mean() + (5 if col.lower() in TEXT_NAME_HINTS else 0)
        if score > best_len:
            text_col, best_len = col, score
    if best_len < 5:
        text_col = None
    return label_col, text_col


def clean_dataset(raw, label_col, text_col):
    """Validate and clean the selected columns. Returns (clean_df, report)."""
    if label_col not in raw.columns:
        raise UserError(f"Label column '{label_col}' was not found in the file.")
    if text_col not in raw.columns:
        raise UserError(f"Text column '{text_col}' was not found in the file.")
    if label_col == text_col:
        raise UserError("The label column and the text column must be different.")

    df = raw[[label_col, text_col]].copy()
    df.columns = ["label", "message"]
    raw_rows = len(df)
    missing_label = int(df["label"].isna().sum())
    missing_text = int(df["message"].isna().sum())
    df = df.dropna(subset=["label", "message"])

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    empty_mask = df["message"] == ""
    empty_messages = int(empty_mask.sum())
    df = df[~empty_mask]

    bad_mask = ~df["label"].isin(VALID_LABELS)
    unexpected = int(bad_mask.sum())
    unexpected_examples = sorted(df.loc[bad_mask, "label"].unique().tolist())[:5]
    if len(df) == 0 or unexpected > 0.5 * len(df):
        raise UserError(
            "The label column does not contain the expected classes 'ham' and 'spam'"
            + (f" (found values such as: {', '.join(map(str, unexpected_examples))})." if unexpected_examples else "."),
            extra={"needs_mapping": True},
        )
    df = df[~bad_mask].reset_index(drop=True)

    duplicates = int(df.duplicated(subset=["message"]).sum())
    report = {
        "raw_rows": raw_rows,
        "valid_rows": len(df),
        "missing_label": missing_label,
        "missing_text": missing_text,
        "missing_values": missing_label + missing_text,
        "empty_messages": empty_messages,
        "unexpected_labels": unexpected,
        "unexpected_label_examples": [str(x) for x in unexpected_examples],
        "duplicates": duplicates,
    }
    return df, report


def validate_size(model_df):
    counts = model_df["label"].value_counts()
    if len(model_df) < MIN_MESSAGES:
        raise UserError(f"The dataset is too small: {len(model_df)} unique messages after cleaning. "
                        f"At least {MIN_MESSAGES} are needed to train and test the models.")
    for lab in VALID_LABELS:
        if counts.get(lab, 0) < MIN_PER_CLASS:
            raise UserError(f"The dataset needs at least {MIN_PER_CLASS} '{lab}' messages "
                            f"(found {int(counts.get(lab, 0))}).")


# ---------------------------------------------------------------------------
# Metrics helpers
# ---------------------------------------------------------------------------
def r4(x):
    return None if x is None else float(round(float(x), 4))


def thin_curve(fpr, tpr, max_points=200):
    if len(fpr) <= max_points:
        return fpr.tolist(), tpr.tolist()
    idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
    return fpr[idx].tolist(), tpr[idx].tolist()


def classification_metrics(y_true, y_pred, scores):
    """Metrics with 'spam' (1) as the positive class."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    out = {
        "accuracy": r4(accuracy_score(y_true, y_pred)),
        "precision": r4(precision_score(y_true, y_pred, zero_division=0)),
        "recall": r4(recall_score(y_true, y_pred, zero_division=0)),
        "f1": r4(f1_score(y_true, y_pred, zero_division=0)),
        "specificity": r4(tn / (tn + fp)) if (tn + fp) else None,
        "roc_auc": None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "roc": None,
        "test_size": int(len(y_true)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = r4(roc_auc_score(y_true, scores))
        fpr, tpr, _ = roc_curve(y_true, scores)
        f, t = thin_curve(fpr, tpr)
        out["roc"] = {"fpr": f, "tpr": t}
    return out


# ---------------------------------------------------------------------------
# Application state (models are kept in server memory)
# ---------------------------------------------------------------------------
class State:
    def __init__(self):
        self.lock = threading.RLock()
        self.clear()

    def clear(self):
        self.source_name = None
        self.is_sample = False
        self.loaded_at = None
        self.df = None              # all valid rows (duplicates kept) - overview, preview, EDA
        self.model_df = None        # de-duplicated rows used for modelling
        self.report = None
        self.columns_used = None
        self.pending_raw = None     # an upload waiting for manual column selection
        self.pending_name = None
        # split + features
        self.X_train_text = self.X_test_text = None
        self.y_train = self.y_test = None
        self.vectorizer = None
        self.Xtr = self.Xte = None
        # models
        self.svm = None
        self.svm_params = dict(DEFAULT_SVM)
        self.svm_result = None
        self.blr = None
        self.blr_params = dict(DEFAULT_BLR)
        self.blr_result = None
        self.nb_result = None
        self.kernel_results = None
        self.tuning_results = None
        self.cv_results = None
        self.boundary_cache = None
        # test-set outputs (used for PR curves, threshold explorer and CSV export)
        self.svm_scores = self.svm_pred = None
        self.blr_proba = None

    def require_data(self):
        if self.df is None:
            raise UserError("No dataset is loaded. Upload a CSV or reset to the sample dataset.", 409)

    def require_models(self):
        self.require_data()
        if self.svm is None or self.blr is None:
            raise UserError("The models are not trained yet. Load a dataset first.", 409)


STATE = State()


def api(fn):
    """Wrap an endpoint so errors always come back as friendly JSON."""
    @wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except UserError as exc:
            body = {"ok": False, "error": exc.message}
            body.update(exc.extra)
            return jsonify(body), exc.status
        except Exception as exc:  # unexpected problem: report, never crash
            app.logger.exception("Unexpected error")
            return jsonify({"ok": False, "error": f"Server error: {type(exc).__name__}: {exc}"}), 500
    return wrapper


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------
def load_dataset(raw, label_col, text_col, source_name, is_sample):
    """Validate -> clean -> split -> TF-IDF -> train SVM + BLR. Atomic: the old
    dataset stays active if anything fails."""
    df, report = clean_dataset(raw, label_col, text_col)
    model_df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)
    validate_size(model_df)

    y = (model_df["label"] == "spam").astype(int).values
    X_tr, X_te, y_tr, y_te = train_test_split(
        model_df["message"].values, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    vec = build_vectorizer(len(X_tr))
    Xtr = vec.fit_transform(X_tr)      # fit on TRAINING data only (no leakage)
    Xte = vec.transform(X_te)          # test data is only transformed
    if Xtr.shape[1] < 2:
        raise UserError("Too few distinct words were found to build TF-IDF features.")

    new = State()   # a new dataset starts from the default model settings
    new.source_name, new.is_sample = source_name, is_sample
    new.loaded_at = time.strftime("%Y-%m-%d %H:%M:%S")
    new.df, new.model_df, new.report = df, model_df, report
    new.columns_used = {"label": label_col, "text": text_col}
    new.X_train_text, new.X_test_text, new.y_train, new.y_test = X_tr, X_te, y_tr, y_te
    new.vectorizer, new.Xtr, new.Xte = vec, Xtr, Xte

    try:
        train_svm_into(new, new.svm_params)
    except UserError:
        raise
    except Exception:
        train_svm_into(new, dict(DEFAULT_SVM))
    train_blr_into(new, new.blr_params)
    train_nb_into(new)

    with STATE.lock:
        STATE.__dict__.update({k: v for k, v in new.__dict__.items() if k != "lock"})
        STATE.pending_raw = None
        STATE.pending_name = None
    return report


def make_svc(params):
    kw = {"kernel": params["kernel"], "C": float(params["C"]),
          "class_weight": "balanced" if params.get("balanced") else None}
    if params["kernel"] in ("rbf", "poly"):
        kw["gamma"] = params.get("gamma", "scale")
    if params["kernel"] == "poly":
        kw["degree"] = 2
        kw["coef0"] = 1.0
    return SVC(**kw)


def parse_svm_params(data):
    kernel = str(data.get("kernel", "linear")).lower()
    if kernel not in ("linear", "rbf", "poly"):
        raise UserError("Kernel must be one of: linear, rbf, poly.")
    try:
        C = float(data.get("C", 1.0))
    except (TypeError, ValueError):
        raise UserError("C must be a number.")
    if not (0.001 <= C <= 1000):
        raise UserError("C must be between 0.001 and 1000.")
    gamma = data.get("gamma", "scale")
    if isinstance(gamma, str) and gamma.strip().lower() in ("scale", "auto"):
        gamma = gamma.strip().lower()
    else:
        try:
            gamma = float(gamma)
        except (TypeError, ValueError):
            raise UserError("Gamma must be 'scale', 'auto' or a positive number.")
        if not (0 < gamma <= 100):
            raise UserError("Gamma must be greater than 0 and at most 100.")
    balanced = bool(data.get("balanced", False))
    return {"kernel": kernel, "C": C, "gamma": gamma, "balanced": balanced}


def train_svm_into(st, params):
    model = make_svc(params)
    t0 = time.perf_counter()
    model.fit(st.Xtr, st.y_train)
    train_time = time.perf_counter() - t0
    scores = model.decision_function(st.Xte)
    pred = model.predict(st.Xte)
    metrics = classification_metrics(st.y_test, pred, scores)

    result = {
        "params": params,
        "train_time_sec": r4(train_time),
        "n_train": int(st.Xtr.shape[0]),
        "n_test": int(st.Xte.shape[0]),
        "n_features": int(st.Xtr.shape[1]),
        "support_vectors": {"ham": int(model.n_support_[0]), "spam": int(model.n_support_[1]),
                            "total": int(model.n_support_.sum())},
        "support_vector_pct": r4(100 * model.n_support_.sum() / st.Xtr.shape[0]),
        "metrics": metrics,
        "top_terms": None,
        "misclassified": misclassified_examples(st, pred, scores, "decision score"),
    }
    if params["kernel"] == "linear":
        coef = model.coef_
        coef = np.asarray(coef.todense()).ravel() if sparse.issparse(coef) else np.ravel(coef)
        names = st.vectorizer.get_feature_names_out()
        order = np.argsort(coef)
        result["top_terms"] = {
            "spam": [{"term": names[i], "weight": r4(coef[i])} for i in order[::-1][:15]],
            "ham": [{"term": names[i], "weight": r4(coef[i])} for i in order[:15]],
            "intercept": r4(model.intercept_[0]),
        }
    st.svm, st.svm_params, st.svm_result = model, params, result
    st.svm_scores, st.svm_pred = scores, pred
    st.boundary_cache = None
    st.cv_results = None          # cross-validation must be re-run for new settings
    return result


def misclassified_examples(st, pred, scores, score_name, limit=6):
    out = {"false_positives": [], "false_negatives": []}
    for i in np.where((pred == 1) & (st.y_test == 0))[0][:limit]:
        out["false_positives"].append({"message": str(st.X_test_text[i]), "score": r4(scores[i])})
    for i in np.where((pred == 0) & (st.y_test == 1))[0][:limit]:
        out["false_negatives"].append({"message": str(st.X_test_text[i]), "score": r4(scores[i])})
    out["score_name"] = score_name
    return out




# ---- Bayesian Logistic Regression (MAP + Laplace approximation) -----------
def parse_blr_params(data):
    try:
        pv = float(data.get("prior_variance", DEFAULT_BLR["prior_variance"]))
        k = int(data.get("n_features", DEFAULT_BLR["n_features"]))
    except (TypeError, ValueError):
        raise UserError("Prior variance must be a number and feature count must be an integer.")
    if not (0.01 <= pv <= 1000):
        raise UserError("Prior variance must be between 0.01 and 1000.")
    if not (50 <= k <= 2000):
        raise UserError("Number of features must be between 50 and 2000.")
    return {"prior_variance": pv, "n_features": k}


def add_bias(X):
    return sparse.hstack([X, np.ones((X.shape[0], 1))], format="csr")


def fit_blr(Xtr, ytr, params):
    """
    Model:      p(spam | x, w) = sigmoid(w^T x)
    Prior:      w ~ N(0, s2 * I)                (s2 = prior variance)
    Posterior:  p(w | D) is proportional to likelihood * prior (no closed form)
    MAP:        w_map = argmin  -log p(D|w) + ||w||^2 / (2 s2)
    Laplace:    p(w | D) ~ N(w_map, S),  S^-1 = H = X^T R X + I / s2,  R = diag(p(1-p))
    Prediction: p(spam | x) ~ sigmoid( kappa(var_a) * mu_a ),
                mu_a = w_map^T x,  var_a = x^T S x,  kappa = (1 + pi var_a / 8)^-1/2
    The full covariance S is (k+1)x(k+1), so we keep the k most informative
    TF-IDF features (chi-square test on the TRAINING data only).
    """
    k = min(params["n_features"], Xtr.shape[1])
    selector = SelectKBest(chi2, k=k).fit(Xtr, ytr)
    Xs = add_bias(selector.transform(Xtr))
    y = ytr.astype(float)

    prior_prec = np.full(Xs.shape[1], 1.0 / params["prior_variance"])
    prior_prec[-1] = 1.0 / BLR_BIAS_PRIOR_VARIANCE

    def neg_log_posterior(w):
        a = Xs @ w
        nll = np.sum(np.logaddexp(0.0, a) - y * a)
        value = nll + 0.5 * np.sum(prior_prec * w * w)
        grad = Xs.T @ (expit(a) - y) + prior_prec * w
        return value, grad

    res = minimize(neg_log_posterior, np.zeros(Xs.shape[1]), jac=True, method="L-BFGS-B",
                   options={"maxiter": 2000})
    w_map = res.x
    # Hessian of the negative log posterior at w_map  ->  Laplace covariance
    p_train = expit(Xs @ w_map)
    r = p_train * (1 - p_train)
    H = (Xs.T @ Xs.multiply(r[:, None]).tocsr()).toarray() + np.diag(prior_prec)
    Sigma = cho_solve(cho_factor(H), np.eye(H.shape[0]))
    return {"selector": selector, "w": w_map, "Sigma": Sigma, "k": k, "opt": res}


def blr_predict(model, X):
    """Returns (mu, var, p_map, p_bayes) for raw TF-IDF rows X."""
    Xb = add_bias(model["selector"].transform(X))
    return blr_predict_arrays(Xb, model["w"], model["Sigma"])


def train_blr_into(st, params):
    t0 = time.perf_counter()
    model = fit_blr(st.Xtr, st.y_train, params)
    train_time = time.perf_counter() - t0
    res, w_map, Sigma, selector, k = model["opt"], model["w"], model["Sigma"], model["selector"], model["k"]

    mu, var, p_map, p_bayes = blr_predict(model, st.Xte)
    pred = (p_bayes >= 0.5).astype(int)
    metrics = classification_metrics(st.y_test, pred, p_bayes)
    metrics_map = classification_metrics(st.y_test, (p_map >= 0.5).astype(int), p_map)

    names = st.vectorizer.get_feature_names_out()[selector.get_support()]
    w = w_map[:-1]
    std = np.sqrt(np.clip(np.diag(Sigma)[:-1], 0, None))
    order = np.argsort(w)

    def term_rows(idx):
        return [{"term": names[i], "weight": r4(w[i]), "posterior_std": r4(std[i]),
                 "ci_low": r4(w[i] - 1.96 * std[i]), "ci_high": r4(w[i] + 1.96 * std[i])} for i in idx]

    eps = 1e-12
    result = {
        "params": params,
        "n_features_used": int(k),
        "converged": bool(res.success),
        "iterations": int(res.nit),
        "train_time_sec": r4(train_time),
        "neg_log_posterior": r4(res.fun),
        "bias": r4(w_map[-1]),
        "bias_std": r4(np.sqrt(Sigma[-1, -1])),
        "metrics": metrics,
        "metrics_map": metrics_map,
        "log_loss_map": r4(log_loss(st.y_test, np.clip(p_map, eps, 1 - eps), labels=[0, 1])),
        "log_loss_bayes": r4(log_loss(st.y_test, np.clip(p_bayes, eps, 1 - eps), labels=[0, 1])),
        "mean_predictive_std": r4(np.mean(np.sqrt(var))),
        "top_terms": {"spam": term_rows(order[::-1][:12]), "ham": term_rows(order[:12])},
        "misclassified": misclassified_examples(st, pred, p_bayes, "P(spam)"),
        "examples": [
            {"message": str(st.X_test_text[i]), "actual": "spam" if st.y_test[i] else "ham",
             "p_map": r4(p_map[i]), "p_bayes": r4(p_bayes[i]), "activation_std": r4(np.sqrt(var[i]))}
            for i in np.argsort(-np.abs(p_map - p_bayes))[:6]
        ],
    }
    st.blr = {"selector": selector, "w": w_map, "Sigma": Sigma}
    st.blr_params, st.blr_result = params, result
    st.blr_proba = p_bayes
    st.cv_results = None
    return result


def train_nb_into(st):
    """Multinomial Naive Bayes: a probabilistic GENERATIVE reference model.
    It models p(words | class) and p(class) and applies Bayes' theorem."""
    model = MultinomialNB(alpha=1.0)
    t0 = time.perf_counter()
    model.fit(st.Xtr, st.y_train)
    t = time.perf_counter() - t0
    proba = model.predict_proba(st.Xte)[:, 1]
    st.nb_result = {
        "metrics": classification_metrics(st.y_test, (proba >= 0.5).astype(int), proba),
        "train_time_sec": r4(t), "alpha": 1.0,
        "class_prior": {"ham": r4(np.exp(model.class_log_prior_[0])), "spam": r4(np.exp(model.class_log_prior_[1]))},
    }
    return st.nb_result


def blr_predict_arrays(X_with_bias, w_map, Sigma):
    mu = X_with_bias @ w_map
    var = np.asarray(X_with_bias.multiply(X_with_bias @ Sigma).sum(axis=1)).ravel()
    var = np.clip(var, 0, None)
    kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8.0)
    return mu, var, expit(mu), expit(kappa * mu)


# ---------------------------------------------------------------------------
# Regression helpers (Linear Regression practical - separate from spam task)
# ---------------------------------------------------------------------------
def regression_frame(st):
    msgs = st.model_df["message"].astype(str)
    feats = pd.DataFrame({
        "word_count": msgs.str.split().str.len(),
        "avg_word_length": msgs.apply(lambda m: np.mean([len(w) for w in m.split()]) if m.split() else 0.0),
        "digit_count": msgs.str.count(r"\d"),
        "uppercase_count": msgs.str.count(r"[A-Z]"),
        "punctuation_count": msgs.str.count(r"[^\w\s]"),
    })
    feats["char_length"] = msgs.str.len()
    feats["message"] = msgs
    return feats


def reg_metrics(y_true, y_pred):
    return {"mae": r4(mean_absolute_error(y_true, y_pred)),
            "rmse": r4(np.sqrt(mean_squared_error(y_true, y_pred))),
            "r2": r4(r2_score(y_true, y_pred))}


def subsample_idx(n, k, seed=0):
    if n <= k:
        return np.arange(n)
    return np.sort(np.random.RandomState(seed).choice(n, k, replace=False))


# ---------------------------------------------------------------------------
# Routes - pages
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html", title=PROJECT_TITLE, dataset_url=DATASET_URL)


# ---------------------------------------------------------------------------
# Routes - dataset
# ---------------------------------------------------------------------------
@app.route("/api/status")
@api
def api_status():
    st = STATE
    return jsonify({
        "ok": True,
        "loaded": st.df is not None,
        "source_name": st.source_name,
        "is_sample": st.is_sample,
        "loaded_at": st.loaded_at,
        "columns_used": st.columns_used,
        "trained": st.svm is not None and st.blr is not None,
        "svm_params": st.svm_params,
        "blr_params": st.blr_params,
        "project_title": PROJECT_TITLE,
        "dataset_url": DATASET_URL,
    })


@app.route("/api/upload", methods=["POST"])
@api
def api_upload():
    label_col = (request.form.get("label_col") or "").strip() or None
    text_col = (request.form.get("text_col") or "").strip() or None
    file = request.files.get("file")

    if file and file.filename:
        name = os.path.basename(file.filename)
        if not name.lower().endswith((".csv", ".txt", ".tsv")):
            raise UserError("Please upload a .csv, .txt, or .tsv file (the Kaggle file is called spam.csv).")
        raw = read_csv_bytes(file.read())
        with STATE.lock:
            STATE.pending_raw, STATE.pending_name = raw, name
    else:
        if STATE.pending_raw is None:
            raise UserError("Choose a CSV file to upload.")
        raw, name = STATE.pending_raw, STATE.pending_name

    auto = False
    if not (label_col and text_col):
        d_label, d_text = detect_columns(raw)
        label_col = label_col or d_label
        text_col = text_col or d_text
        auto = True
        if not (label_col and text_col):
            raise UserError(
                "Could not automatically detect the label and message columns. "
                "Please select them manually below.", 422,
                extra={"needs_mapping": True, "columns": list(map(str, raw.columns)),
                       "detected": {"label": label_col, "text": text_col},
                       "sample_rows": raw.head(5).astype(str).values.tolist()})
    try:
        report = load_dataset(raw, label_col, text_col, name, is_sample=False)
    except UserError as exc:
        exc.extra.setdefault("columns", list(map(str, raw.columns)))
        exc.extra.setdefault("needs_mapping", True)
        raise
    return jsonify({"ok": True, "message": "Dataset uploaded successfully",
                    "auto_detected": auto, "columns_used": {"label": label_col, "text": text_col},
                    "report": report, "source_name": name})


@app.route("/api/reset", methods=["POST"])
@api
def api_reset():
    if not os.path.exists(SAMPLE_PATH):
        raise UserError("The sample dataset file data/sample_sms.csv is missing.", 500)
    with open(SAMPLE_PATH, "rb") as f:
        raw = read_csv_bytes(f.read())
    label_col, text_col = detect_columns(raw)
    report = load_dataset(raw, label_col, text_col, "sample_sms.csv", is_sample=True)
    return jsonify({"ok": True, "message": "Sample dataset loaded", "report": report})


@app.route("/api/overview")
@api
def api_overview():
    st = STATE
    st.require_data()
    df, rep = st.df, st.report
    total = len(df)
    spam = int((df["label"] == "spam").sum())
    ham = total - spam
    return jsonify({
        "ok": True,
        "source_name": st.source_name,
        "is_sample": st.is_sample,
        "total": total,
        "spam": spam,
        "ham": ham,
        "spam_pct": r4(100 * spam / total),
        "ham_pct": r4(100 * ham / total),
        "duplicates": rep["duplicates"],
        "missing_values": rep["missing_values"],
        "unique_messages": int(len(st.model_df)),
        "raw_rows": rep["raw_rows"],
        "removed_rows": rep["raw_rows"] - total,
        "report": rep,
        "train_size": int(len(st.y_train)),
        "test_size": int(len(st.y_test)),
        "train_spam": int(st.y_train.sum()),
        "test_spam": int(st.y_test.sum()),
    })


@app.route("/api/preview")
@api
def api_preview():
    st = STATE
    st.require_data()
    try:
        page = max(1, int(request.args.get("page", 1)))
        per_page = min(100, max(5, int(request.args.get("per_page", 25))))
    except ValueError:
        raise UserError("Page numbers must be integers.")
    search = (request.args.get("search") or "").strip().lower()
    label = (request.args.get("label") or "all").lower()

    df = st.df.reset_index().rename(columns={"index": "row"})
    if label in VALID_LABELS:
        df = df[df["label"] == label]
    if search:
        df = df[df["message"].str.lower().str.contains(search, regex=False)]
    total = len(df)
    pages = max(1, int(np.ceil(total / per_page)))
    page = min(page, pages)
    chunk = df.iloc[(page - 1) * per_page: page * per_page]
    return jsonify({"ok": True, "total": total, "page": page, "pages": pages, "per_page": per_page,
                    "rows": [{"row": int(r.row) + 1, "label": r.label, "message": r.message}
                             for r in chunk.itertuples()]})


@app.route("/api/preprocess")
@api
def api_preprocess():
    st = STATE
    st.require_data()
    df = st.df
    examples, used = [], set()

    def pick(mask):
        for i in df.index[mask]:
            if i not in used:
                used.add(i)
                return i
        return None

    msg = df["message"]
    candidates = [
        msg.str.contains("&lt;|&gt;|&amp;", regex=True),
        (df["label"] == "spam") & msg.str.contains(r"[!£$€]|http|www", regex=True),
        (df["label"] == "ham") & msg.str.contains(r"[A-Z].*[,.?!']", regex=True),
        (df["label"] == "spam"),
        (df["label"] == "ham"),
    ]
    for mask in candidates:
        i = pick(mask)
        if i is not None:
            step = preprocessing_steps(df.at[i, "message"])
            step["label"] = df.at[i, "label"]
            examples.append(step)
        if len(examples) == 3:
            break
    examples.sort(key=lambda e: e["label"] != "spam")  # show a spam example first

    sample = df["message"].head(2000)
    raw_tokens = sample.str.split().str.len().mean()
    clean_tokens = sample.apply(lambda m: len(tokenize_clean(m))).mean()
    return jsonify({"ok": True, "examples": examples,
                    "avg_tokens_raw": r4(raw_tokens), "avg_tokens_clean": r4(clean_tokens),
                    "n_stopwords": len(STOP_WORDS)})


@app.route("/api/eda")
@api
def api_eda():
    st = STATE
    st.require_data()
    df = st.df
    lengths = df["message"].str.len()
    words = df["message"].str.split().str.len()
    is_spam = df["label"] == "spam"

    def top_words(series, n=15):
        c = Counter()
        for m in series:
            c.update(tokenize_clean(m))
        return [{"word": w, "count": int(k)} for w, k in c.most_common(n)]

    def stats(mask):
        return {"count": int(mask.sum()),
                "avg_chars": r4(lengths[mask].mean()), "median_chars": r4(lengths[mask].median()),
                "avg_words": r4(words[mask].mean()), "max_chars": int(lengths[mask].max()),
                "min_chars": int(lengths[mask].min())}

    common_lengths = lengths.value_counts().head(5)
    return jsonify({
        "ok": True,
        "counts": {"ham": int((~is_spam).sum()), "spam": int(is_spam.sum())},
        "lengths": {"ham": lengths[~is_spam].astype(int).tolist(), "spam": lengths[is_spam].astype(int).tolist()},
        "stats": {"ham": stats(~is_spam), "spam": stats(is_spam), "all": stats(lengths > -1)},
        "most_common_lengths": [{"length": int(k), "count": int(v)} for k, v in common_lengths.items()],
        "top_words": {"all": top_words(df["message"]), "ham": top_words(df.loc[~is_spam, "message"]),
                      "spam": top_words(df.loc[is_spam, "message"])},
        "imbalance_ratio": r4((~is_spam).sum() / max(1, is_spam.sum())),
    })


@app.route("/api/tfidf")
@api
def api_tfidf():
    st = STATE
    st.require_data()
    vec, Xtr, Xte = st.vectorizer, st.Xtr, st.Xte
    names = vec.get_feature_names_out()
    idf = vec.idf_
    n_feat = len(names)
    nnz = Xtr.nnz
    total_cells = Xtr.shape[0] * Xtr.shape[1]

    spam_mean = np.asarray(Xtr[st.y_train == 1].mean(axis=0)).ravel()
    ham_mean = np.asarray(Xtr[st.y_train == 0].mean(axis=0)).ravel()
    sample_idx = np.linspace(0, n_feat - 1, min(40, n_feat)).astype(int)

    # one training message with its non-zero weights
    lens = np.diff(Xtr.indptr)
    ex_i = int(np.argmax((lens >= 5) & (lens <= 12))) if np.any((lens >= 5) & (lens <= 12)) else 0
    row = Xtr[ex_i]
    ex_terms = sorted(zip(row.indices, row.data), key=lambda t: -t[1])[:12]
    bigrams = int(sum(1 for n in names if " " in n))
    return jsonify({
        "ok": True,
        "n_features": int(n_feat),
        "n_unigrams": int(n_feat - bigrams),
        "n_bigrams": bigrams,
        "train_shape": [int(Xtr.shape[0]), int(Xtr.shape[1])],
        "test_shape": [int(Xte.shape[0]), int(Xte.shape[1])],
        "nonzero_train": int(nnz),
        "sparsity_pct": r4(100 * (1 - nnz / total_cells)),
        "avg_nonzero_per_message": r4(nnz / Xtr.shape[0]),
        "settings": {"ngram_range": "1-2 (single words and two-word phrases)",
                     "min_df": vec.min_df, "max_features": vec.max_features,
                     "stop_words": "English (scikit-learn list)", "sublinear_tf": True},
        "sample_vocabulary": [names[i] for i in sample_idx],
        "top_spam_terms": [{"term": names[i], "score": r4(spam_mean[i])} for i in np.argsort(-spam_mean)[:15]],
        "top_ham_terms": [{"term": names[i], "score": r4(ham_mean[i])} for i in np.argsort(-ham_mean)[:15]],
        "lowest_idf": [{"term": names[i], "idf": r4(idf[i])} for i in np.argsort(idf)[:10]],
        "highest_idf": [{"term": names[i], "idf": r4(idf[i])} for i in np.argsort(-idf)[:10]],
        "example": {"message": str(st.X_train_text[ex_i]),
                    "label": "spam" if st.y_train[ex_i] else "ham",
                    "terms": [{"term": names[j], "weight": r4(v)} for j, v in ex_terms]},
    })


# ---------------------------------------------------------------------------
# Routes - SVM
# ---------------------------------------------------------------------------
@app.route("/api/train-svm", methods=["GET", "POST"])
@api
def api_train_svm():
    st = STATE
    st.require_data()
    if request.method == "POST":
        params = parse_svm_params(request.get_json(silent=True) or {})
        with st.lock:
            try:
                result = train_svm_into(st, params)
            except Exception as exc:
                raise UserError(f"SVM training failed: {exc}", 500)
    else:
        st.require_models()
        result = st.svm_result
    return jsonify({"ok": True, "result": result})


@app.route("/api/svm-boundary")
@api
def api_svm_boundary():
    """2-D picture of hyperplane, margin and support vectors. The TF-IDF matrix is
    projected to 2 dimensions (Truncated SVD) and a separate SVM with the same
    settings is trained on that projection ONLY for visualisation."""
    st = STATE
    st.require_models()
    if st.boundary_cache is not None:
        return jsonify(st.boundary_cache)
    params = st.svm_params
    svd = TruncatedSVD(n_components=2, random_state=RANDOM_STATE)
    Z = svd.fit_transform(st.Xtr)
    Z = StandardScaler().fit_transform(Z)
    idx = subsample_idx(len(Z), 1200, seed=RANDOM_STATE)
    Zs, ys = Z[idx], st.y_train[idx]
    model2 = make_svc(params).fit(Zs, ys)
    pad = 0.5
    x0, x1 = np.percentile(Zs[:, 0], [1, 99])
    y0, y1 = np.percentile(Zs[:, 1], [1, 99])
    gx = np.linspace(x0 - pad, x1 + pad, 70)
    gy = np.linspace(y0 - pad, y1 + pad, 70)
    xx, yy = np.meshgrid(gx, gy)
    zz = model2.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    sv_mask = np.zeros(len(Zs), dtype=bool)
    sv_mask[model2.support_] = True
    show = subsample_idx(len(Zs), 700, seed=1)
    payload = {
        "ok": True,
        "params": params,
        "grid_x": np.round(gx, 4).tolist(), "grid_y": np.round(gy, 4).tolist(),
        "z": np.round(zz, 4).tolist(),
        "points": {"x": np.round(Zs[show, 0], 4).tolist(), "y": np.round(Zs[show, 1], 4).tolist(),
                   "label": ys[show].astype(int).tolist(), "is_sv": sv_mask[show].tolist()},
        "accuracy_2d": r4(model2.score(Zs, ys)),
        "n_points": int(len(Zs)),
        "n_sv_2d": int(model2.support_.size),
        "explained_variance_pct": r4(100 * svd.explained_variance_ratio_.sum()),
    }
    st.boundary_cache = payload
    return jsonify(payload)


@app.route("/api/kernel-experiment", methods=["GET", "POST"])
@api
def api_kernel_experiment():
    st = STATE
    st.require_models()
    if request.method == "GET":
        if st.kernel_results is None:
            return jsonify({"ok": True, "available": False})
        return jsonify({"ok": True, "available": True, **st.kernel_results})
    data = request.get_json(silent=True) or {}
    base = parse_svm_params({**st.svm_params, **data})
    rows = []
    for kernel in ("linear", "rbf", "poly"):
        p = dict(base, kernel=kernel, gamma=base["gamma"] if kernel != "linear" else "scale")
        model = make_svc(p)
        t0 = time.perf_counter()
        model.fit(st.Xtr, st.y_train)
        t = time.perf_counter() - t0
        scores = model.decision_function(st.Xte)
        m = classification_metrics(st.y_test, model.predict(st.Xte), scores)
        rows.append({"kernel": kernel,
                     "settings": f"C={p['C']}" + (f", gamma={p['gamma']}" if kernel != "linear" else "")
                     + (", degree=2, coef0=1" if kernel == "poly" else "")
                     + (", balanced weights" if p["balanced"] else ""),
                     "accuracy": m["accuracy"], "precision": m["precision"], "recall": m["recall"],
                     "f1": m["f1"], "roc_auc": m["roc_auc"], "train_time_sec": r4(t),
                     "support_vectors": int(model.n_support_.sum())})
    st.kernel_results = {"rows": rows, "n_train": int(st.Xtr.shape[0]), "n_test": int(st.Xte.shape[0])}
    return jsonify({"ok": True, "available": True, **st.kernel_results})


@app.route("/api/tune-svm", methods=["GET", "POST"])
@api
def api_tune_svm():
    """Small grid search with 3-fold cross-validation on the TRAINING set only.
    TF-IDF is inside the pipeline so it is re-fitted on each fold (no leakage)."""
    st = STATE
    st.require_models()
    if request.method == "GET":
        if st.tuning_results is None:
            return jsonify({"ok": True, "available": False})
        return jsonify({"ok": True, "available": True, **st.tuning_results})
    data = request.get_json(silent=True) or {}
    balanced = bool(data.get("balanced", st.svm_params.get("balanced", False)))
    pipe = Pipeline([("tfidf", build_vectorizer(len(st.X_train_text))),
                     ("svm", SVC(class_weight="balanced" if balanced else None))])
    grid = [
        {"svm__kernel": ["linear"], "svm__C": [0.1, 1, 10]},
        {"svm__kernel": ["rbf"], "svm__C": [1, 10], "svm__gamma": ["scale", 0.1]},
        {"svm__kernel": ["poly"], "svm__C": [1, 10], "svm__degree": [2], "svm__coef0": [1.0]},
    ]
    n_splits = 3 if min(np.bincount(st.y_train)) >= 3 else 2
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    t0 = time.perf_counter()
    search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1, refit=False)
    search.fit(st.X_train_text, st.y_train)
    elapsed = time.perf_counter() - t0

    res = search.cv_results_
    rows = []
    for i, p in enumerate(res["params"]):
        rows.append({"kernel": p["svm__kernel"], "C": p["svm__C"],
                     "gamma": p.get("svm__gamma", "-") if p["svm__kernel"] != "linear" else "-",
                     "mean_f1": r4(res["mean_test_score"][i]), "std_f1": r4(res["std_test_score"][i]),
                     "rank": int(res["rank_test_score"][i])})
    rows.sort(key=lambda r: r["rank"])
    best = search.best_params_
    best_params = {"kernel": best["svm__kernel"], "C": float(best["svm__C"]),
                   "gamma": best.get("svm__gamma", "scale"), "balanced": balanced}
    apply = bool(data.get("apply", True))
    result = None
    if apply:
        with st.lock:
            result = train_svm_into(st, best_params)
    st.tuning_results = {"rows": rows, "best_params": best_params, "cv_folds": n_splits,
                         "best_cv_f1": r4(search.best_score_), "time_sec": r4(elapsed),
                         "applied": apply}
    return jsonify({"ok": True, "available": True, **st.tuning_results, "result": result})


# ---------------------------------------------------------------------------
# Routes - Bayesian Logistic Regression
# ---------------------------------------------------------------------------
@app.route("/api/train-bayesian-logistic", methods=["GET", "POST"])
@api
def api_train_blr():
    st = STATE
    st.require_data()
    if request.method == "POST":
        params = parse_blr_params(request.get_json(silent=True) or {})
        with st.lock:
            try:
                result = train_blr_into(st, params)
            except np.linalg.LinAlgError:
                raise UserError("The Hessian could not be inverted. Try a smaller prior variance.", 500)
    else:
        st.require_models()
        result = st.blr_result
    return jsonify({"ok": True, "result": result})


# ---------------------------------------------------------------------------
# Routes - evaluation
# ---------------------------------------------------------------------------
@app.route("/api/metrics")
@api
def api_metrics():
    st = STATE
    st.require_models()
    yt = st.y_test
    majority = max(yt.mean(), 1 - yt.mean())
    return jsonify({"ok": True,
                    "svm": st.svm_result["metrics"], "svm_params": st.svm_params,
                    "blr": st.blr_result["metrics"], "blr_params": st.blr_params,
                    "test_size": int(len(yt)), "test_spam": int(yt.sum()), "test_ham": int(len(yt) - yt.sum()),
                    "majority_baseline_accuracy": r4(majority)})


@app.route("/api/compare-models")
@api
def api_compare():
    st = STATE
    st.require_models()
    s, b = st.svm_result["metrics"], st.blr_result["metrics"]
    keys = ["accuracy", "precision", "recall", "f1", "specificity", "roc_auc"]
    rows = [{"metric": k, "svm": s[k], "blr": b[k]} for k in keys]
    yt = st.y_test

    def pr_curve(scores):
        prec, rec, _ = precision_recall_curve(yt, scores)
        r_, p_ = thin_curve(rec, prec)
        return {"recall": r_, "precision": p_, "average_precision": r4(average_precision_score(yt, scores))}

    thresholds = []
    for thr in np.round(np.arange(0.05, 0.96, 0.05), 2):
        pred = (st.blr_proba >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()
        thresholds.append({"threshold": float(thr), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                           "precision": r4(precision_score(yt, pred, zero_division=0)),
                           "recall": r4(recall_score(yt, pred, zero_division=0)),
                           "f1": r4(f1_score(yt, pred, zero_division=0)),
                           "accuracy": r4(accuracy_score(yt, pred))})
    both = len(np.unique(yt)) == 2
    return jsonify({"ok": True, "rows": rows,
                    "pr": {"svm": pr_curve(st.svm_scores), "blr": pr_curve(st.blr_proba)} if both else None,
                    "thresholds": thresholds,
                    "nb": st.nb_result,
                    "svm": {"metrics": s, "params": st.svm_params, "train_time_sec": st.svm_result["train_time_sec"],
                            "roc_auc_source": "SVM decision function"},
                    "blr": {"metrics": b, "params": st.blr_params, "train_time_sec": st.blr_result["train_time_sec"],
                            "roc_auc_source": "Bayesian predictive probability"},
                    "test_size": int(len(yt)), "test_spam": int(yt.sum()),
                    "majority_baseline_accuracy": r4(max(yt.mean(), 1 - yt.mean()))})


@app.route("/api/confusion-matrix")
@api
def api_confusion():
    st = STATE
    st.require_models()
    return jsonify({"ok": True,
                    "svm": st.svm_result["metrics"]["confusion"],
                    "blr": st.blr_result["metrics"]["confusion"],
                    "svm_misclassified": st.svm_result["misclassified"],
                    "blr_misclassified": st.blr_result["misclassified"],
                    "test_size": int(len(st.y_test))})


@app.route("/api/cross-validate", methods=["GET", "POST"])
@api
def api_cross_validate():
    """Stratified k-fold cross-validation of BOTH models on the de-duplicated data.
    For every fold the TF-IDF vectorizer and the models are fitted on that fold's
    training part only, so the estimate is leakage-free. This checks how much the
    single train/test split result could vary."""
    st = STATE
    st.require_models()
    if request.method == "GET":
        if st.cv_results is None:
            return jsonify({"ok": True, "available": False})
        return jsonify({"ok": True, "available": True, **st.cv_results})

    texts = st.model_df["message"].values
    y = (st.model_df["label"] == "spam").astype(int).values
    k = int(min(5, np.bincount(y).min()))
    if k < 2:
        raise UserError("Not enough spam messages for cross-validation.")
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    folds = {"svm": [], "blr": []}
    t0 = time.perf_counter()
    for tr, te in cv.split(texts, y):
        vec = build_vectorizer(len(tr))
        Xtr = vec.fit_transform(texts[tr])
        Xte = vec.transform(texts[te])
        svm = make_svc(st.svm_params).fit(Xtr, y[tr])
        sc = svm.decision_function(Xte)
        m = classification_metrics(y[te], svm.predict(Xte), sc)
        folds["svm"].append({key: m[key] for key in keys})
        blr = fit_blr(Xtr, y[tr], st.blr_params)
        p = blr_predict(blr, Xte)[3]
        m = classification_metrics(y[te], (p >= 0.5).astype(int), p)
        folds["blr"].append({key: m[key] for key in keys})
    elapsed = time.perf_counter() - t0

    def summary(rows):
        return {key: {"mean": r4(np.mean([r[key] for r in rows if r[key] is not None])),
                      "std": r4(np.std([r[key] for r in rows if r[key] is not None]))} for key in keys}

    st.cv_results = {"k": k, "n_messages": int(len(y)), "time_sec": r4(elapsed),
                     "svm_params": dict(st.svm_params), "blr_params": dict(st.blr_params),
                     "folds": folds, "summary": {"svm": summary(folds["svm"]), "blr": summary(folds["blr"])}}
    return jsonify({"ok": True, "available": True, **st.cv_results})


def strip_curves(metrics):
    return {k: v for k, v in metrics.items() if k != "roc"}


@app.route("/api/export/predictions")
@api
def api_export_predictions():
    """CSV of every test-set message with both models' outputs (proof of real predictions)."""
    st = STATE
    st.require_models()
    out = pd.DataFrame({
        "message": st.X_test_text,
        "actual_label": np.where(st.y_test == 1, "spam", "ham"),
        "svm_prediction": np.where(st.svm_pred == 1, "spam", "ham"),
        "svm_decision_score": np.round(st.svm_scores, 4),
        "bayesian_lr_prediction": np.where(st.blr_proba >= 0.5, "spam", "ham"),
        "bayesian_lr_p_spam": np.round(st.blr_proba, 4),
    })
    out["svm_correct"] = out["svm_prediction"] == out["actual_label"]
    out["bayesian_lr_correct"] = out["bayesian_lr_prediction"] == out["actual_label"]
    csv = out.to_csv(index=False)
    return Response(csv, mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=test_set_predictions.csv"})


@app.route("/api/export/summary")
@api
def api_export_summary():
    """JSON file with the complete measured results of the current run."""
    st = STATE
    st.require_models()
    rep_ = st.report
    summary = {
        "project_title": PROJECT_TITLE,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {"source_file": st.source_name, "is_sample": st.is_sample, "dataset_url": DATASET_URL,
                    "columns_used": st.columns_used, "validation_report": rep_,
                    "total_valid_messages": int(len(st.df)),
                    "spam": int((st.df["label"] == "spam").sum()), "ham": int((st.df["label"] == "ham").sum()),
                    "unique_messages_for_modelling": int(len(st.model_df))},
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE, "stratified": True,
                  "train_messages": int(len(st.y_train)), "test_messages": int(len(st.y_test)),
                  "test_spam": int(st.y_test.sum())},
        "tfidf": {"n_features": int(st.Xtr.shape[1]), "train_shape": list(st.Xtr.shape)},
        "svm": {"params": st.svm_params, "metrics": strip_curves(st.svm_result["metrics"]),
                "support_vectors": st.svm_result["support_vectors"]},
        "bayesian_logistic_regression": {"params": st.blr_params,
                                         "n_features_used": st.blr_result["n_features_used"],
                                         "metrics": strip_curves(st.blr_result["metrics"]),
                                         "log_loss_map": st.blr_result["log_loss_map"],
                                         "log_loss_bayes": st.blr_result["log_loss_bayes"]},
        "naive_bayes_reference": {"metrics": strip_curves(st.nb_result["metrics"])},
        "kernel_experiment": st.kernel_results,
        "grid_search": st.tuning_results,
        "cross_validation": ({k: v for k, v in st.cv_results.items() if k != "folds"} if st.cv_results else None),
        "findings": build_findings(st),
    }
    return Response(json.dumps(summary, indent=2), mimetype="application/json",
                    headers={"Content-Disposition": "attachment; filename=results_summary.json"})


# ---------------------------------------------------------------------------
# Routes - live prediction
# ---------------------------------------------------------------------------
@app.route("/api/predict", methods=["POST"])
@api
def api_predict():
    st = STATE
    st.require_models()
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        raise UserError("Type an SMS message to classify.")
    if len(message) > 2000:
        raise UserError("The message is too long (maximum 2000 characters).")

    with st.lock:
        x = st.vectorizer.transform([message])
        names = st.vectorizer.get_feature_names_out()
        known_terms = [names[j] for j in x.indices]

        score = float(st.svm.decision_function(x)[0])
        svm_label = "spam" if st.svm.predict(x)[0] == 1 else "ham"
        contributions = None
        if st.svm_params["kernel"] == "linear":
            coef = st.svm.coef_
            coef = np.asarray(coef.todense()).ravel() if sparse.issparse(coef) else np.ravel(coef)
            contrib = [(names[j], float(v * coef[j])) for j, v in zip(x.indices, x.data)]
            contrib.sort(key=lambda t: -abs(t[1]))
            contributions = [{"term": t, "contribution": r4(c)} for t, c in contrib[:8]]

        mu, var, p_map, p_bayes = blr_predict(st.blr, x)
        blr_vocab = set(names[st.blr["selector"].get_support()])

    p_b = float(p_bayes[0])
    return jsonify({
        "ok": True,
        "message": message,
        "cleaned": preprocess_text(message),
        "known_terms": known_terms[:30],
        "n_known_terms": len(known_terms),
        "svm": {"label": svm_label, "decision_score": r4(score),
                "kernel": st.svm_params["kernel"], "params": st.svm_params,
                "contributions": contributions},
        "blr": {"label": "spam" if p_b >= 0.5 else "ham",
                "p_spam_bayes": r4(p_b), "p_spam_map": r4(float(p_map[0])),
                "activation_mean": r4(float(mu[0])), "activation_std": r4(float(np.sqrt(var[0]))),
                "n_terms_in_blr_vocab": int(sum(1 for t in known_terms if t in blr_vocab))},
    })


# ---------------------------------------------------------------------------
# Routes - regression practical (separate from spam classification)
# ---------------------------------------------------------------------------
@app.route("/api/linear-regression")
@api
def api_linear_regression():
    st = STATE
    st.require_data()
    inject = request.args.get("outliers", "0") == "1"
    f = regression_frame(st)
    x = f["word_count"].values.astype(float)
    y = f["char_length"].values.astype(float)
    idx = np.arange(len(x))
    tr, te = train_test_split(idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    x_tr, y_tr, x_te, y_te = x[tr], y[tr], x[te], y[te]

    n_out = 0
    if inject:  # synthetic outliers added to the TRAINING set only, clearly labelled
        rng = np.random.RandomState(RANDOM_STATE)
        n_out = max(3, int(0.05 * len(x_tr)))
        xo = rng.uniform(np.percentile(x_tr, 70), x_tr.max() + 1, n_out)
        yo = rng.uniform(0, max(5.0, np.percentile(y_tr, 10)), n_out)
        x_tr, y_tr = np.r_[x_tr, xo], np.r_[y_tr, yo]

    # Least squares by the normal equations:  w = (Phi^T Phi)^-1 Phi^T t
    Phi = np.c_[np.ones_like(x_tr), x_tr]
    w_ls, *_ = np.linalg.lstsq(Phi, y_tr, rcond=None)
    sk = LinearRegression().fit(x_tr.reshape(-1, 1), y_tr)
    resid = y_tr - Phi @ w_ls
    sigma2_ml = float(np.mean(resid ** 2))       # ML estimate of noise variance

    huber = HuberRegressor(max_iter=1000).fit(x_tr.reshape(-1, 1), y_tr)
    pred_te = sk.predict(x_te.reshape(-1, 1))
    pred_tr = sk.predict(x_tr.reshape(-1, 1))
    pred_hub = huber.predict(x_te.reshape(-1, 1))

    show_tr = subsample_idx(len(x_tr), 700, 2)
    show_te = subsample_idx(len(x_te), 300, 3)
    line_x = [float(min(x.min(), x_tr.min())), float(max(x.max(), x_tr.max()))]
    train_table = [{"message": f.loc[tr[i], "message"][:70], "word_count": int(x[tr[i]]), "char_length": int(y[tr[i]])}
                   for i in range(min(8, len(tr)))]
    table = [{"message": f.loc[te[i], "message"][:70], "word_count": int(x_te[i]),
              "actual": int(y_te[i]), "predicted": r4(pred_te[i]), "error": r4(y_te[i] - pred_te[i])}
             for i in range(min(10, len(te)))]
    return jsonify({
        "ok": True,
        "task": "Predict SMS character length from its word count (educational regression demo)",
        "n_train": int(len(x_tr)), "n_test": int(len(x_te)), "synthetic_outliers": int(n_out),
        "normal_equation": {"intercept": r4(w_ls[0]), "slope": r4(w_ls[1])},
        "sklearn": {"intercept": r4(sk.intercept_), "slope": r4(sk.coef_[0])},
        "huber": {"intercept": r4(huber.intercept_), "slope": r4(huber.coef_[0]),
                  "test": reg_metrics(y_te, pred_hub), "n_outliers_flagged": int(huber.outliers_.sum())},
        "sigma2_ml": r4(sigma2_ml), "sigma_ml": r4(np.sqrt(sigma2_ml)),
        "train_metrics": reg_metrics(y_tr, pred_tr), "test_metrics": reg_metrics(y_te, pred_te),
        "scatter_train": {"x": x_tr[show_tr].tolist(), "y": y_tr[show_tr].tolist(),
                          "is_outlier": (show_tr >= len(tr)).tolist()},
        "scatter_test": {"x": x_te[show_te].tolist(), "y": y_te[show_te].tolist()},
        "line_x": line_x,
        "line_ols": [r4(sk.intercept_ + sk.coef_[0] * v) for v in line_x],
        "line_huber": [r4(huber.intercept_ + huber.coef_[0] * v) for v in line_x],
        "actual_vs_pred": {"actual": y_te[show_te].tolist(), "predicted": np.round(pred_te[show_te], 2).tolist()},
        "table": table,
        "train_table": train_table,
    })


@app.route("/api/polynomial")
@api
def api_polynomial():
    st = STATE
    st.require_data()
    f = regression_frame(st)
    x = f["word_count"].values.astype(float).reshape(-1, 1)
    y = f["char_length"].values.astype(float)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(x_tr)
    grid = np.linspace(x.min(), x.max(), 80).reshape(-1, 1)
    rows, curves = [], {}
    for d in range(1, 9):
        poly = PolynomialFeatures(degree=d, include_bias=False)
        Ptr = poly.fit_transform(scaler.transform(x_tr))
        model = LinearRegression().fit(Ptr, y_tr)
        p_tr = model.predict(Ptr)
        p_te = model.predict(poly.transform(scaler.transform(x_te)))
        rows.append({"degree": d, "train_rmse": r4(np.sqrt(mean_squared_error(y_tr, p_tr))),
                     "test_rmse": r4(np.sqrt(mean_squared_error(y_te, p_te))),
                     "test_r2": r4(r2_score(y_te, p_te))})
        curves[str(d)] = np.round(model.predict(poly.transform(scaler.transform(grid))), 3).tolist()
    show = subsample_idx(len(x_tr), 600, 4)
    best = min(rows, key=lambda r: r["test_rmse"])
    return jsonify({"ok": True, "rows": rows, "curves": curves, "grid_x": grid.ravel().round(3).tolist(),
                    "train_points": {"x": x_tr[show].ravel().tolist(), "y": y_tr[show].tolist()},
                    "best_degree_by_test_rmse": best["degree"],
                    "n_train": int(len(y_tr)), "n_test": int(len(y_te))})


@app.route("/api/ridge")
@api
def api_ridge():
    st = STATE
    st.require_data()
    try:
        alpha = float(request.args.get("alpha", 1000))
    except ValueError:
        raise UserError("Alpha must be a number.")
    if not (0 < alpha <= 100000):
        raise UserError("Alpha must be greater than 0 and at most 100000.")
    f = regression_frame(st)
    feature_names = ["word_count", "avg_word_length", "digit_count", "uppercase_count", "punctuation_count"]
    X = f[feature_names].values.astype(float)
    y = f["char_length"].values.astype(float)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scaler = StandardScaler().fit(X_tr)
    Xtr_s, Xte_s = scaler.transform(X_tr), scaler.transform(X_te)

    ols = LinearRegression().fit(Xtr_s, y_tr)
    ridge = Ridge(alpha=alpha).fit(Xtr_s, y_tr)
    bayes = BayesianRidge().fit(Xtr_s, y_tr)
    b_mean, b_std = bayes.predict(Xte_s, return_std=True)

    alphas = np.logspace(-2, 5, 30)
    path, test_rmse = [], []
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr_s, y_tr)
        path.append(m.coef_.round(4).tolist())
        test_rmse.append(r4(np.sqrt(mean_squared_error(y_te, m.predict(Xte_s)))))
    corr = np.corrcoef(X_tr, rowvar=False)
    return jsonify({
        "ok": True,
        "task": "Predict SMS character length from five numeric message features (standardised)",
        "features": feature_names, "alpha": alpha,
        "n_train": int(len(y_tr)), "n_test": int(len(y_te)),
        "ols": {"coef": [r4(c) for c in ols.coef_], "intercept": r4(ols.intercept_),
                "test": reg_metrics(y_te, ols.predict(Xte_s)), "coef_norm": r4(np.linalg.norm(ols.coef_))},
        "ridge": {"coef": [r4(c) for c in ridge.coef_], "intercept": r4(ridge.intercept_),
                  "test": reg_metrics(y_te, ridge.predict(Xte_s)), "coef_norm": r4(np.linalg.norm(ridge.coef_))},
        "bayesian": {"coef": [r4(c) for c in bayes.coef_], "intercept": r4(bayes.intercept_),
                     "test": reg_metrics(y_te, b_mean),
                     "noise_precision_alpha": r4(bayes.alpha_), "weight_precision_lambda": r4(bayes.lambda_),
                     "mean_predictive_std": r4(b_std.mean()),
                     "examples": [{"actual": int(y_te[i]), "mean": r4(b_mean[i]), "std": r4(b_std[i])}
                                  for i in range(min(6, len(y_te)))]},
        "path": {"alphas": alphas.round(4).tolist(), "coefs": path, "test_rmse": test_rmse},
        "correlation": np.round(np.nan_to_num(corr), 3).tolist(),
    })


# ---------------------------------------------------------------------------
# Routes - findings
# ---------------------------------------------------------------------------
def build_findings(st):
    df, rep_ = st.df, st.report
    total = len(df)
    spam = int((df["label"] == "spam").sum())
    lengths = df["message"].str.len()
    is_spam = df["label"] == "spam"
    common = lengths.value_counts().head(3)
    s, b = st.svm_result, st.blr_result
    sm, bm = s["metrics"], b["metrics"]
    yt = st.y_test
    majority = max(yt.mean(), 1 - yt.mean())

    def pct(v):
        return "n/a" if v is None else f"{100 * v:.2f}%"

    findings = [
        f"The loaded dataset ({st.source_name}) contains {total:,} valid SMS messages: "
        f"{total - spam:,} ham and {spam:,} spam ({100 * spam / total:.2f}% spam).",
        f"{rep_['duplicates']:,} duplicate messages were found; {len(st.model_df):,} unique messages were used "
        f"for modelling so that identical texts cannot appear in both the training and test sets.",
        f"{rep_['missing_values']:,} missing values, {rep_['empty_messages']:,} empty messages and "
        f"{rep_['unexpected_labels']:,} rows with unexpected labels were removed during validation.",
        f"Spam messages average {lengths[is_spam].mean():.1f} characters versus {lengths[~is_spam].mean():.1f} "
        f"for ham. The most common message lengths are "
        + ", ".join(f"{int(k)} chars ({int(v)} msgs)" for k, v in common.items()) + ".",
        f"TF-IDF (fitted on {len(st.y_train):,} training messages only) produced {st.Xtr.shape[1]:,} features; "
        f"the training matrix is {100 * (1 - st.Xtr.nnz / (st.Xtr.shape[0] * st.Xtr.shape[1])):.2f}% zeros.",
        f"The SVM ({st.svm_params['kernel']} kernel, C={st.svm_params['C']}) kept "
        f"{s['support_vectors']['total']:,} support vectors ({s['support_vector_pct']:.1f}% of training messages).",
        f"On the {len(yt):,}-message test set the SVM measured accuracy {pct(sm['accuracy'])}, precision "
        f"{pct(sm['precision'])}, recall {pct(sm['recall'])} and F1 {pct(sm['f1'])} "
        f"(TP={sm['confusion']['tp']}, TN={sm['confusion']['tn']}, FP={sm['confusion']['fp']}, FN={sm['confusion']['fn']}).",
        f"Bayesian Logistic Regression (Laplace approximation, prior variance {st.blr_params['prior_variance']}, "
        f"{b['n_features_used']} features) measured accuracy {pct(bm['accuracy'])}, precision {pct(bm['precision'])}, "
        f"recall {pct(bm['recall'])} and F1 {pct(bm['f1'])} on the same test set.",
        f"Always predicting the majority class would give {pct(majority)} accuracy on this test set, which is why "
        f"precision, recall and F1 are reported alongside accuracy.",
    ]
    if st.nb_result:
        nm = st.nb_result["metrics"]
        findings.append(f"The generative reference model (Multinomial Naive Bayes) measured F1 {pct(nm['f1'])} "
                        f"(precision {pct(nm['precision'])}, recall {pct(nm['recall'])}) on the same test set.")
    if st.kernel_results:
        best_f1 = max(st.kernel_results["rows"], key=lambda r: r["f1"])
        findings.append("In the kernel experiment, the highest measured F1 on this split came from the "
                        f"{best_f1['kernel']} kernel ({pct(best_f1['f1'])}). This is specific to this dataset and split.")
    if st.tuning_results:
        t = st.tuning_results
        findings.append(f"Grid search ({t['cv_folds']}-fold CV on training data) selected kernel="
                        f"{t['best_params']['kernel']}, C={t['best_params']['C']} with mean CV F1 {pct(t['best_cv_f1'])}.")
    if st.cv_results:
        c = st.cv_results
        findings.append(f"{c['k']}-fold cross-validation on all {c['n_messages']:,} unique messages gave mean F1 "
                        f"{pct(c['summary']['svm']['f1']['mean'])} ± {100 * c['summary']['svm']['f1']['std']:.2f} for the SVM "
                        f"and {pct(c['summary']['blr']['f1']['mean'])} ± {100 * c['summary']['blr']['f1']['std']:.2f} "
                        f"for Bayesian logistic regression.")
    if st.is_sample:
        findings.append("Note: these results come from the small built-in sample dataset. "
                        "Upload the Kaggle SMS Spam Collection file for the real project results.")
    return findings


@app.route("/api/findings")
@api
def api_findings():
    st = STATE
    st.require_models()
    return jsonify({"ok": True, "findings": build_findings(st), "is_sample": st.is_sample})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(413)
def too_large(_e):
    return jsonify({"ok": False, "error": f"The file is larger than {MAX_UPLOAD_MB} MB."}), 413


@app.errorhandler(404)
def not_found(e):
    if request.path.startswith("/api/"):
        return jsonify({"ok": False, "error": f"Unknown API endpoint: {request.path}"}), 404
    return e


@app.errorhandler(405)
def bad_method(_e):
    return jsonify({"ok": False, "error": "This endpoint does not accept that HTTP method."}), 405


# ---------------------------------------------------------------------------
# Start-up: train on the sample dataset so the website works immediately
# ---------------------------------------------------------------------------
def bootstrap():
    try:
        with open(SAMPLE_PATH, "rb") as f:
            raw = read_csv_bytes(f.read())
        lc, tc = detect_columns(raw)
        load_dataset(raw, lc, tc, "sample_sms.csv", is_sample=True)
        print(f"[startup] Sample dataset loaded and models trained ({len(STATE.df)} messages).")
    except Exception as exc:  # the app still starts; the user can upload a file
        print(f"[startup] Could not load the sample dataset: {exc}")


bootstrap()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n  SMS Spam Intelligence Dashboard running at  http://127.0.0.1:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
