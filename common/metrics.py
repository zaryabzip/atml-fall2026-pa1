"""Metrics shared by all tasks."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


def stratified_split(labels, val_fraction: float, seed: int):
    """Stratified train/val split of indices 0..N-1. Returns sorted (train_idx, val_idx)."""
    labels = np.asarray(labels)
    idx = np.arange(len(labels))
    train_idx, val_idx = train_test_split(idx, test_size=val_fraction, stratify=labels, random_state=seed)
    return np.sort(train_idx), np.sort(val_idx)


def classification_metrics(y_true, y_pred, num_classes: int) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)), zero_division=0)),
    }


def per_class_accuracy(y_true, y_pred, num_classes: int) -> np.ndarray:
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    with np.errstate(invalid="ignore", divide="ignore"):
        return cm.diagonal() / cm.sum(axis=1)


def confusion(y_true, y_pred, num_classes: int) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


# ---- Open-set helpers (larger unknownness score = more novel) ----

def auroc_known_vs_unknown(u_known, u_unknown) -> float:
    scores = np.concatenate([u_known, u_unknown])
    is_unknown = np.concatenate([np.zeros(len(u_known)), np.ones(len(u_unknown))])
    return float(roc_auc_score(is_unknown, scores))


def threshold_from_validation(u_val_known, accept_rate: float = 0.95) -> float:
    """tau = 95th percentile of known validation scores; accept x when u(x) <= tau."""
    return float(np.percentile(u_val_known, 100 * accept_rate))


def rejection_report(u_known_test, u_unknown, tau: float) -> dict:
    known_accept = float(np.mean(u_known_test <= tau))
    unknown_accept = float(np.mean(u_unknown <= tau))
    return {
        "threshold": tau,
        "known_acceptance": known_accept,
        "unknown_rejection": 1.0 - unknown_accept,
        "fpr_at_95tpr": unknown_accept,
    }


# ---- Linear probe (domain separability in Tasks 2 and 3) ----

def linear_probe_accuracy(features, labels, seed: int, test_fraction: float = 0.3, C: float = 1.0) -> float:
    """Held-out accuracy of a class-balanced logistic regression (binary or multinomial).
    Features are used as given (no scaling); state this in the report if you keep it."""
    x_tr, x_te, y_tr, y_te = train_test_split(
        features, labels, test_size=test_fraction, stratify=labels, random_state=seed
    )
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=5000)
    clf.fit(x_tr, y_tr)
    return float(clf.score(x_te, y_te))
