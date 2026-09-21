"""Clean / color / patch-shuffle / translation evaluation and shape-bias scoring."""
import numpy as np

from common.metrics import classification_metrics

NUM_CLASSES = 10  # STL-10 has 10 classes


def prediction_report(y_true, pred, confidence, pred_clean=None) -> dict:
    """Top-1 accuracy, macro-F1, mean max confidence and, if `pred_clean` is given, prediction
    consistency = mean(pred == pred_clean) plus accuracy change vs clean.
    `confidence` is either per-image max probabilities (N,) or the full probability matrix (N, C)."""
    # Normalize inputs, and collapse a full probability matrix down to just the top-class probability.
    y_true, pred, confidence = np.asarray(y_true), np.asarray(pred), np.asarray(confidence)
    if confidence.ndim == 2:
        confidence = confidence.max(axis=1)

    # Core metrics: accuracy, macro-F1, and how confident the model's top prediction was on average.
    report = classification_metrics(y_true, pred, NUM_CLASSES)
    report["mean_max_confidence"] = float(confidence.mean())

    # If we were given the model's prediction on the clean, untransformed image, also report how
    # much the prediction changed and how accuracy moved relative to that baseline.
    report["prediction_consistency"] = None
    report["accuracy_change_vs_clean"] = None
    if pred_clean is not None:
        pred_clean = np.asarray(pred_clean)
        consistency = (pred == pred_clean).mean()
        clean_accuracy = (y_true == pred_clean).mean()
        report["prediction_consistency"] = float(consistency)
        report["accuracy_change_vs_clean"] = report["accuracy"] - float(clean_accuracy)

    return report


def shape_bias(pred, content_labels, style_labels) -> dict:
    """N_shape (pred == content), N_texture (pred == style), N_other (neither).
    shape_bias = N_shape / (N_shape + N_texture) * 100; coverage = (N_shape + N_texture) / N_total * 100."""
    # Sort every cue-conflict prediction into "sided with shape", "sided with texture", or neither.
    pred, content_labels, style_labels = np.asarray(pred), np.asarray(content_labels), np.asarray(style_labels)
    is_shape = pred == content_labels
    is_texture = pred == style_labels
    n_shape, n_texture = int(is_shape.sum()), int(is_texture.sum())
    n_total = len(pred)
    n_other = n_total - n_shape - n_texture

    # shape_bias is % shape among "decided" images (shape or texture); coverage is how many of all
    # the images were decided at all. Both are None rather than a divide-by-zero if there's nothing.
    n_decided = n_shape + n_texture
    shape_bias_pct = 100.0 * n_shape / n_decided if n_decided > 0 else None
    coverage_pct = 100.0 * n_decided / n_total if n_total > 0 else None

    return {
        "n_shape": n_shape,
        "n_texture": n_texture,
        "n_other": n_other,
        "n_total": n_total,
        "shape_bias": shape_bias_pct,
        "coverage": coverage_pct,
    }
