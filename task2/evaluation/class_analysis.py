"""Per-class target accuracy change vs Source-only, and dominant confusions. Final analysis only."""
import numpy as np

from common.metrics import confusion, per_class_accuracy


def _top_confusions(cm: np.ndarray, class_idx: int, class_names, k: int = 3) -> list:
    """The k classes this class gets most often mistaken for (excludes correct predictions)."""
    row = cm[class_idx].copy()
    row[class_idx] = 0  # zero out the diagonal: we only want MIS-predictions here
    top_idx = np.argsort(row)[::-1][:k]
    return [{"predicted_as": class_names[j], "count": int(row[j])} for j in top_idx if row[j] > 0]


def per_class_changes(y_true, pred_method, pred_source_only, class_names) -> dict:
    """Per-class accuracy for both, the difference, the biggest gain/drop, and their confusions."""
    num_classes = len(class_names)

    # Per-class accuracy for this method and for Source-only, and how much each class changed.
    acc_method = per_class_accuracy(y_true, pred_method, num_classes)
    acc_source = per_class_accuracy(y_true, pred_source_only, num_classes)
    change = acc_method - acc_source  # positive = this method helped that class

    # Find the single biggest winner and loser, and this method's own confusion matrix.
    biggest_gain_idx = int(np.argmax(change))
    biggest_drop_idx = int(np.argmin(change))
    cm_method = confusion(y_true, pred_method, num_classes)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy_method": float(acc_method[i]),
            "accuracy_source_only": float(acc_source[i]),
            "change": float(change[i]),
        }

    return {
        "per_class": per_class,
        "biggest_gain": {
            "class": class_names[biggest_gain_idx],
            "change": float(change[biggest_gain_idx]),
            "top_confusions": _top_confusions(cm_method, biggest_gain_idx, class_names),
        },
        "biggest_drop": {
            "class": class_names[biggest_drop_idx],
            "change": float(change[biggest_drop_idx]),
            "top_confusions": _top_confusions(cm_method, biggest_drop_idx, class_names),
        },
    }
