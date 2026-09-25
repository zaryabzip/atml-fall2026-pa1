"""Inspect unknowns accepted under the vanilla MLS threshold. Analysis only: never feed back into training."""
import numpy as np

from task4.data.cifar import CIFAR10_CLASSES  # noqa: E402

# Known classes an unknown could reasonably be mistaken for (fixed before looking at any prediction).
PLAUSIBLE = {
    "bus": {"automobile", "truck"}, "pickup_truck": {"automobile", "truck"}, "tractor": {"automobile", "truck"},
    "motorcycle": {"automobile", "truck"}, "wolf": {"dog", "cat", "deer"}, "fox": {"dog", "cat", "deer"},
    "leopard": {"cat", "dog", "deer"}, "camel": {"horse", "deer"},
}


def accepted_unknowns(u_unknown, tau, pred_known, fine_labels, fine_class_names, k: int = 3):
    """Unknowns with u <= tau (wrongly accepted), most confident first (lowest u).
    Returns (the k most confident failures, per-class acceptance summary)."""
    accepted = np.flatnonzero(u_unknown <= tau)
    accepted = accepted[np.argsort(u_unknown[accepted])]

    def describe(i):
        unknown = fine_class_names[fine_labels[i]]
        predicted = CIFAR10_CLASSES[pred_known[i]]
        return {"index": int(i), "unknown_class": unknown, "predicted_class": predicted,
                "score": float(u_unknown[i]), "threshold": float(tau),
                "kind": "semantically plausible" if predicted in PLAUSIBLE.get(unknown, set()) else "surprising"}

    # Take the most confident failure of each unknown class first, so the examples are not all one class.
    picked, seen = [], set()
    for i in accepted:
        c = fine_labels[i]
        if c not in seen:
            seen.add(c)
            picked.append(i)
        if len(picked) == k:
            break

    summary = {}
    for c in np.unique(fine_labels):
        name = fine_class_names[c]
        members = fine_labels == c
        acc_members = members & (u_unknown <= tau)
        preds, counts = np.unique(pred_known[acc_members], return_counts=True)
        summary[name] = {"accepted_rate": float(acc_members.sum() / members.sum()),
                         "absorbed_by": {CIFAR10_CLASSES[p]: int(n) for p, n in
                                         sorted(zip(preds, counts), key=lambda t: -t[1])}}
    return [describe(i) for i in picked], summary
