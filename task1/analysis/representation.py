"""t-SNE / UMAP of clean + transformed features (one joint projection per backbone)."""
import numpy as np


def _fit_tsne(joint_feats, seed, settings):
    """Fit t-SNE on the combined clean+transformed features and return 2-D coordinates."""
    from sklearn.manifold import TSNE
    perplexity = settings.get("perplexity", 30)
    reducer = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    return reducer.fit_transform(joint_feats)


def _fit_umap(joint_feats, seed, settings):
    """Fit UMAP on the combined clean+transformed features and return 2-D coordinates."""
    import umap
    n_neighbors = settings.get("n_neighbors", 15)
    min_dist = settings.get("min_dist", 0.1)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
    return reducer.fit_transform(joint_feats)


def joint_projection(clean_feats, transformed_feats, labels, method: str, seed: int, **settings):
    """Fit ONE 2-D projection on concatenated [clean; transformed]; color = true class,
    marker = clean vs transformed. Records all settings. Don't compare coordinates across backbones."""
    clean, transformed, labels = np.asarray(clean_feats), np.asarray(transformed_feats), np.asarray(labels)
    n = len(labels)

    # Fit ONE projection on both halves stacked together, so they land in the same 2-D space.
    joint_feats = np.concatenate([clean, transformed], axis=0)  # (2N, D)
    if method == "tsne":
        embedding = _fit_tsne(joint_feats, seed, settings)
    elif method == "umap":
        embedding = _fit_umap(joint_feats, seed, settings)
    else:
        raise ValueError(f"Unknown method {method!r}; choose 'tsne' or 'umap'.")

    # class_labels/domain tell the caller how to color/mark each of the 2N rows when plotting.
    class_labels = np.concatenate([labels, labels])  # clean half, then transformed half
    domain = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])  # 0 = clean, 1 = transformed

    return {
        "embedding": embedding,
        "class_labels": class_labels,
        "domain": domain,
        "method": method,
        "seed": seed,
        "settings": dict(settings),
    }
