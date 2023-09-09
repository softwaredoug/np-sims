from typing import Optional

import numpy as np


def pca(vectors: np.ndarray, k: Optional[int] = None) -> np.ndarray:
    """Run a PCA to get top k components."""
    means = vectors.mean(axis=0)
    vectors_ctr = vectors - means
    cov = np.cov(vectors_ctr.T)
    eigvals, eigvecs = np.linalg.eig(cov)

    eigvals_sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[eigvals_sorted_indices]
    eigvecs = eigvecs[:, eigvals_sorted_indices]

    if k is None:
        return eigvecs
    else:
        top_eigvals = eigvals.argsort()[::-1][:k]
        return eigvecs[:, top_eigvals]
