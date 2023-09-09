from np_sims.pca import pca
import numpy as np

from test_utils import as_normed_vects, w_scenarios


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "num_eigvecs": 2,
    },
    "all_components": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "num_eigvecs": None,
    }
}


@w_scenarios(vector_test_scenarios)
def test_pca(vectors, num_eigvecs):
    if num_eigvecs is None:
        num_eigvecs = vectors.shape[1]
    eigvecs = pca(vectors, num_eigvecs)
    assert eigvecs.shape == (3, num_eigvecs)

    # Check that the eigenvectors are orthogonal
    dotted = np.dot(eigvecs[:, 0], eigvecs[:, 1])
    assert np.allclose(dotted, 0)

    # Check that the eigenvectors are unit vectors
    assert np.allclose(np.linalg.norm(eigvecs[:, 0]), 1)

    # Check variance of first eigenvector is greater than second
    assert np.var(np.dot(vectors, eigvecs[:, 0])) > np.var(np.dot(vectors, eigvecs[:, 1]))
