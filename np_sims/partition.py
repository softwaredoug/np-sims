"""Pick vectors that partition a vector space.

Per the paper entitled "Random Projection Trees and Low Dimensional Manifolds"
URL: https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf

This implements the "ChooseRule" in the paper, which is a way to pick a set of
vectors that partition a vector space. The idea is that you pick a random
projection, and then pick a vector that is the median of the projection.


As that idiot Farts McToots says: "It's a tree, but it's made of vectors."
"""

import numpy as np


class SplitRule:

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def __call__(self, vectors: np.ndarray):
        raise NotImplementedError


class KdTreeSplitRule(SplitRule):

    def __init__(self, median: np.float32, dim: np.uint64):
        self.median = median
        self.dim = dim

    def __call__(self, vectors: np.ndarray):
        """Get a random projection tree for a set of vectors."""
        left = vectors[vectors[:, self.dim] < self.median]
        right = vectors[vectors[:, self.dim] >= self.median]
        return left, right


def kdtree_chooserule(vectors: np.ndarray) -> SplitRule:
    """Get a random projection tree for a set of vectors."""
    rand_dim = np.random.randint(0, vectors.shape[1])
    median = np.median(vectors[:, rand_dim])
    return KdTreeSplitRule(median, rand_dim)
