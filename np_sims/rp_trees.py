from typing import Tuple, Optional
import pickle

import numpy as np
from np_sims.lsh import random_projection


def all_but(vect, idx):
    if idx < 0:
        raise ValueError("Only indexes > 0 supported")
    if idx == len(vect):
        excluded = vect[:-1]
    else:
        excluded = np.concatenate([vect[:idx], vect[idx + 1:]])
    assert len(excluded) == len(vect) - 1
    return excluded


# You can find a projection (p1...p2) that gives the opposite
# signed dot products by solving the inequalities
#
# a1 * p1 + a2 * p2 ... + an * pn >= 0
# b1 * p1 + b2 * b2 ... + bn * pn < 0
#
# --------
# Subtract one of the dot product components
#
# a1 * p1 + a2 * p2 ... >= -(an * pn)
# b1 * p1 + b2 * b2 ... <  -(bn * pn)
#
# -------
# Multiply both sides by -1
#
# -(a1 * p1 + a2 * p2 ...) <= an * pn
# -(b1 * p1 + b2 * b2 ...) >  bn * pn
#
# --------
# Rearrange
#
# an * pn >= -(a1 * p1 + a2 * p2 + ... )
# bn * pn <  -(b1 * p1 + b2 * p2 + ... )
#
# BOTH POSITIVE / BOTH NEGATIVE
# -----------------------------
# If an, bn are positive, we simply divide by an and bn
#
# pn >= -(a1 * p1 + a2 * p2 + ... ) / an
# pn <  -(b1 * p1 + b2 * p2 + ... ) / bn
#
# If both are negative, we flip b and a, and we get a similar
# inequality. Regardless one of the inequalities is the floor,
# the other is the ceiling, and pn can be anything in between.
#
# -----------
# IF an IS NEGATIVE, BUT bn IS POSITIVE
#
# an is negative sign flips for 'a' inequality
# pn < -(a1 * p1 + a2 * p2 + ... ) / an
# but b remains...
# pn < -(b1 * p1 + b2 * p2 + ... ) / bn
#
# Since vectors a and b are normalized, pn could just be -1
# with p1..pn-1 as 0s
#
# -----------
# If an POSITIVE, BUT bn IS NEGATIVE
#
# Same as before, except pn=1
#
# Since these are normalized vectors, set pn to -1 and normarize
def projection_between(vect1: np.ndarray,
                       vect2: np.ndarray,
                       dim=0):
    """Get a vector that gives opposite signed dot products for vect1/vect2."""
    if np.isclose(vect1, vect2):
        raise ValueError("Vectors cannot be near identical")
    dims = len(vect1)

    while vect1[dim] == 0.0 or vect2[dim] == 0.0:
        dim += 1

    # To get the floor / ceiling we solve an inequality
    # whose sign flips depending on whether vectn[dim] is
    # positive or negative
    if np.sign(vect1[dim]) == np.sign(vect2[dim]):
        projection = random_projection(dims)

        dot1 = np.dot(all_but(projection, dim),
                      all_but(vect1, dim))
        dot2 = np.dot(all_but(projection, dim),
                      all_but(vect2, dim))

        floor = -dot1 / vect1[dim]
        ceiling = -dot2 / vect2[dim]
        print(dim, floor, ceiling)
        projection[dim] = (floor + ceiling) / 2.0

    # Sign is different we just set to -1 or 1 which is basically
    # like bisecting by this axis (like in 2 dims, this would be
    # the y axis)
    else:
        projection = np.zeros(vect1.shape)
        projection[dim] = np.sign(vect1[dim])

    projection /= np.linalg.norm(projection)
    return projection


def _fit(vectors: np.ndarray, depth: int = 63):
    """Build a random projection tree from a set of vectors."""
    # Pick two random vectors from vectors
    v1, v2 = vectors[np.random.choice(len(vectors), 2, replace=False)]
    root = None
    try_count = 0
    while root is None:
        try:
            v1, v2 = vectors[np.random.choice(len(vectors), 2, replace=False)]
            root = projection_between(v1, v2)
        except ValueError:
            try_count += 1
            if try_count > 5:
                return None

    dotted = np.dot(vectors, root)
    left = vectors[dotted < 0]
    right = vectors[dotted >= 0]
    assert len(left) != 0
    assert len(right) != 0

    if depth > 1:
        left = _fit(left, depth - 1) if len(left) > 1 else None
        right = _fit(right, depth - 1) if len(right) > 1 else None

        return root, left, right
    else:
        return root, None, None


def _rp_hash(tree: Tuple,
             vectors: np.ndarray[np.float64],
             hashes: np.ndarray[np.uint64],
             depth: int = 63):
    """Walk a vector down a tree to get a hash."""
    if tree is None or depth <= 0:
        return hashes

    root, left, right = tree
    dotted = np.dot(vectors, root)

    lhs_idx = np.ravel(np.argwhere(dotted < 0))
    rhs_idx = np.ravel(np.argwhere(dotted >= 0))

    lhs_vectors = vectors[lhs_idx]
    rhs_vectors = vectors[rhs_idx]

    lhs_hashes = hashes[lhs_idx]
    rhs_hashes = hashes[rhs_idx]

    hashes[lhs_idx] |= _rp_hash(left, lhs_vectors, lhs_hashes, depth - 1)
    # Set the 'depth' bit in rhs_idx
    hashes[rhs_idx] |= (1 << depth) | (_rp_hash(right, rhs_vectors, rhs_hashes, depth - 1))

    return hashes


def _hash_table(hashes):
    """Sort hashes, produce an array of the sorts with the actual values."""
    sorted_idxs = np.argsort(hashes, kind="stable")
    sorted_hashes = hashes[sorted_idxs]
    return sorted_idxs, sorted_hashes


class RandomProjectionTree:
    """Pure Python random projection tree implementation."""

    def __init__(self, root, sorted_hashes, sorted_idxs):
        self.sorted_hashes = sorted_hashes
        self.sorted_idxs = sorted_idxs
        self.root = root

    def save(self, path: str):
        """Save the tree to a file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def build(vectors: np.ndarray, seed: Optional[int] = None):
        """Build a random projection tree from a set of vectors."""
        if seed is not None:
            np.random.seed(seed)
        root = _fit(vectors)
        hashes = _rp_hash(root, vectors, np.zeros(len(vectors), dtype=np.uint64))
        sorted_idxs, sorted_hashes = _hash_table(hashes)
        return RandomProjectionTree(root, sorted_hashes, sorted_idxs)

    @staticmethod
    def load(path: str):
        """Load the tree from a file."""
        with open(path, "rb") as f:
            return pickle.load(f)

    def hash_of(self, vectors: np.ndarray[np.float64]) -> np.ndarray[np.uint64]:
        """Get the hash of a vector."""
        hashes = np.zeros(dtype=np.uint64, shape=(len(vectors)))
        return _rp_hash(self.root, vectors, hashes)

    def query(self, query: np.ndarray[np.float64]) -> np.ndarray[np.uint64]:
        """Run a query against the tree."""
        most_similar = np.searchsorted(self.sorted_hashes, self.hash_of(query))
        return self.sorted_idxs[most_similar]

    def _depth(self, tree):
        """Get the depth of a tree."""
        if tree is None:
            return 0
        else:
            return 1 + max(self._depth(tree[1]), self._depth(tree[2]))

    def depth(self):
        """Get the depth of the tree."""
        return self._depth(self.root)
