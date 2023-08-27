"""Pick vectors that partition a vector space.

Per the paper entitled "Random Projection Trees and Low Dimensional Manifolds"
URL: https://cseweb.ucsd.edu/~dasgupta/papers/rptree-stoc.pdf

This implements the "ChooseRule" in the paper, which is a way to pick a set of
vectors that partition a vector space. The idea is that you pick a random
projection, and then pick a vector that is the median of the projection.


As that idiot Farts McToots says: "It's a tree, but it's made of vectors."
"""

import numpy as np
from np_sims.lsh import random_projection

# glove_queen = np.array([vectors[7613]])
# glove_king = np.array([vectors[3598]])


class SplitRule:

    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors

    def split(self, vectors: np.ndarray):
        raise NotImplementedError


class KdTreeSplitRule(SplitRule):

    def __init__(self, median: np.float32, dim: np.uint64):
        self.median = median
        self.dim = dim

    def split(self, vectors: np.ndarray):
        """Get a random projection tree for a set of vectors."""
        left = np.ravel(np.argwhere(vectors[:, self.dim] < self.median))
        right = np.ravel(np.argwhere(vectors[:, self.dim] >= self.median))
        return left, right


def kdtree_maxvar_chooserule(vectors: np.ndarray) -> SplitRule:
    """Get a random projection tree for a set of vectors."""
    # rand_dim = np.random.randint(0, vectors.shape[1])
    max_var_dim = np.argmax(np.var(vectors, axis=0))
    median = np.median(vectors[:, max_var_dim])
    return KdTreeSplitRule(median, max_var_dim)


def kdtree_randdim_chooserule(vectors: np.ndarray) -> SplitRule:
    """Get a random projection tree for a set of vectors."""
    # rand_dim = np.random.randint(0, vectors.shape[1])
    rand_dim = np.random.choice(vectors.shape[1])
    median = np.median(vectors[:, rand_dim])
    return KdTreeSplitRule(median, rand_dim)


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
    if np.isclose(vect1, vect2).all():
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
        projection[dim] = (floor + ceiling) / 2.0

    # Sign is different we just set to -1 or 1 which is basically
    # like bisecting by this axis (like in 2 dims, this would be
    # the y axis)
    else:
        projection = np.zeros(vect1.shape)
        projection[dim] = np.sign(vect1[dim])

    projection /= np.linalg.norm(projection)
    return projection


class ProjectionBetweenSplitRule(SplitRule):

    def __init__(self, projection: np.ndarray):
        self.projection = projection

    def split(self, vectors: np.ndarray):
        """Split vectors by the projection."""
        left = np.ravel(np.argwhere(np.dot(vectors, self.projection) < 0))
        right = np.ravel(np.argwhere(np.dot(vectors, self.projection) >= 0))
        return left, right


def projection_between_chooserule(vectors: np.ndarray, max_proximity=0.95) -> SplitRule:
    """Choose a random projection vector that splits two random vectors chosen."""
    # Choose two random vectors
    vect1, vect2 = vectors[np.random.choice(len(vectors), 2, replace=False)]

    tries = 0
    while np.dot(vect1, vect2) > max_proximity:
        vect1, vect2 = vectors[np.random.choice(len(vectors), 2, replace=False)]
        tries += 1
        if tries > 10:
            return kdtree_maxvar_chooserule(vectors)

    proj = projection_between(vect1, vect2)
    return ProjectionBetweenSplitRule(proj)


class RPTreeMaxSplitRule(SplitRule):

    def __init__(self, delta, median, projection):
        self.delta = delta
        self.median = median
        self.projection = projection

    # Here v is a random projection
    # Rule(x) := x · v ≤ (median({z · v : z ∈ S}) + δ)
    # With delta and median at 0, this is just a random projection
    def split(self, vectors: np.ndarray):
        """Get a random projection tree for a set of vectors."""
        dotted = np.dot(vectors, self.projection)
        # median = np.median(np.dot(vectors, self.projection))
        left = np.ravel(np.argwhere(dotted <= (self.median + self.delta)))
        right = np.ravel(np.argwhere(dotted > (self.median + self.delta)))
        return left, right


# choose a random unit direction v ∈ R
# pick any x ∈ S; let y ∈ S be the farthest point from it
# choose δ uniformly at random in [−1, 1] · 6||x − y||/√D
# Rule(x) := x · v ≤ (median({z · v : z ∈ S}) + δ)
def rptree_max_chooserule(vectors: np.ndarray) -> SplitRule:
    projection = random_projection(vectors.shape[1])
    x = vectors[np.random.choice(len(vectors))]
    farthest_from_x = vectors[np.dot(vectors, x).argmin()]

    def euclidean_distance(v1, v2):
        return np.linalg.norm(v1 - v2)
    dist = euclidean_distance(x, farthest_from_x)

    chosen = np.random.uniform(low=-1.0, high=1.0)
    delta = chosen * (6 * dist) / np.sqrt(vectors.shape[1])

    # Median of this will tend t obe 0, no?
    median = np.median(np.dot(vectors, projection))

    return RPTreeMaxSplitRule(delta, median, projection)
