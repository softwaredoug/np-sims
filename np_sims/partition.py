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
from np_sims.pca import pca

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

    def __init__(self, projection, median=0.0, delta=0.0):
        self.delta = delta
        self.median = median
        self.projection = projection
        self.split_point = self.median + self.delta

    # Here v is a random projection
    # Rule(x) := x · v ≤ (median({z · v : z ∈ S}) + δ)
    # With delta and median at 0, this is just a random projection
    def split(self, vectors: np.ndarray):
        """Get a random projection tree for a set of vectors."""
        dotted = np.dot(vectors, self.projection)
        # median = np.median(np.dot(vectors, self.projection))
        left = np.ravel(np.argwhere(dotted <= self.split_point))
        right = np.ravel(np.argwhere(dotted > self.split_point))
        return left, right


def random_partition_chooserule(vectors: np.ndarray) -> SplitRule:
    projection = random_projection(vectors.shape[1])
    return RPTreeMaxSplitRule(projection)


def _rptree_chooserule_with_median(vectors, projection, quantile_delta=0.25) -> SplitRule:
    # x = vectors[np.random.choice(len(vectors))]
    # farthest_from_x = vectors[np.dot(vectors, x).argmin()]

    # def euclidean_distance(v1, v2):
    #     return np.linalg.norm(v1 - v2)
    # dist = euclidean_distance(x, farthest_from_x)

    # Paper suggests this 'jitter' which seems to often go out of the bounds
    # I am using quantiles as suggested by the rpforests paper
    # delta = chosen * (6 * dist) / np.sqrt(vectors.shape[1])

    dotted = np.dot(vectors, projection)
    quartiles = np.quantile(dotted, [0.5 - quantile_delta, 0.5,  0.5 + quantile_delta])

    delta = np.random.uniform(low=quartiles[0], high=quartiles[2])

    return RPTreeMaxSplitRule(projection, median=quartiles[1], delta=delta)


# choose a random unit direction v ∈ R
# pick any x ∈ S; let y ∈ S be the farthest point from it
# choose δ uniformly at random in [−1, 1] · 6||x − y||/√D
# Rule(x) := x · v ≤ (median({z · v : z ∈ S}) + δ)
def rptree_max_chooserule(vectors: np.ndarray) -> SplitRule:
    # Just go with projection between for very small sets
    # which we probably shouldn't be using anyway
    if len(vectors) < 20:
        return projection_between_chooserule(vectors)
    projection = random_projection(vectors.shape[1])
    return _rptree_chooserule_with_median(vectors, projection)


def rptree_proj_maxvar_chooserule(vectors: np.ndarray, projection=None) -> SplitRule:
    """Find random projections with maximim variance."""
    max_var = 0
    max_var_proj = None

    if projection is None:
        # Really this could just be getting the PCA of the vectors
        for i in range(0, 1):
            projection = random_projection(vectors.shape[1])
            var = np.dot(vectors, projection).max()
            if var > max_var:
                max_var = var
                max_var_proj = projection
    else:
        max_var_proj = projection

    return _rptree_chooserule_with_median(vectors, max_var_proj)


def rptree_pca_chooserule(vectors: np.ndarray, quantile_delta=0.25) -> SplitRule:
    """Find random projections using Principal Component Analysis."""
    components = pca(vectors)

    # Get the eigenvector with the largest eigenvalue
    return _rptree_chooserule_with_median(vectors, components[0], quantile_delta=quantile_delta)


class MultiProjectionSplitRule(SplitRule):
    def __init__(self, projections):
        self.projections = projections

    # Spli
    def split(self, vectors: np.ndarray):
        """Get a random projection tree for a set of vectors."""
        side = np.zeros(len(vectors), dtype=np.int8)
        for proj in self.projections:
            dotted = np.dot(vectors, proj)
            lhs = np.ravel(np.argwhere(dotted <= 0))
            rhs = np.ravel(np.argwhere(dotted > 0))
            side[lhs] -= 1
            side[rhs] += 1
        # Count > 0
        left = np.ravel(np.argwhere(side > 0))
        right = np.ravel(np.argwhere(side <= 0))
        assert len(left) + len(right) == len(vectors)
        return left, right


def multi_projection_chooserule(vectors: np.ndarray, num_projections=20) -> SplitRule:
    """Choose a random projection vector that splits two random vectors chosen."""
    # Choose two random vectors
    projections = np.random.normal(size=(num_projections, vectors.shape[1]))
    projections /= np.linalg.norm(projections, axis=1, keepdims=True)
    return MultiProjectionSplitRule(projections)
