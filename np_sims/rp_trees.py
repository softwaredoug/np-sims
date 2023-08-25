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
    if (vect1 == vect2).all():
        raise ValueError("Vectors cannot be identical")
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


def fit(vectors: np.ndarray, depth: int = 10):
    """Build a random projection tree from a set of vectors."""
    # Pick two random vectors from vectors
    v1, v2 = vectors[np.random.choice(len(vectors), 2, replace=False)]
    root = projection_between(v1, v2)

    dotted = np.dot(vectors, root)
    left = vectors[dotted < 0]
    right = vectors[dotted >= 0]
    assert len(left) != 0
    assert len(right) != 0

    if depth > 1:
        left = fit(left, depth - 1) if len(left) > 1 else None
        right = fit(right, depth - 1) if len(right) > 1 else None

        return root, left, right
    else:
        return root, None, None


def rp_hash(tree, vectors, hashes=None, depth=63):
    """Walk a vector down a tree to get a hash."""
    if tree is None or depth < 0:
        return hashes

    if hashes is None:
        hashes = np.zeros(dtype=np.uint64, shape=(len(vectors)))

    root, left, right = tree
    dotted = np.dot(vectors, root)

    lhs_idx = np.ravel(np.argwhere(dotted < 0))
    rhs_idx = np.ravel(np.argwhere(dotted >= 0))

    print(f"Depth: {depth}, LHS: {lhs_idx}, RHS: {rhs_idx} -- vectors: {len(vectors)}")

    lhs_vectors = vectors[lhs_idx]
    rhs_vectors = vectors[rhs_idx]

    lhs_hashes = hashes[lhs_idx]
    rhs_hashes = hashes[rhs_idx]

    hashes[lhs_idx] |= rp_hash(left, lhs_vectors, lhs_hashes, depth - 1)
    # Set the 'depth' bit in rhs_idx
    hashes[rhs_idx] |= (1 << depth) | (rp_hash(right, rhs_vectors, rhs_hashes, depth - 1))

    return hashes


def hash_table(hashes):
    """Sort hashes, produce an array of the sorts with the actual values."""
    sorted_idxs = np.argsort(hashes, kind="stable")
    sorted_hashes = hashes[sorted_idxs]
    return sorted_idxs, sorted_hashes


def binary_search(sorted_hashes: np.ndarray[np.uint64], query_hashed: np.ndarray):
    return np.searchsorted(sorted_hashes, query_hashed)


def depth(tree):
    """Get the depth of a tree."""
    if tree is None:
        return 0
    else:
        return 1 + max(depth(tree[1]), depth(tree[2]))
