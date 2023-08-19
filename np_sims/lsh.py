import numpy as np
from np_sims import hamming_c


def random_projection(dims):
    """Sample a unit vector from a sphere in N dimensions.

    It's actually important this is gaussian
    https://stackoverflow.com/questions/59954810/generate-random-points-on-10-dimensional-unit-sphere

    IE Don't do this
        projection = np.random.random_sample(size=num_dims)
        projection /= np.linalg.norm(projection)

    """
    projection = np.random.normal(size=dims)
    projection /= np.linalg.norm(projection)
    return projection


def create_projections(num_projections=0,
                       dims=0) -> np.ndarray:
    """Create a set of random projections."""
    projs = []
    for _ in range(num_projections):
        projs.append(random_projection(dims))

    return np.array(projs)


def set_bit(one_hash, idx):
    bit = idx % 64
    mask = np.uint64(np.uint64(2 ** bit))
    one_hash[idx // 64] |= mask
    return one_hash


def clear_bit(one_hash, idx):
    bit = idx % 64
    mask = np.uint64(np.uint64(2 ** bit))
    mask = ~mask
    one_hash[idx // 64] &= mask
    return one_hash


def index(vectors, projections):
    if len(projections) % 64 != 0:
        raise ValueError("Projections must be a multiple of 64")
    hashes = np.zeros((len(vectors), (len(projections) // 64)), dtype=np.uint64)
    for vect_idx, vect in enumerate(vectors):
        if vect_idx > 0 and vect_idx % 1000 == 0:
            print(f"idx: {vect_idx} -- {vect_idx / len(vectors)}")
        for bit_idx, proj in enumerate(projections):
            dot = np.dot(vect, proj)
            if dot >= 0:
                # print("+", end="")
                hashes[vect_idx] = set_bit(hashes[vect_idx], bit_idx)
            else:
                # print("-", end="")
                hashes[vect_idx] = clear_bit(hashes[vect_idx], bit_idx)
    return hashes


def query(vector, hashes, projections, hamming_func=hamming_c):
    # Compute the hashes for the vector (is this now a bottleneck?)
    query_hash = index([vector], projections)
    # Compute the hamming distance between the query and all hashes
    hammings = hamming_func(hashes, query_hash)
    # Sort ascending, 0 is most similar
    idxs = np.argsort(hammings)[0][0:10]
    lsh_sims = hammings[0][idxs]
    return idxs, lsh_sims
