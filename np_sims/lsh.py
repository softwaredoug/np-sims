import numpy as np
from np_sims import hamming_c, hamming_top_10, hamming_top_cand, hamming_naive


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


def query_with_hamming_then_slow_argsort(vector, hashes, projections, hamming_func=hamming_c, n=10):
    """Query using C-based hamming similarity."""
    # Compute the hashes for the vector
    query_hash = index([vector], projections)[0]
    # Compute the hamming distance between the query and all hashes
    hammings = hamming_func(hashes, query_hash)
    # Sort ascending, 0 is most similar <-- actual bottleneck
    idxs = np.argsort(hammings)[:n]
    lsh_sims = hammings[idxs]
    return idxs, lsh_sims


def query_pure_python(vector, hashes, projections, hamming_func=hamming_naive, n=10):
    # Compute the hashes for the vector
    query_hash = index([vector], projections)[0]
    # Compute the hamming distance between the query and all hashes
    hammings = hamming_func(hashes, query_hash)
    # Sort ascending, 0 is most similar <-- actual bottleneck
    idxs = np.argsort(hammings)[:n]
    lsh_sims = hammings[idxs]
    return idxs, lsh_sims


def query_with_hamming_top_n(vector, hashes, projections, hamming_func=hamming_c, n=10):
    """Query using C-based hamming similarity w/ top N."""
    query_hash = index([vector], projections)[0]
    best = hamming_top_10(hashes, query_hash)
    return best, None


def front_hashes(hashes, front_bits=64):
    """Return the first front_bits bits of each hash."""
    front_len_64 = front_bits // 64
    # Get front_len_64 columns
    return hashes[:, 0:front_len_64].copy()


def query_with_hamming_top_n_two_phase(vector, front_hashes, hashes, projections, hamming_func=hamming_c, n=10):
    """Query using C-based hamming similarity w/ top N."""
    query_hash = index([vector], projections)[0]
    query_hash_front = query_hash[:front_hashes.shape[1]].copy()
    candidates = hamming_top_cand(front_hashes, query_hash_front)
    best = hamming_top_10(hashes[candidates], query_hash)
    return candidates[best], None
    # idxs = np.argsort(hammings)[:n]
    # lsh_sims = hammings[idxs]
    # return idxs, lsh_sims


query = query_with_hamming_top_n  # Default query to fastest
