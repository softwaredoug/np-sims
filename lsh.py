import numpy as np
from similarities import hamming
from time import perf_counter


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


def query(vector, hashes, projections):
    # Compute the hashes for the vector
    query_hash = index([vector], projections)
    # Compute the hamming distance between the query and all hashes
    hammings = hamming(hashes, query_hash)
    # Sort ascending, 0 is most similar
    idxs = np.argsort(hammings)[0][0:10]
    lsh_sims = hammings[0][idxs]
    return idxs, lsh_sims


def glove(num_to_sample=10000000):
    terms = []
    vectors = []

    terms_file = "glove.840B.300d_terms.txt"
    vectors_file = "glove.840B.300d_terms.npz"

    try:
        with open(terms_file, "r") as f:
            terms = [line.strip() for line in f]
        vectors = np.load(vectors_file)["arr_0"]
        return terms, vectors
    except FileNotFoundError:
        with open("glove.840B.300d.txt", "r") as f:
            for line in f:
                term, *vector = line.split()
                try:
                    vect = np.array([np.float32(val) for val in vector])
                    vect /= np.linalg.norm(vect)
                    if len(vect) != 300:
                        raise ValueError(f"Vector not 300 dims: {len(vect)}")
                    vectors.append(vect)
                    terms.append(term)

                    if len(terms) == num_to_sample:
                        break
                except ValueError as e:
                    print(e)
                    print("Skipping line ", line)
                    pass
            # Dump to file
            np.savez_compressed(vectors_file, vectors)
            with open(terms_file, "w") as f:
                for term in terms:
                    f.write(term + "\n")
        print(f"Loaded up to term: {terms[-1]} -- {len(terms)} terms")
        return terms, np.array(vectors)


def most_similar_cos(vectors, query_idx):
    query_vector = vectors[query_idx]
    sims = np.dot(vectors, query_vector)
    top_idxs = np.argsort(sims)[::-1][0:10]
    return top_idxs, sims[top_idxs]


def benchmark(terms, vectors, projs, hashes):

    # Randomly select 100 terms
    query_idxs = np.random.randint(0, len(terms), size=100)

    # Collect groundtruths
    results_gt = []
    for query_idx in query_idxs:
        result, sims = most_similar_cos(vectors, query_idx)
        results_gt.append(list(zip(result, sims)))

    # Run LSH
    results = []
    execution_times = 0
    for query_idx in query_idxs:
        start = perf_counter()
        result, sims = query(vectors[query_idx], hashes, projs)
        results.append(list(zip(result, sims)))
        execution_times += perf_counter() - start

    # Report results
    recall = 0
    avg_recall = 0
    for query_idx, result, result_gt in zip(query_idxs, results, results_gt):
        sims = [float(r[1]) for r in result]
        sims_gt = [float(r[1]) for r in result_gt]

        result = [r[0] for r in result]
        result_gt = [r[0] for r in result_gt]

        recall = len(set(result).intersection(set(result_gt))) / len(result_gt)
        query_term = terms[query_idx]
        avg_recall += recall
        print("----------------------------")
        print(f"Term: {query_term} | Recall: {recall}")

        term_with_sims = list(zip([terms[idx] for idx in result[:10]] , sims))
        term_with_sims_gt = list(zip([terms[idx] for idx in result_gt[:10]] , sims_gt))

        term_with_sims.sort(key=lambda x: x[1], reverse=False)
        term_with_sims_gt.sort(key=lambda x: x[1], reverse=True)

        print(f"LSH: {term_with_sims}")
        print(f" GT: {term_with_sims_gt}")
    avg_recall /= len(query_idxs)

    print("----------------------")

    print(f"Mean recall: {avg_recall}")
    print(f"Exec time: {execution_times / len(query_idxs)}")


def load_or_build_index(vectors, num_projections=2560):
    dims = vectors.shape[1]
    suffix = f"{len(vectors)}_{num_projections}_{dims}.npy"
    hashes_file = f"hashes_{suffix}"
    projs_file = f"projs_{suffix}"

    try:
        hashes = np.load(hashes_file)
        projs = np.load(projs_file)
        print(f"loaded from {hashes_file} and {projs_file}")
        return hashes, projs
    except FileNotFoundError:
        projs = create_projections(num_projections, dims)
        hashes = index(vectors, projs)
        np.save(hashes_file, hashes)
        np.save(projs_file, projs)
        print(f"saved {hashes_file} and {projs_file}")
        return hashes, projs


if __name__ == "__main__":
    terms, vectors = glove()

    hashes, projs = load_or_build_index(vectors)
    print("Done indexing")

    benchmark(terms, vectors, projs, hashes)
