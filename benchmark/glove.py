import numpy as np

from data_dir import read_np, write_np, read_lines, write_lines, lines_of
from time import perf_counter

from np_sims.lsh import index, query, create_projections


def glove(num_to_sample=10000000):
    terms = []
    vectors = []

    terms_file = "glove.840B.300d_terms"
    vectors_file = "glove.840B.300d_vectors"

    try:
        terms = read_lines(terms_file)
        vectors = read_np(vectors_file)
        return terms, vectors
    except FileNotFoundError:
        for line in lines_of("glove.840B.300d"):
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
        write_np(vectors_file, vectors)
        write_lines(terms_file, terms)
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
    suffix = f"{len(vectors)}_{num_projections}_{dims}"
    hashes_file = f"hashes_{suffix}"
    projs_file = f"projs_{suffix}"

    try:
        hashes = read_np(hashes_file)
        projs = read_np(projs_file)
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
