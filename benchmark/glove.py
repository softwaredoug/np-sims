import numpy as np

from data_dir import read_np, write_np, read_lines, write_lines, lines_of
from time import perf_counter
import cProfile
import pstats
from concurrent.futures import ThreadPoolExecutor, as_completed

# Get current working dir
import os
import sys
cwd = os.getcwd()
sys.path.append(cwd)

from np_sims.lsh import index, create_projections  # noqa: E402
from np_sims.lsh import query_with_hamming_then_slow_argsort, query_pure_python, query_with_hamming_top_n  # noqa: E402


def glove(num_to_sample=10000000):
    terms = []
    vectors = []

    terms_file = "glove.840B.300d_terms"
    vectors_file = "glove.840B.300d_vectors"

    try:
        terms = read_lines(terms_file)[:num_to_sample]
        vectors = read_np(vectors_file)[:num_to_sample]
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


def benchmark(terms, vectors, projs, hashes, query_fn, debug=False, workers=4):

    # Randomly select 100 terms
    query_idxs = np.random.randint(0, len(terms), size=100)

    # Collect groundtruths
    results_gt = []
    for query_idx in query_idxs:
        result, sims = most_similar_cos(vectors, query_idx)
        results_gt.append(list(zip(result, sims)))

    zero_sims = [0] * 10

    def query(query_idx):
        start = perf_counter()
        result, sims = query_fn(vectors[query_idx], hashes, projs)
        return result, sims, perf_counter() - start

    with cProfile.Profile() as pr:
        # Run LSH
        results = []
        execution_times = 0     # Total wall time running the queries
        single_query_times = 0  # Time per query in its thread
        futures = []
        with ThreadPoolExecutor(max_workers=workers) as executor:
            start = perf_counter()
            for query_idx in query_idxs:
                futures.append(executor.submit(query, query_idx))

            for future in as_completed(futures):
                result, sims, time = future.result()
                stop = perf_counter()
                single_query_times += time
                if sims is None:
                    sims = zero_sims
                results.append(list(zip(result, sims)))
            execution_times = stop - start

        pr.dump_stats("top_n.prof")

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
        print("  ----------------------------") if debug else None
        print(f"Term: {query_term} | Recall: {recall}") if debug else None

        term_with_sims = list(zip([terms[idx] for idx in result[:10]] , sims))
        term_with_sims_gt = list(zip([terms[idx] for idx in result_gt[:10]] , sims_gt))

        term_with_sims.sort(key=lambda x: x[1], reverse=False)
        term_with_sims_gt.sort(key=lambda x: x[1], reverse=True)

        print(f"LSH: {term_with_sims}") if debug else None
        print(f" GT: {term_with_sims_gt}") if debug else None
    avg_recall /= len(query_idxs)

    time_per_query = execution_times / len(query_idxs)
    qps = len(query_idxs) / execution_times
    pstats.Stats("top_n.prof").strip_dirs().sort_stats("cumulative").print_stats(20) if debug else None
    print("Run num queries: ", len(query_idxs))
    print(f"Mean recall: {avg_recall}")
    print(f"  Exec time: {time_per_query}")
    print(f"        QPS: {qps}")
    print("----------------------")

    return qps, avg_recall


def load_or_build_index(vectors, num_projections=640):
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
        write_np(hashes_file, hashes)
        write_np(projs_file, projs)
        print(f"saved {hashes_file} and {projs_file}")
        return hashes, projs


if __name__ == "__main__":
    terms, vectors = glove()
    query_method = "pure_python" if len(sys.argv) < 2 else sys.argv[1]

    if query_method == "pure_python":
        query_fn = query_pure_python
    elif query_method == "pure_c":
        query_fn = query_with_hamming_top_n
    elif query_method == "hamming_c":
        query_fn = query_with_hamming_then_slow_argsort

    results = []
    for projections in sys.argv[2:]:
        num_projections = int(projections)
        print("----------------------")
        print(f"Running with {num_projections} projections")

        hashes, projs = load_or_build_index(vectors, num_projections)
        print("Done indexing")

        results.append((num_projections, benchmark(terms, vectors, projs, hashes, query_fn)))

    for num_projections, result in results:
        qps, recall = result
        print(f"{num_projections} projections -- QPS {qps} -- Recall {recall}")


# -------------------------------
# Loop unrolled
# 64 projections -- QPS 840.2823049187605 -- Recall 0.1289999999999998
# 128 projections -- QPS 594.8292691966128 -- Recall 0.193
# 256 projections -- QPS 365.12440313724113 -- Recall 0.25200000000000006
# 512 projections -- QPS 194.034303547326 -- Recall 0.40800000000000003
# 640 projections -- QPS 146.14705980853586 -- Recall 0.41000000000000014
# 1280 projections -- QPS 57.83595190887417 -- Recall 0.5739999999999997
# 2560 projections -- QPS 38.97597500828167 -- Recall 0.6549999999999996
