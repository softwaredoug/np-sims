import numpy as np
import argparse

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
from np_sims.lsh import query_pure_python, query_with_hamming_top_n  # noqa: E402
from np_sims.lsh import front_hashes, query_with_hamming_top_n_two_phase  # noqa: E402
from np_sims.rp_trees import RandomProjectionTree  # noqa: E402
from np_sims.partition import kdtree_maxvar_chooserule, rptree_max_chooserule, rptree_pca_chooserule  # noqa: E402


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


def lsh_query_fn(query_method, hashes, projs, num_fronts=None):
    if query_method == "lsh_pure_python":
        return lambda query_vector: query_pure_python(query_vector, hashes, projs)
    elif query_method == "lsh_pure_c":
        return lambda query_vector: query_with_hamming_top_n(query_vector, hashes, projs)
    elif query_method == "lsh_rerank_cand":
        front = front_hashes(hashes, num_fronts)
        return lambda query_vector: query_with_hamming_top_n_two_phase(query_vector, front, hashes, projs)


def rp_tree_query_fn(query_method, rp_tree):
    return lambda query_vector: rp_tree.query(query_vector)


def parse_algorithm_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm",
                        nargs="+",
                        type=str,
                        help="The algorithm (and possible variant) to use this test")
    parser.add_argument("--num_projections",
                        nargs="+",
                        type=int,
                        help="For LSH, the number of projections to use")
    parser.add_argument("--num_fronts",
                        nargs="+",
                        type=int,
                        default=None,
                        help="For LSH rerank, the first pass projection to use")

    parser.add_argument("--max_depth",
                        type=int,
                        default=10,
                        help="For RpTree, max depth of tree")
    parser.add_argument("--max_leaf_size",
                        type=int,
                        default=100,
                        help="For RpTree, max leaf size of tree")

    parser.add_argument("--workers",
                        nargs=1,
                        default=1,
                        type=int,
                        help="Number of workers")
    parser.add_argument("--verbose",
                        action="store_true",
                        help="Enable verbose debug output")

    parser.add_argument("--seed",
                        default=None,
                        type=int,
                        help="Enable verbose debug output")
    parser.add_argument("--num_queries",
                        type=int,
                        default=1000,
                        help="Number of queries to run")

    args = parser.parse_args()
    return args


def test_train_split(terms, vectors, num_queries):
    """Split off num_queries from vectors and remove them."""
    query_idxs = np.random.randint(0, len(terms), num_queries)
    query_vectors = vectors[query_idxs]
    query_terms = [terms[idx] for idx in query_idxs]
    # Remove from vectors
    index_vectors = np.delete(vectors, query_idxs, axis=0)
    index_terms = np.delete(terms, query_idxs, axis=0)
    return query_terms, query_vectors, index_vectors, index_terms


def get_test_algorithms(cmd_args, terms, vectors):
    for query_method in cmd_args.algorithm:
        if query_method in ["lsh_pure_python", "lsh_pure_c", "lsh_rerank_cand"]:
            fronts = cmd_args.num_fronts
            if fronts is None:
                fronts = [None] * len(cmd_args.num_projections)
            for fronts, projections in zip(fronts, cmd_args.num_projections):
                num_projections = int(projections)
                print("----------------------")
                print(f"Running with {num_projections} projections")

                hashes, projs = load_or_build_index(vectors, num_projections)
                name = f"{query_method}_{num_projections}"
                if fronts is not None:
                    name += f"_fronts_{fronts}"
                yield name, lsh_query_fn(query_method, hashes, projs, fronts)
        elif query_method in ["kd_tree", "rp_tree", "pca_tree"]:
            start = perf_counter()
            chooserule = None
            if query_method == "kd_tree":
                chooserule = kdtree_maxvar_chooserule
            elif query_method == "rp_tree":
                chooserule = rptree_max_chooserule
            elif query_method == "pca_tree":
                chooserule = rptree_pca_chooserule

            tree = RandomProjectionTree.build(vectors,
                                              depth=cmd_args.max_depth,
                                              max_leaf_size=cmd_args.max_leaf_size,
                                              chooserule=chooserule)
            print(f"Building tree took {perf_counter() - start} seconds")
            # tree.save('data/rp_tree.pkl')
            term_idx = terms.index("king\n")
            tree.debug_dump(terms, vectors, term_idx)
            yield "rp_tree", rp_tree_query_fn(query_method, tree)
        else:
            raise ValueError(f"Unknown algorithm: {query_method}")


def benchmark(terms, vectors, query_fn, workers,
              debug=False, ground_truth=True, num_queries=50):

    query_idxs = np.random.randint(0, len(terms), size=num_queries)

    # Collect groundtruths
    gt_start = perf_counter()
    gt_time = 0
    results_gt = {}
    for query_idx in query_idxs:
        if ground_truth:
            result, sims = most_similar_cos(vectors, query_idx)
        else:
            result = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            sims = [0] * 10
        gt_time += (perf_counter() - gt_start)
        results_gt[query_idx] = list(zip(result, sims))

    def query(query_idx):
        pr = cProfile.Profile()
        pr.enable()
        my_start = perf_counter()
        result, sims = query_fn(vectors[query_idx])
        pr.disable()
        return query_idx, result, sims, perf_counter() - my_start, pr

    # Run LSH
    results = {}
    execution_times = 0     # Total wall time running the queries
    single_query_times = 0  # Time per query in its thread
    futures = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        start = perf_counter()
        for query_idx in query_idxs:
            futures.append(executor.submit(query, query_idx))

        pr_stats_all = pstats.Stats()
        for future in as_completed(futures):
            query_idx, result, sims, time, pr = future.result()
            stop = perf_counter()
            pr_stats_all.add(pr)
            single_query_times += time
            if sims is None:
                sims = [0] * len(result)
            results[query_idx] = list(zip(result, sims))
        execution_times = stop - start

        pr_stats_all.dump_stats("top_n.prof")

    # Report results
    recall = 0
    avg_recall_10 = 0
    avg_recall_max = 0
    max_len = 0
    for query_idx in query_idxs:
        result = results[query_idx]
        result_gt = results_gt[query_idx]

        sims = [float(r[1]) for r in result]
        sims_gt = [float(r[1]) for r in result_gt]

        result = [r[0] for r in result]
        result_gt = [r[0] for r in result_gt]

        recall_10 = len(set(result[:10]).intersection(set(result_gt))) / len(result_gt)
        recall_max = len(set(result).intersection(set(result_gt))) / len(result_gt)
        max_len = len(result)
        query_term = terms[query_idx]
        avg_recall_10 += recall_10
        avg_recall_max += recall_max
        print("  ----------------------------") if debug else None
        print(f"Term: {query_term} | Recall: {recall}") if debug else None

        term_with_sims = list(zip([terms[idx] for idx in result[:10]] , sims))
        term_with_sims_gt = list(zip([terms[idx] for idx in result_gt[:10]] , sims_gt))

        term_with_sims.sort(key=lambda x: x[1], reverse=False)
        term_with_sims_gt.sort(key=lambda x: x[1], reverse=True)

        print(f"LSH: {term_with_sims}") if debug else None
        print(f" GT: {term_with_sims_gt}") if debug else None
    avg_recall_10 /= len(query_idxs)
    avg_recall_max /= len(query_idxs)

    qps = len(query_idxs) / execution_times
    gt_qp = len(query_idxs) / gt_time
    pstats.Stats("top_n.prof").strip_dirs().sort_stats("cumulative").print_stats(20) if debug else None
    print("Run num queries: ", len(query_idxs))
    print(f"Mean recall@10    : {avg_recall_10}")
    if max_len != 10:
        print(f"Mean recall@{max_len}>10: {avg_recall_max}")
    print(f"     GT QPS       : {gt_qp}")
    print(f"        QPS       : {qps}")
    print("----------------------")

    return qps, avg_recall_10


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


def main():
    terms, vectors = glove()
    cmd_args = parse_algorithm_args()

    if cmd_args.seed is not None:
        np.random.seed(cmd_args.seed)

    # query_terms, query_vectors, index_terms, index_vectors = test_train_split(terms, vectors, cmd_args.query_size)

    results = []
    for name, query_fn in get_test_algorithms(cmd_args, terms, vectors):
        print(f"Running {name}")
        results.append((name, benchmark(terms, vectors,
                                        query_fn=query_fn,
                                        workers=cmd_args.workers[0],
                                        debug=cmd_args.verbose,
                                        num_queries=cmd_args.num_queries)))

    for name, result in results:
        qps, recall = result
        print(f"Algo: {name} -- QPS {qps} -- Recall {recall}")


if __name__ == "__main__":
    main()


# LSH Benchmarks
# Doug's m1

# --------------------------------
# Pure Python
#   64 projections -- QPS 15.089427548862796 -- Recall 0.13459999999999817
#   128 projections -- QPS 9.34182765170921 -- Recall 0.188299999999998
#   256 projections -- QPS 6.847121609496162 -- Recall 0.2648999999999981
#   512 projections -- QPS 5.061889430500477 -- Recall 0.36490000000000017
#   640 projections -- QPS 4.417822593950075 -- Recall 0.4122999999999998
#   1280 projections -- QPS 2.6906025585082607 -- Recall 0.5301999999999994
#   2560 projections -- QPS 1.2121089854928582 -- Recall 0.6400999999999979

# -------------------------------
# Just C, no optimizations
#
#
#    64 projections -- QPS 355.0464873171851 -- Recall 0.136499999999998
#    128 projections -- QPS 241.46964706002143 -- Recall 0.19549999999999798
#    256 projections -- QPS 146.8177150077879 -- Recall 0.283999999999999
#    512 projections -- QPS 130.2396482333485 -- Recall 0.38470000000000054
#    640 projections -- QPS 96.629607995171 -- Recall 0.4202000000000004
#    1280 projections -- QPS 55.37952621776572 -- Recall 0.5329
#    2560 projections -- QPS 29.13870808612148 -- Recall 0.6328999999999989
#
#
#
#
# -------------------------------
# Loop unrolled
# 64 projections -- QPS 840.2823049187605 -- Recall 0.1289999999999998
# 128 projections -- QPS 594.8292691966128 -- Recall 0.193
# 256 projections -- QPS 365.12440313724113 -- Recall 0.25200000000000006
# 512 projections -- QPS 194.034303547326 -- Recall 0.40800000000000003
# 640 projections -- QPS 146.14705980853586 -- Recall 0.41000000000000014
# 1280 projections -- QPS 57.83595190887417 -- Recall 0.5739999999999997
# 2560 projections -- QPS 38.97597500828167 -- Recall 0.6549999999999996
#
#
#
# -------------------------------
# Threading enabled (ie not GIL bound)
#
# 64 projections -- QPS 2395.0580356424443 -- Recall 0.12969999999999815
# 128 projections -- QPS 1556.6209180039205 -- Recall 0.19139999999999796
# 256 projections -- QPS 870.3183265554875 -- Recall 0.2803999999999986
# 512 projections -- QPS 499.93975726613667 -- Recall 0.3816000000000004
# 640 projections -- QPS 358.7610832214154 -- Recall 0.41790000000000066
# 1280 projections -- QPS 167.48470966960681 -- Recall 0.5302000000000001
# 2560 projections -- QPS 81.02739932963804 -- Recall 0.6396999999999985
#
# -------------------------------
# Using SIMD (dropped)
# Algo: lsh_pure_c_64 -- QPS 2550.8807669409057 -- Recall 0.1380999999999981
# Algo: lsh_pure_c_128 -- QPS 1585.288417006297 -- Recall 0.1870999999999977
# Algo: lsh_pure_c_256 -- QPS 830.3471393242446 -- Recall 0.27399999999999886
# Algo: lsh_pure_c_512 -- QPS 417.7410735582799 -- Recall 0.3799000000000006
# Algo: lsh_pure_c_640 -- QPS 348.9405170382391 -- Recall 0.4237000000000006
# Algo: lsh_pure_c_1280 -- QPS 145.09469040393756 -- Recall 0.5310999999999997
# Algo: lsh_pure_c_2560 -- QPS 83.26799802170994 -- Recall 0.6331999999999985
