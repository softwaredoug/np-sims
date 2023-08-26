import numpy as np

from np_sims.rp_trees import RandomProjectionTree, _fit, _rp_hash

from test_utils import as_normed_vects, many_close_to, w_scenarios


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 0,
    },
    "base_rng_1000": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 1000,
    },
    "a_bit_bigger": {
        "vectors": as_normed_vects(np.array([[0, 1.1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]
                                             ])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 0,
    },
    "a_bit_bigger_100": {
        "vectors": as_normed_vects(np.array([[0, 1.1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 100,
    },
    "identical_vectors": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4.5, 5], [6, 7.1, 8],
                                             [0, 1, 2], [3, 4.5, 5], [6, 7.1, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 0,
    },
    "many_closeby": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]
                                             ] + many_close_to([3.0, 4.0, 5.0]))),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "rng_seed": 0,
    },
}


@w_scenarios(vector_test_scenarios)
def test_fitting(vectors, query, rng_seed):
    """Test internals of rp tree."""
    np.random.seed(rng_seed)
    nearest_neighbor = np.dot(vectors, query[0])
    exact_nn = None
    if np.max(nearest_neighbor) == 1.0:
        exact_nn = np.argmax(nearest_neighbor)
    for depth in range(5, 40):
        tree = _fit(vectors, depth=depth)
        assert tree is not None
        vectors_hashed = np.zeros(len(vectors), dtype=np.uint64)
        vectors_hashed = _rp_hash(tree, vectors, vectors_hashed, depth=depth)
        query_hash = np.zeros(1, dtype=np.uint64)
        query_hash = _rp_hash(tree, query, query_hash, depth=depth)
        if exact_nn is not None:
            assert np.argwhere(vectors_hashed == query_hash)[0][0] == exact_nn


@w_scenarios(vector_test_scenarios)
def test_rp_tree(vectors, query, rng_seed):
    nearest_neighbor = np.dot(vectors, query[0])
    nearest_neighbor = np.argmax(nearest_neighbor)

    tree = RandomProjectionTree.build(vectors, seed=rng_seed)
    nearest = tree.query(query)[0]
    assert nearest == nearest_neighbor

    tree.save("/tmp/test.pkl")
    tree2 = RandomProjectionTree.load("/tmp/test.pkl")
    assert (tree2.sorted_hashes == tree.sorted_hashes).all()
    assert (tree2.sorted_idxs == tree.sorted_idxs).all()
    nearest2 = tree2.query(query)[0]
    assert nearest2 == nearest_neighbor


def test_with_flat_query_vector():
    vectors = as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]))
    query = as_normed_vects(np.array([[0, 1, 2]]))
    query_flattened = query[0]

    nearest_neighbor = np.dot(vectors, query_flattened)
    nearest_neighbor = np.argmax(nearest_neighbor)

    tree = RandomProjectionTree.build(vectors, seed=0)
    nearest_flattened = tree.query(query_flattened)[0]
    nearest_unflattened = tree.query(query)[0]

    assert nearest_flattened.shape == nearest_unflattened.shape
    assert nearest_flattened == nearest_neighbor
    assert nearest_unflattened == nearest_neighbor
