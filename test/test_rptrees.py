import pytest
from typing import Dict, Any, List

import numpy as np

from np_sims.rp_trees import RandomProjectionTree, _fit, _rp_hash, _hash_table


def as_normed_vects(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def num_unique(lst: List):
    return len(set(lst))


def each_unique(lst: List):
    return num_unique(lst) == len(lst)


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0,
        "rng_seed": 0,
    },
    "base_rng_1000": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0,
        "rng_seed": 1000,
    },
    "large": {
        "vectors": as_normed_vects(np.array([[0, 1.1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0,
        "rng_seed": 0,
    },
    "large_rng_100": {
        "vectors": as_normed_vects(np.array([[0, 1.1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0,
        "rng_seed": 100,
    }
}


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


@w_scenarios(vector_test_scenarios)
def test_fitting(vectors, query, expected_match, rng_seed):
    """Test internals of rp tree."""
    np.random.seed(rng_seed)
    for depth in range(0, 10):
        tree = _fit(vectors, depth=depth)
        dims = vectors.shape[1]
        vectors_hashed = np.zeros(vectors.shape, dtype=np.uint32)
        vectors_hashed = _rp_hash(tree, vectors, vectors_hashed, depth=depth)
        query_hash = np.zeros((1, dims), dtype=np.uint32)
        query_hash = _rp_hash(tree, query, query_hash, depth=depth)


@w_scenarios(vector_test_scenarios)
def test_rp_tree(vectors, query, expected_match, rng_seed):
    tree = RandomProjectionTree.build(vectors, seed=rng_seed)
    assert each_unique(tree.sorted_hashes)
    nearest = tree.query(query)[0]
    assert nearest == expected_match

    tree.save("/tmp/test.pkl")
    tree2 = RandomProjectionTree.load("/tmp/test.pkl")
    assert (tree2.sorted_hashes == tree.sorted_hashes).all()
    assert (tree2.sorted_idxs == tree.sorted_idxs).all()
    nearest2 = tree2.query(query)[0]
    assert nearest2 == expected_match
