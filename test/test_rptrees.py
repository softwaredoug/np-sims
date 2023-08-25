import pytest
from typing import Dict, Any, List

import numpy as np

from np_sims.rp_trees import RandomProjectionTree


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
def test_tree_building(vectors, query, expected_match, rng_seed):
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
