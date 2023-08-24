import pytest
from typing import Dict, Any

import numpy as np

from np_sims.rp_trees import fit, rp_hash


def as_normed_vects(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)


def str_to_bits(s):
    bits = np.uint64(0)
    for idx, curr_char in enumerate(s):
        if curr_char == "1":
            bits |= np.uint64(1) << (len(s) - idx - 1)
    return np.array([bits])


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_query_hash": str_to_bits("000"),
    }
}


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


@w_scenarios(vector_test_scenarios)
def test_tree_building(vectors, query, expected_query_hash):
    tree = fit(vectors, depth=3)
    vectors_hashed = rp_hash(tree, vectors, depth=3)
    assert len(set(vectors_hashed)) == vectors_hashed.shape[0]
