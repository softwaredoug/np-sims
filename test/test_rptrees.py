import pytest
from typing import Dict, Any

import numpy as np

from np_sims.rp_trees import fit, rp_hash


def as_normed_vects(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0
    },
    "large": {
        "vectors": as_normed_vects(np.array([[0, 1.1, 2], [3, 4.1, 5], [6, 7.1, 8],
                                             [0, 1.5, 2], [3, 4.5, 5], [6, 7.5, 8]])),
        "query": as_normed_vects(np.array([[0, 1, 2]])),
        "expected_match": 0
    }
}


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


@w_scenarios(vector_test_scenarios)
def test_tree_building(vectors, query, expected_match):
    tree = fit(vectors, depth=10)
    vectors_hashed = rp_hash(tree, vectors, depth=10)
    assert len(set(vectors_hashed)) == vectors_hashed.shape[0]
