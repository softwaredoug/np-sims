import pytest
from typing import Dict, Any

import numpy as np

from np_sims.rp_trees import fit, depth


vector_test_scenarios = {
    "base": {
        "vectors": np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
        "query": np.array([0, 1, 2]),
        "expected": 0
    }
}


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    return pytest.mark.parametrize(
        "vectors,query,expected",
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )


@w_scenarios(vector_test_scenarios)
def test_tree_building(vectors, query, expected):
    tree = fit(vectors, depth=1)
    assert depth(tree) == 1
