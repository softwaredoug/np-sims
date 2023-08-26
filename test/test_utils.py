from typing import List, Dict, Any

import pytest
import numpy as np


def as_normed_vects(vectors):
    return vectors / np.linalg.norm(vectors, axis=1)[:, None]


def num_unique(lst: List):
    return len(set(lst))


def each_unique(lst: List):
    return num_unique(lst) == len(lst)


def many_close_to(lst: List, n=100):
    """Create many variations of a single vector barely similar to each other."""
    epsilon = 1.0001
    arr = np.array(lst)
    lst = [arr * epsilon ** i for i in range(n)]
    lst = np.array(lst).tolist()
    return lst


def random_vector(n=3):
    """Return random unit vector of n dimensions."""
    vect = np.random.normal(size=n)
    vect /= np.linalg.norm(vect)
    return vect


def w_scenarios(scenarios: Dict[str, Dict[str, Any]]):
    """Decorate for parametrizing tests that names the scenarios and params."""
    return pytest.mark.parametrize(
        [key for key in scenarios.values()][0].keys(),
        [tuple(scenario.values()) for scenario in scenarios.values()],
        ids=list(scenarios.keys())
    )
