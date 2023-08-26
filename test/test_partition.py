"""Test space partitionining splits."""
import numpy as np

from test_utils import as_normed_vects, w_scenarios, many_close_to
from np_sims.partition import kdtree_chooserule


vector_test_scenarios = {
    "base": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])),
        "rng_seed": 0,
    },
    "many_close": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]
                                            + many_close_to([3.0, 4.0, 5.0]))),
        "rng_seed": 0
    },
    "many_close_100": {
        "vectors": as_normed_vects(np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]
                                            + many_close_to([3.0, 4.0, 5.0]))),
        "rng_seed": 100
    }
}


@w_scenarios(vector_test_scenarios)
def test_partition_kdtree(vectors, rng_seed):
    np.random.seed(rng_seed)

    splitter = kdtree_chooserule(vectors)
    left, right = splitter(vectors)
    assert len(left) != 0
    assert len(right) != 0
    assert len(left) + len(right) == len(vectors)

    assert (left[:, splitter.dim] < splitter.median).all()
    assert (right[:, splitter.dim] >= splitter.median).all()
