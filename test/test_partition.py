"""Test space partitionining splits."""
import numpy as np

from test_utils import as_normed_vects, w_scenarios, many_close_to
from np_sims.partition import kdtree_maxvar_chooserule, kdtree_randdim_chooserule, projection_between_chooserule
from np_sims.partition import rptree_max_chooserule
from np_sims.partition import KdTreeSplitRule, ProjectionBetweenSplitRule

# Larger dimensional test cases, fixture created with nearest and farthest neighbors of
# king
# vectors[np.random.choice(vectors.shape[0], 1000)] + vectors[king_nn] + vectors[king_fn]
# vectors[[ l for l in set(np.random.choice(vectors.shape[0], 1000).tolist() + king_fn.tolist() + king_nn.tolist())]]

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
    },
    "all_close": {
        "vectors": as_normed_vects(many_close_to([3.0, 4.0, 5.0])),
        "rng_seed": 1289519
    },
    "glove_sample": {
        "vectors": np.load("test/glove_sample.npy"),
        "rng_seed": 100
    }
}


@w_scenarios(vector_test_scenarios)
def test_partition_kdtree(vectors, rng_seed):
    np.random.seed(rng_seed)

    for rule in [kdtree_randdim_chooserule, kdtree_maxvar_chooserule]:
        splitter = rule(vectors)
        left, right = splitter.split(vectors)
        # assert len(left) != 0
        # assert len(right) != 0
        assert len(left) + len(right) == len(vectors)
        left_vectors = vectors[left]
        right_vectors = vectors[right]

        assert (left_vectors[:, splitter.dim] < splitter.median).all()
        assert (right_vectors[:, splitter.dim] >= splitter.median).all()


@w_scenarios(vector_test_scenarios)
def test_partition_proj_between(vectors, rng_seed):
    np.random.seed(rng_seed)

    max_proximity = 0.95
    splitter = projection_between_chooserule(vectors, max_proximity)
    left, right = splitter.split(vectors)
    # assert len(left) != 0
    # assert len(right) != 0
    assert len(left) + len(right) == len(vectors)
    left_vectors = vectors[left]
    right_vectors = vectors[right]

    if isinstance(splitter, KdTreeSplitRule):
        assert (left_vectors[:, splitter.dim] < splitter.median).all()
        assert (right_vectors[:, splitter.dim] >= splitter.median).all()
        # This should be true when we get a kdtree back
        # Dot product vectors together
        # dotted = np.dot(vectors, vectors.T)
        # assert (dotted >= max_proximity).all()
    elif isinstance(splitter, ProjectionBetweenSplitRule):
        assert (np.dot(left_vectors, splitter.projection) < 0).all()
        assert (np.dot(right_vectors, splitter.projection) >= 0).all()
    else:
        raise AssertionError("Expected either KdTreeSplitRule or ProjectionBetweenSplitRule")


@w_scenarios(vector_test_scenarios)
def test_partition_rptree_max(vectors, rng_seed):
    np.random.seed(rng_seed)

    splitter = rptree_max_chooserule(vectors)
    left, right = splitter.split(vectors)
    # assert len(left) != 0
    # assert len(right) != 0
    assert len(left) + len(right) == len(vectors)
    left_vectors = vectors[left]
    right_vectors = vectors[right]

    assert len(left) > 0
    assert len(right) > 0

    # Avg similarity of left and right should be higher
    # than similarity of left and right to each other
    left_avg_sim = np.mean(np.dot(left_vectors, left_vectors.T))
    right_avg_sim = np.mean(np.dot(right_vectors, right_vectors.T))
    all_avg_sim = np.mean(np.dot(vectors, vectors.T))

    assert left_avg_sim > all_avg_sim
    assert right_avg_sim > all_avg_sim
