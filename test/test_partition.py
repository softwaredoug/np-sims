"""Test space partitionining splits."""
import numpy as np
import pytest

from test_utils import as_normed_vects, w_scenarios, many_close_to
import glove_samples
from np_sims.partition import kdtree_maxvar_chooserule, kdtree_randdim_chooserule, projection_between_chooserule
from np_sims.partition import rptree_max_chooserule, rptree_proj_maxvar_chooserule, rptree_pca_chooserule
from np_sims.partition import KdTreeSplitRule, ProjectionBetweenSplitRule
from np_sims.partition import multi_projection_chooserule

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

    if np.dot(vectors, vectors.T).min() > 0.84:
        pytest.skip("Skipping test because vectors are too close together")

    splitter = rptree_max_chooserule(vectors)
    left, right = splitter.split(vectors)
    # assert len(left) != 0
    # assert len(right) != 0
    assert len(left) + len(right) == len(vectors)
    left_vectors = vectors[left]
    right_vectors = vectors[right]

    assert len(left) > 0
    assert len(right) > 0

    # Split by split point in teh direction of the projection
    assert (np.dot(left_vectors, splitter.projection) <= splitter.split_point).all()
    assert (np.dot(right_vectors, splitter.projection) > splitter.split_point).all()


def nn_on_correct_side(vectors, idx, left, right):
    """Check if idx is grouped with its neighbors on the correct side of the split."""
    vect = vectors[idx]
    dotted = np.dot(vectors, vect)
    nn = dotted.argsort()[-10:]
    left_nn_count = len(set(nn).intersection(set(left)))
    right_nn_count = len(set(nn).intersection(set(right)))

    if idx in right and right_nn_count > left_nn_count:
        return True
    elif idx in left and left_nn_count > right_nn_count:
        return True
    else:
        return False


def run_chooserule_scenario(vectors, seeds=[0], query_vectors=[],
                            chooserule=kdtree_randdim_chooserule, verbose=False):
    pass_count = 0
    runs = 0
    failed_seeds = set()
    for seed in seeds:

        np.random.seed(seed)
        splitter = chooserule(vectors)

        left, right = splitter.split(vectors)
        assert len(left) != 0
        assert len(right) != 0

        for vector_idx in query_vectors:
            print(f"Seed: {seed}, Vector: {vector_idx}")
            if nn_on_correct_side(vectors, vector_idx, left, right):
                pass_count += 1
                print("✅") if verbose else None
            else:
                failed_seeds.add(seed)
                print("❌") if verbose else None

            runs += 1

    pass_rate = pass_count / runs
    print(f"Pass rate: {pass_rate}")
    print(f"Failed seeds: {failed_seeds}")
    return pass_rate


# Force a seed for testing
@pytest.mark.parametrize("seed", [None])
@pytest.mark.parametrize("query_vectors", [None])
def test_pca_rptree_with_biased_glove(seed, query_vectors):
    vectors = np.load("test/glove_sample.npy")

    if seed is not None:
        seeds = range(seed, seed + 1)
    else:
        seeds = range(0, 400)

    if query_vectors is not None:
        query_vectors = range(query_vectors, query_vectors + 1)
    else:
        query_vectors = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 774]

    pass_rate = run_chooserule_scenario(vectors,
                                        seeds=seeds,
                                        query_vectors=query_vectors,
                                        chooserule=rptree_pca_chooserule,
                                        verbose=False)
    assert pass_rate > 0.68


@pytest.mark.parametrize("seed", [None])
@pytest.mark.parametrize("query_vectors", [None])
def test_rptree_with_biased_glove(seed, query_vectors):
    vectors = np.load("test/glove_sample.npy")
    if seed is not None:
        seeds = range(seed, seed + 1)
    else:
        seeds = range(0, 400)

    if query_vectors is not None:
        query_vectors = range(query_vectors, query_vectors + 1)
    else:
        query_vectors = range(0, 10)

    pass_rate = run_chooserule_scenario(vectors,
                                        seeds=seeds,
                                        query_vectors=query_vectors,
                                        chooserule=rptree_proj_maxvar_chooserule)
    assert pass_rate > 0.60


@pytest.mark.parametrize("seed", [None])
@pytest.mark.parametrize("query_vectors", [None])
def test_rptree_multiprojection(seed, query_vectors):
    vectors = np.load("test/glove_sample.npy")
    if seed is not None:
        seeds = range(seed, seed + 1)
    else:
        seeds = range(0, 400)

    if query_vectors is not None:
        query_vectors = range(query_vectors, query_vectors + 1)
    else:
        query_vectors = range(0, 1000)

    pass_rate = run_chooserule_scenario(vectors,
                                        seeds=seeds,
                                        query_vectors=query_vectors,
                                        chooserule=multi_projection_chooserule,
                                        verbose=False)
    assert pass_rate > 0.60


def test_rptree_with_glove_best_dim_chosen():
    vectors = np.load("test/glove_sample.npy")

    np.random.seed(0)
    splitter = rptree_proj_maxvar_chooserule(vectors, projection=glove_samples.king)

    left, right = splitter.split(vectors)
    assert len(left) != 0
    assert len(right) != 0
    assert len(left) + len(right) == len(vectors)

    king_idx = 774

    king = glove_samples.king
    king_nn = np.dot(vectors, king).argsort()[-10:]
    # king_fn = np.dot(vectors, king).argsort()[:10]

    king_right_nn_count = len(set(king_nn).intersection(set(right)))
    king_left_nn_count = len(set(king_nn).intersection(set(left)))

    # king_right_fn_count = len(set(king_fn).intersection(set(right)))
    # king_left_fn_count = len(set(king_fn).intersection(set(left)))

    if king_idx in right:
        assert king_right_nn_count == 10
    else:
        assert king_left_nn_count == 10
