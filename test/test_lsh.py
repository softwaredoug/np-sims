import np_sims.lsh as lsh
import numpy as np

import pytest


def create_even_odd_dissimilar(num_projections, num_dims):
    """Create a vector that is dissimilar to the first half of the dimensions."""
    query = np.zeros(num_dims)
    query[0] = 0.8
    query[1] = 0.2
    query /= np.linalg.norm(query)

    # Tweak many vectors alike and different N times
    all_vectors = []
    similar_row_idxs = []
    different_row_idxs = []
    for _ in range(0, 100):
        different = np.zeros(num_dims)
        alike = np.zeros(num_dims)
        alike[0] = 1.0
        different[-1] = 1.0
        alike /= np.linalg.norm(alike)
        different /= np.linalg.norm(different)

        similar_row_idxs.append(len(all_vectors))
        all_vectors.append(alike)
        different_row_idxs.append(len(all_vectors))
        all_vectors.append(different)

    return query, np.array(all_vectors), similar_row_idxs, different_row_idxs


@pytest.mark.parametrize("seed", [0, 1000])
@pytest.mark.parametrize("num_projections", [64, 128, 256])
@pytest.mark.parametrize("num_dims", [50, 100, 300])
def test_projections_normalized(seed, num_projections, num_dims):
    """Test that projections are normalized."""
    np.random.seed(seed)
    projections = lsh.create_projections(num_projections, dims=num_dims)
    for proj in projections:
        assert pytest.approx(np.linalg.norm(proj)) == 1.0


@pytest.mark.parametrize("seed", [0, 1000])
@pytest.mark.parametrize("num_projections", [64, 128, 256, 2560])
@pytest.mark.parametrize("num_dims", [50, 100, 300])
@pytest.mark.parametrize("num_fronts", [None, 64, 256])
def test_index_and_query_gets_more_similar(seed, num_projections, num_dims, num_fronts):
    """Test that projections are normalized."""
    if num_fronts is not None and num_fronts <= num_projections:
        pytest.skip("Skipping illegal state (we should test another place")
    np.random.seed(seed)

    projections = lsh.create_projections(num_projections,
                                         dims=num_dims)
    query = np.zeros(num_dims)
    query[0] = 0.8
    query[1] = 0.2
    query /= np.linalg.norm(query)

    query, all_vectors, similar_row_idxs, different_row_idxs = create_even_odd_dissimilar(num_projections, num_dims)

    hashed = lsh.index(all_vectors, projections)
    if num_fronts is None:
        query_results, _ = lsh.query(query, hashed, projections)
    else:
        front_hashes = lsh.front_hashes(hashed, num_fronts)
        query_results, _ = lsh.query_two_phase(query, front_hashes, hashed, projections)

    for result in query_results:
        assert result in similar_row_idxs
        assert result not in different_row_idxs
