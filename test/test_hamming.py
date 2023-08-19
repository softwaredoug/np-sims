import pytest
import numpy as np
from np_sims.hamming import bit_count64, hamming_naive
from np_sims import hamming_c


bit_tests = [
    ([0b0010], np.uint64(1)),
    ([0b0111], np.uint64(3)),
    ([np.iinfo(np.uint64).max], np.uint64(64)),
    ([np.iinfo(np.uint64).min], np.uint64(0)),
    ([np.iinfo(np.uint64).max, np.iinfo(np.uint64).max], [np.uint64(64), np.uint64(64)]),
]


@pytest.mark.parametrize("bit_array, expected", bit_tests)
def test_naive_bitcount(bit_array, expected):
    assert (bit_count64(np.array(bit_array, dtype=np.uint64)) == expected).all()


# Given hashes (first param) compute number of DISSIMILAR bits
hamming_tests = [
    ([[0b0010]],           [0b0010], np.uint64(0)),  # noqa: E241
    ([[0b0010], [0b1101]], [0b0010], [np.uint64(0), np.uint64(4)]),
    ([[0], [np.iinfo(np.uint64).max]], [0], [np.uint64(0), np.uint64(64)]),
]


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming(hashes, query, expected):
    assert (hamming_naive(np.array(hashes, dtype=np.uint64),
                          np.array(query, dtype=np.uint64)) == expected).all()


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming_c(hashes, query, expected):
    assert (hamming_c(np.array(hashes, dtype=np.uint64),
                      np.array(query, dtype=np.uint64)) == expected).all()
