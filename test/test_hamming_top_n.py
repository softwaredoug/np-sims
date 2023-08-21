import pytest
from typing import List

import numpy as np
from np_sims import hamming_top_n


def same_elements(a, b):
    return set(a) == set(b)


UINT64_MAX = np.iinfo(np.uint64).max


def as_top_n(arr: List[int], n: int = 10) -> List[np.uint64]:
    as_64 = [np.uint64(x) for x in sorted(arr)[:n]]
    while len(as_64) < n:
        as_64.append(np.uint64(UINT64_MAX))
    return as_64


hamming_tests = [
    # Tests with various hash lengths to test all the unrolled code paths
    ([[0b0010], [0b1101]], [0b0010], as_top_n([0, 1])),
    ([[0b0010, 0], [0b1101, 0]], [0b0010, 0], as_top_n([0, 1])),
    ([[0b0010, 0, 0], [0b1101, 0, 0]], [0b0010, 0, 0], as_top_n([0, 1])),
    ([[0b0010, 0, 0, 0], [0b1101, 0, 0, 0]], [0b0010, 0, 0, 0], as_top_n([0, 1])),
    ([[0b0010, 0, 0, 0, 0], [0b1101, 0, 0, 0, 0]], [0b0010, 0, 0, 0, 0], as_top_n([0, 1])),
    ([[0b0010, 0, 0, 0, 0, 0], [0b1101, 0, 0, 0, 0, 0]], [0b0010, 0, 0, 0, 0, 0], as_top_n([0, 1])),
    ([[0b0010] + [0] * 20, [0b1101] + [0] * 20],
     [0b0010] + [0] * 20,
     as_top_n([0, 1])),

    # Various cases with very long hashes, probably not unrolled
    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [1],
     as_top_n([0, 1])),
    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1])),

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3])),

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3, 4, 5, 6])),

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),

    # Long hash, most similar at end, but not exclusively
    ([[0b0010] + [UINT64_MAX] * 20 + [1], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b0010] + [0] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b0010] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b0010] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 10, 11, 3, 12, 5, 6, 7, 8, 9])),

    ([[UINT64_MAX], [UINT64_MAX], [UINT64_MAX],  # should NOT be included
      [0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]],
     [0b0010],  # query
     as_top_n([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),  # all but the first 3

    ([[0], [UINT64_MAX]], [0], as_top_n([0, 1])),

    # Should detect the first 10
    ([[0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]]
     + ([[UINT64_MAX]] * 10),
     [0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all the first ones

    # Should detect the first 10
    ([[0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]]
     + ([[UINT64_MAX]] * 1000),
     [0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all but the first 3

    # Test with multiple hashes per row
    # Should detect the first 3
    ([[0b1111, 0b0010], [0b1101, 0b1101]],
     [0b1111, 0b0010],  # query
     as_top_n([0, 1])),  # all but the first 3

    # Test with multiple hashes per row
    # Should detect the first 10
    ([[0b1111, 0b0010], [0b1101, 0b1101], [0b1011, 0b1101], [0b1010, 0b1111], [0b1101, 0b0010], [0b1001, 0b1101], [0b0001, 0b1101], [0b0000, 0b1111], [0b0010, 0b0011], [0b1101, 0b0110]]
     + ([[UINT64_MAX, UINT64_MAX]] * 1000),
     [0b1111, 0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all but the first 3
]


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming_top_n(hashes, query, expected):
    result = hamming_top_n(np.array(hashes, dtype=np.uint64),
                           np.array(query, dtype=np.uint64))
    assert same_elements(result, expected)
