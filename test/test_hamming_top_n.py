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
    ([[0b0010], [0b1101]], [0b0010], as_top_n([0, 1])),
    ([[UINT64_MAX], [UINT64_MAX], [UINT64_MAX],  # should NOT be included
      [0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]],
     [0b0010],  # query
     as_top_n([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),  # all but the first 3
    ([[0], [UINT64_MAX]], [0], as_top_n([0, 1])),
]


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming_top_n(hashes, query, expected):
    result = hamming_top_n(np.array(hashes, dtype=np.uint64),
                           np.array(query, dtype=np.uint64))
    assert same_elements(result, expected)
