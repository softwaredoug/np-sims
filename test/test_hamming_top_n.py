import pytest
from typing import List

import numpy as np
from np_sims import hamming_top_10, hamming_top_cand


def same_elements(a, b):
    return set(a) == set(b)


UINT64_MAX = np.iinfo(np.uint64).max


def as_top_n(arr: List[int], n: int = 10) -> List[np.uint64]:
    as_64 = [np.uint64(x) for x in sorted(arr)[:n]]
    while len(as_64) < n:
        as_64.append(np.uint64(UINT64_MAX))
    return as_64


def repeated(x, n):
    return [x] * n


def rpt(x, n):
    return repeated(x, n)


def has_candidates(results, expected_cands):
    for idx in expected_cands:
        if idx in results and idx != UINT64_MAX:
            return True
    return False


hamming_tests = [
    # Tests with various hash lengths to test all the unrolled code paths
    ([[0b0010], [0b1101]], [0b0010], as_top_n([0, 1])),       # 0
    ([[0b0010, 0], [0b1101, 0]], [0b0010, 0], as_top_n([0, 1])),       # 1
    ([[0b0010, 0, 0], [0b1101, 0, 0]], [0b0010, 0, 0], as_top_n([0, 1])),     # 2
    ([[0b0010, 0, 0, 0], [0b1101, 0, 0, 0]], [0b0010, 0, 0, 0], as_top_n([0, 1])),    # 3
    ([[0b0010, 0, 0, 0, 0], [0b1101, 0, 0, 0, 0]], [0b0010, 0, 0, 0, 0], as_top_n([0, 1])),  # 4

    ([[0b0010, 0, 0, 0, 0, 0], [0b1101, 0, 0, 0, 0, 0]], [0b0010, 0, 0, 0, 0, 0], as_top_n([0, 1])),  # 5

    ([[0b0010] + [0] * 20, [0b1101] + [0] * 20],
     [0b0010] + [0] * 20,
     as_top_n([0, 1])),          # 6

    # Detect cases where we dereference the pointer incorrectly (add one to teh value, not teh pointer)
    ([[0b0001, 0b0001],  # Very dissimilar, but QUERY turns int 0b0001 0b0001
      [0b0000, 0b0000], [0b0000, 0b0000], [0b0000, 0b0000],
      [0b0000, 0b0000], [0b0000, 0b0000], [0b0000, 0b0000], [0b0000, 0b0001], [0b0000, 0b0000], [0b0000, 0b0000],
      [0b0000, 0b0000]],
     [0b0000, 0b0000],  # Due to bad dereference, turns into 0b0000 0b0001
     as_top_n([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])),    # 7

    # Various cases with very long hashes, probably not unrolled
    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [1],
     as_top_n([0, 1])),       # 8

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1])),         # 9

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3])),    # 10

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3, 4, 5, 6])),      # 11

    ([[0b0010] + [0] * 20 + [1], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b1101] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),   # 12

    # Long hash, most similar at end, but not exclusively
    ([[0b0010] + [UINT64_MAX] * 20 + [1], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b0010] + [0] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b0010] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0], [0b1101] + [UINT64_MAX] * 20 + [0],
      [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0], [0b1101] + [0] * 20 + [0],
      [0b0010] + [0] * 20 + [0]],
     [0b0010] + [0] * 20 + [0],
     as_top_n([0, 10, 11, 3, 12, 5, 6, 7, 8, 9])),   # 13

    ([[UINT64_MAX], [UINT64_MAX], [UINT64_MAX],  # should NOT be included
      [0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]],
     [0b0010],  # query
     as_top_n([3, 4, 5, 6, 7, 8, 9, 10, 11, 12])),  # all but the first ones     # 14

    ([[0], [UINT64_MAX]], [0], as_top_n([0, 1])),   # 15

    # Should detect the first 10
    ([[0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]]
     + ([[UINT64_MAX]] * 10),   # 16
     [0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all the first ones

    # Should detect the first 10
    ([[0b0010], [0b1101], [0b1101], [0b1111], [0b0010], [0b1101], [0b1101], [0b1111], [0b0011], [0b0110]]
     + ([[UINT64_MAX]] * 1000),             # 17
     [0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all but the first 3

    # Test with multiple hashes per row
    # Should detect the first 3
    ([[0b1111, 0b0010], [0b1101, 0b1101]],        # 18
     [0b1111, 0b0010],  # query
     as_top_n([0, 1])),  # all but the first 3

    # Test with multiple hashes per row
    # Should detect the first 10
    ([[0b1111, 0b0010], [0b1101, 0b1101], [0b1011, 0b1101], [0b1010, 0b1111], [0b1101, 0b0010], [0b1001, 0b1101], [0b0001, 0b1101], [0b0000, 0b1111], [0b0010, 0b0011], [0b1101, 0b0110]]
     + ([[UINT64_MAX, UINT64_MAX]] * 1000),        # 19
     [0b1111, 0b0010],  # query
     as_top_n([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])),  # all but the first 3

    # Test with 128-bit hashes for simd
    # Should detect the last 10
    ([[UINT64_MAX, UINT64_MAX], [UINT64_MAX, UINT64_MAX], [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX], [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, 0b0000],      # 20
      [0b1101, 0b1101], [0b1011, 0b1101], [0b1010, 0b1111], [0b1101, 0b0010], [0b1001, 0b1101], [0b0001, 0b1101], [0b0000, 0b1111], [0b0010, 0b0011], [0b1101, 0b0110], [0b0000, 0b0010]]
     + ([[UINT64_MAX, UINT64_MAX]] * 1000),
     [0b0000, 0b0010],  # query
     as_top_n([10, 11, 12, 13, 14, 15, 16, 17, 18, 19])),  # the last 10

    # Actually looking for UINT64_MAX, but should detect the last 10
    # Should detect the last 10
    ([[0b1101, 0b1101], [0b1011, 0b1101], [0b1010, 0b1111], [0b1101, 0b0010], [0b1001, 0b1101], [0b0001, 0b1101],
      [0b0000, 0b1111], [0b0010, 0b0011], [0b1101, 0b0110], [0b0000, 0b0010], [0b0000, 0b0010], [0b0000, 0b0010],
      [UINT64_MAX, UINT64_MAX], [UINT64_MAX, UINT64_MAX], [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX], [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, 0b0000]]       # 21
     + ([[0b0000, 0b0010]] * 1000),
     [UINT64_MAX, UINT64_MAX],  # query
     as_top_n([12, 13, 14, 15, 16, 17, 18, 19, 20, 21])),  # the last 10

    # Same as last, but a very large number
    ([[0b1101, 0b1101], [0b1011, 0b1101], [0b1010, 0b1111], [0b1101, 0b0010], [0b1001, 0b1101], [0b0001, 0b1101],
      [0b0000, 0b1111], [0b0010, 0b0011], [0b1101, 0b0110], [0b0000, 0b0010], [0b0000, 0b0010], [0b0000, 0b0010],
      [0b0111, 0b000], [0b0000, 0b0000], [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, UINT64_MAX], [0b1111, 0b0010], [UINT64_MAX, UINT64_MAX],
      [0b1111, UINT64_MAX], [UINT64_MAX, 0b0000]]     # 22
     + ([[0b0000, 0b0010]] * 10000)
     + ([[UINT64_MAX, 0b0010]] * 100)
     + ([[0b0000, 0b0010]] * 10000)
     + ([[0b0000, UINT64_MAX]] * 100)
     + ([[0b0000, 0b0010]] * 10000)
     + ([[0b0000, UINT64_MAX]] * 100),
     [UINT64_MAX, UINT64_MAX],  # query
     as_top_n([10022, 10023, 10024, 16, 19, 17, 20, 14, 10025, 15])),

    # Very large hashes
    # This once segfaulted
    ([
     rpt(0b1101, 40) + [0b0000, 0b0000],         # 23
     rpt(0b0000, 40) + [0b0000, 0b0000]]
     + [rpt(0b0000, 40) + [0b0000, 0b0000]] * 100000
     + [rpt(UINT64_MAX, 40) + [UINT64_MAX, 0b0000]],  # < the match

     rpt(UINT64_MAX, 40) + [UINT64_MAX, 0b0000],
     as_top_n([0, 2, 3, 4, 5, 6, 7, 8, 9, 100002])),  # the last 10
]


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming_top_n(hashes, query, expected):
    result = hamming_top_10(np.array(hashes, dtype=np.uint64),
                            np.array(query, dtype=np.uint64))
    assert same_elements(result, expected)


@pytest.mark.parametrize("hashes, query, expected", hamming_tests)
def test_hamming_top_cand(hashes, query, expected):
    result = hamming_top_cand(np.array(hashes, dtype=np.uint64),
                              np.array(query, dtype=np.uint64))
    assert has_candidates(result, expected)
