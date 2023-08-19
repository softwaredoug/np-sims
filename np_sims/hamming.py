import numpy as np


# Naive bitcount implementation without popcount

m1 = np.uint64(0x5555555555555555)
m2 = np.uint64(0x3333333333333333)
m3 = np.uint64(0x0F0F0F0F0F0F0F0F)
m4 = np.uint64(0x0101010101010101)

mask = np.uint64(-1)
# TODO - precompute type specific hashes
s55 = np.uint64(m1 & mask)  # Add more digits for 128bit support
s33 = np.uint64(m2 & mask)
s0F = np.uint64(m3 & mask)
s01 = np.uint64(m4 & mask)
num_bytes_64 = 8


def bit_count64(arr: np.ndarray[np.uint64]) -> np.ndarray:
    #
    # Taken from Mad Physicist on Stackoverflow:
    #   https://stackoverflow.com/a/68943135/8123
    #
    # Make the values type-agnostic (as long as it's integers)
    # baseline
    #
    # arr = arr - ((arr >> 1) & s55)
    # arr = (arr & s33) + ((arr >> 2) & s33)
    # arr = (arr + (arr >> 4)) & s0F
    # return (arr * s01) >> (8 * (arr.itemsize - 1))

    # Inplace, (ie -=) probably fastesr
    arr = arr - ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)

    # Using inplace operators where possible
    arr += (arr >> 4)
    arr &= s0F
    arr *= s01
    arr >>= (8 * (num_bytes_64 - 1))

    return arr
