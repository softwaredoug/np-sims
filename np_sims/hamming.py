import numpy as np


# Save masks to avoid recomputing
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
    """Count bits per element in a 64 bit array.

    Adapted from Mad Physicist's answer on Stackoverflow:
          https://stackoverflow.com/a/68943135/8123

    """
    # Inplace, (ie -=) probably fastesr
    arr -= ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)

    # Using inplace operators where possible
    arr += (arr >> 4)
    arr &= s0F
    arr *= s01
    arr >>= (8 * (num_bytes_64 - 1))

    return arr


def hamming_naive(hashes, query, count_function=bit_count64):
    """Compute an inverse hamming similarity score. 0 is identical, query.shape[1] * 64 is completely different.

    Returns
    -------
    array with each hash's similarity to query, for each row
     - 0 if identical to query, query.shape[1] * 64 if completely different

    normalize to 0-1 by dividing by query.shape[1] / 64

    Parameters
    ----------
    hashes are many bit arrays of LSH hashes corresponding to the vectors we've indexed
    query is a bit array of LSH hashee with the vector we've queried

    """
    # Note there are copies here, if you XOR'd in place, you might save some performance as then you
    # don't copy the giant hash table...
    #
    # but you also have to undo that XOR
    return np.sum(count_function(hashes ^ query), axis=1)
