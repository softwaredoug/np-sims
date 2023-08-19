import timeit
import numpy as np
from doug_ufunc import num_unshared_bits, hamming as hamming_c_ufunc

# Naive bitcount implementation without popcount

m1 = np.int64(0x5555555555555555)
m2 = np.int64(0x3333333333333333)
m3 = np.int64(0x0F0F0F0F0F0F0F0F)
m4 = np.int64(0x0101010101010101)

mask = np.int64(-1)
# TODO - precompute type specific hashes
s55 = np.int64(m1 & mask)  # Add more digits for 128bit support
s33 = np.int64(m2 & mask)
s0F = np.int64(m3 & mask)
s01 = np.int64(m4 & mask)
num_bytes_64 = 8


def bit_count64(arr):
    # Make the values type-agnostic (as long as it's integers)
    # baseline
    #
    # -arr = arr - ((arr >> 1) & s55)
    # +arr -= ((arr >> 1) & s55)
    #
    # before
    #     5106   32.987    0.006   32.987    0.006 hamming.py:27(bit_count64)
    #     5106   32.436    0.006   32.436    0.006 hamming.py:26(bit_count64)
    #     5106   35.277    0.007   35.277    0.007 hamming.py:26(bit_count64)

    # after
    #     5106   26.593    0.005   26.593    0.005 hamming.py:26(bit_count64)
    #     5106   26.308    0.005   26.308    0.005 hamming.py:28(bit_count64)
    #
    # reduce copies by subtract in place

    arr -= ((arr >> 1) & s55)
    arr = (arr & s33) + ((arr >> 2) & s33)

    arr += (arr >> 4)
    arr &= s0F
    arr *= s01
    arr >>= (8 * (num_bytes_64 - 1))

    return arr


# arr = np.array([100, 100, 0])
# orig_arr = arr.copy()
# print(bit_count64(arr))
# assert np.all(arr == orig_arr)


# Random array of 64bit integers
def bitcount_benchmark():
    arr_size = (10000, 100)
    max_val = np.iinfo(np.uint64).max
    min_val = np.iinfo(np.uint64).min
    arr = np.random.randint(min_val, max_val, arr_size, dtype=np.uint64)

    # Benchmark np.bitwise_count
    print("Just bitcount")
    print("Popcount: ", timeit.timeit(lambda: np.bitwise_count(arr), number=1000))

    # Benchmark
    print("Naive:    ", timeit.timeit(lambda: bit_count64(arr), number=1000))

    assert np.all(bit_count64(arr) == np.bitwise_count(arr))


xor1_time = 0
count_function_time = 0
xor2_time = 0
sum_time = 0


def hamming_sim_copy(hashes, query, count_function=bit_count64):
    """Compute 1-hamming similarity score hashes and query hash.

    Returns
    -------
    array with each hash's similarity to query, for each row
     - 0 if identical to query, query.shape[1] * 64 if completely different

    normalize to 0-1 by dividing by query.shape[1] / 64

    Parameters
    ----------
    hashes are bit array of LSH hashes
    query is a bit array of LSH hashes

    """
    # hashes_new = hashes ^ query   # COPY!
    # Num where the do NOT share (to save inverting hashes, easier to 1- scores later)
    return np.sum(count_function(hashes ^ query), axis=1)

    # To make this faster, make a ufunc in C that counts dissimilar bits between query and hash`

    # To normalize
    # np.divide(xor_sim, hashes.shape[1] * 64)


def hamming_sim_nocopy(hashes, query, count_function=bit_count64):
    """Compute 1-hamming similarity score hashes and query hash.

    0 is identical, query.shape[1] * 64 if completely different

    Parameters
    ----------
    hashes are bit array of LSH hashes
    query is a bit array of LSH hashes

    """
    # In-place xor
    hashes ^= query   # SAVE A COPY!
    # Num where the do NOT share (to save inverting hashes, easier to 1- scores later)
    num_unshared_bits = count_function(hashes)

    # Undo xor
    hashes ^= query
    xor_sim = np.sum(num_unshared_bits,
                     axis=1)

    # Optionally normalize
    # xor_sim = np.divide(xor_sim, hashes.shape[1] * 64)
    return xor_sim


def hamming_ufunc_unshared_bits(hashes, query):
    xor_sim = num_unshared_bits(hashes, query)
    return np.sum(xor_sim,
                  axis=1)


def hamming_benchmark(rows=100000000, num_hashes=10, num_executions=10):
    """Benchmark hamming similarity."""
    arr_size = (rows, num_hashes)
    max_val = np.iinfo(np.uint64).max
    min_val = np.iinfo(np.uint64).min
    hashes = np.random.randint(min_val, max_val, arr_size, dtype=np.uint64)
    query = hashes[-1].copy()  # np.random.randint(0, max_val, (1, 100), dtype=np.int64)
    print(query.shape)

    # Make almost similar
    hashes[-2] = hashes[-1]
    hashes[-2][0] = 0
    print(f"Hamming: {rows} rows, {num_hashes} hashes per row")

    # Sanity check
    sims = hamming_ufunc_unshared_bits(hashes, query)
    assert sims[-1] == 0
    hashes_before = hashes.copy()
    runtime = timeit.timeit(lambda: hamming_ufunc_unshared_bits(hashes, query), number=num_executions)
    print("Custom Ufunc - just bc  : ", runtime / num_executions)
    assert np.all(hashes == hashes_before), "Hamming w/ np.bitwise_count was destructive"

    # Sanity check
    sims = hamming_c_ufunc(hashes, query)
    assert sims[-1] == 0
    hashes_before = hashes.copy()
    runtime = timeit.timeit(lambda: hamming_c_ufunc(hashes, query), number=num_executions)
    print("Custom Ufunc - full ham : ", runtime / num_executions)
    assert np.all(hashes == hashes_before), "Hamming w/ np.bitwise_count was destructive"
    return

# -----------------------------------------------
# Stats from previous run
#
# Hamming: 100000000 rows, 10 hashes per row
# Popcount (cpy)  :  3.122813302089926
# Popcount (nocpy):  3.118787500830367
# Naive:     9.449723059160169
#
# --- custom ufuncs
# Custom Ufunc - just bc  :  2.584149041690398
# Custom Ufunc - full ham :  0.3731657250085846


if __name__ == '__main__':
    # bitcount_benchmark()
    hamming_benchmark()
