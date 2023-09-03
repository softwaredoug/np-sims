#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "numpy/npy_math.h"
#include <math.h>
#include <stdio.h>

#if defined(__arm__) || defined(__aarch64__)
#include <arm_neon.h>
#endif

#ifdef _MSC_VER
#  include <intrin.h>   // for __popcnt
#endif

/*
 * multi_arg_logit.c
 * This is the C code for creating your own
 * NumPy ufunc for a multiple argument, multiple
 * return value ufunc. The places where the
 * ufunc computation is carried out are marked
 * with comments.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org.
 */

#ifndef NPY_ALLOW_THREADS
#error NPY_ALLOW_THREADS must be defined
#endif

static PyMethodDef LogitMethods[] = {
    {NULL, NULL, 0, NULL}
};

inline uint8_t popcount(uint64_t val) {
#if (defined(__clang__) || defined(__GNUC__))
    return __builtin_popcountll(val);
#elif defined(_MSC_VER)
    return (uint8_t) __popcnt64(val);
#endif
}

/* The loop definition must precede the PyMODINIT_FUNC. */

static void unshared_bits(char **args, const npy_intp *dimensions,
                          const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];

    uint64_t xord = 0;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        xord = (*(uint64_t *)in1) ^ (*(uint64_t *)in2);
        /* perform popcount */

        *((uint8_t *)out1) += popcount(xord);
        // *((double *)out2) = log(tmp / (1 - tmp));
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

#ifndef UINT64_MAX
#define UINT64_MAX 0xffffffffffffffff
#endif


static void hamming(char **args, const npy_intp *dimensions,
                    const npy_intp *steps, void *data)
{
    npy_intp i, j;
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp n2 = dimensions[2];  /* appears to be size of second dimension */
    char *in1 = args[0], *in2 = args[1], *in2_start = args[1];
    char *out1 = args[2];

    uint64_t xord = 0;

    /* is dimension now what  we're accumulating over? */
    uint64_t sum = 0;
    for (i = 0; i < num_hashes; i++) {
        sum = 0;
        in2 = in2_start;
        for (j = 0; j < n2; j++) {
          xord = (*(uint64_t *)in1) ^ (*(uint64_t *)in2);
          /* perform popcount */

          sum += popcount(xord);
          in2 += sizeof(uint64_t);
          in1 += sizeof(uint64_t);
          /* printf("value1: %llu, value2: %llu, xord: %llu, sum: %llu\n", value1, value2, xord, sum); */
          /* END main ufunc computation */
        }
        (*((uint64_t *)out1)) = sum;

        out1 += sizeof(uint64_t);
    }
}

const uint64_t N = 10;


struct TopNQueue {
    uint64_t out_queue_end;
    uint64_t top_n_sim_scores[10];
    uint64_t worst_in_queue;
    uint64_t worst_in_queue_index;

    uint64_t* best_rows;
};

const struct TopNQueue defaults = {
    .out_queue_end = 0,
    .top_n_sim_scores = {UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX,
                         UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX},
    .best_rows = NULL,
    .worst_in_queue = UINT64_MAX,
    .worst_in_queue_index = 0
};


struct TopNQueue create_queue(uint64_t* best_rows) {
    struct TopNQueue queue = defaults;
    queue.best_rows = best_rows;
    /* Initialize to UINT64_MAX*/
    for (uint64_t i = 0; i < N; i++) {
        best_rows[i] = UINT64_MAX;
    }
    return queue;
}

/*
 *
 *
 *
 *
 */

inline void maybe_insert_into_queue(struct TopNQueue* queue, uint64_t sim_score,
                                    uint64_t row_index, uint64_t queue_len) {
   if (sim_score < queue->worst_in_queue) {
     if (queue->out_queue_end < queue_len) {
       queue->best_rows[queue->out_queue_end] = row_index;
       queue->top_n_sim_scores[queue->out_queue_end] = sim_score;
       queue->out_queue_end++;
     }
     else {
       queue->best_rows[queue->worst_in_queue_index] = row_index;
       queue->top_n_sim_scores[queue->worst_in_queue_index] = sim_score;
     }
     /* find new worst_in_queue */
     queue->worst_in_queue = 0;
     for (uint64_t output_idx = 0; output_idx < queue_len; output_idx++) {
       if (queue->top_n_sim_scores[output_idx] > queue->worst_in_queue) {
         queue->worst_in_queue = queue->top_n_sim_scores[output_idx];
         queue->worst_in_queue_index = output_idx;
       }
     }
   }
}


void hamming_top_n__hash_1_stride(uint64_t* hashes, uint64_t* query, \
                              uint64_t num_hashes, uint64_t* best_rows,
                              uint64_t query_len) { \
    struct TopNQueue queue = create_queue(best_rows);
    uint64_t sum = 0;
    NPY_BEGIN_ALLOW_THREADS \
    for (uint64_t i = 0; i < num_hashes; i++) {
      sum = popcount((*hashes++) ^ (*query));
      maybe_insert_into_queue(&queue, sum, i, 10);
    }
    NPY_END_ALLOW_THREADS
}

/*#define EARLY_EXIT*/

#ifndef __builtin_assume_aligned
#define __builtin_assume_aligned(x, y) (x)
#endif

/* loop unrolled versions of hamming sim computation */



#define HAMMING_TOP_N_HASH(N, BODY) \
void hamming_top_n_hash_##N(uint64_t* hashes, uint64_t* query, \
                          uint64_t num_hashes, uint64_t* best_rows) { \
    struct TopNQueue queue = create_queue(best_rows); \
    uint64_t sum = 0; \
    NPY_BEGIN_ALLOW_THREADS \
    for (uint64_t i = 0; i < num_hashes; i++) { \
      BODY; \
      maybe_insert_into_queue(&queue, sum, i, 10); \
    } \
    NPY_END_ALLOW_THREADS \
}

/* 64 bits */
HAMMING_TOP_N_HASH(1,
    sum = popcount((*hashes++) ^ (*query));
)

/* 128 bits */
HAMMING_TOP_N_HASH(2,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
)

/* 192 bits */
HAMMING_TOP_N_HASH(3,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
)

/* 256 bits */
HAMMING_TOP_N_HASH(4,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
)

/* 320 bits */
HAMMING_TOP_N_HASH(5,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
)

/* 384 bits */
HAMMING_TOP_N_HASH(6,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
)

/* 448 bits */
HAMMING_TOP_N_HASH(7,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
)

/* 512 bits */
HAMMING_TOP_N_HASH(8,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
    sum += popcount((*hashes++) ^ query[7]);
)

/* 576 bits */
HAMMING_TOP_N_HASH(9,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
    sum += popcount((*hashes++) ^ query[7]);
    sum += popcount((*hashes++) ^ query[8]);
)

/* 640 bits */
HAMMING_TOP_N_HASH(10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
    sum += popcount((*hashes++) ^ query[7]);
    sum += popcount((*hashes++) ^ query[8]);
    sum += popcount((*hashes++) ^ query[9]);
)


/* Default not unrolled */

void hamming_top_n_default(uint64_t* hashes, uint64_t* query, uint64_t* query_start,
                           uint64_t num_hashes, uint64_t query_len, uint64_t* best_rows) {
    struct TopNQueue queue = create_queue(best_rows);
    uint64_t sum = 0;
    NPY_BEGIN_ALLOW_THREADS

    for (uint64_t i = 0; i < num_hashes; i++) {
        sum = 0;
        query = query_start;
        #pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < query_len; j++) {
            sum += popcount((*hashes++) ^ (*query++));
        }
        maybe_insert_into_queue(&queue, sum, i,10);
    }
    NPY_END_ALLOW_THREADS
}


/* Unrolled with 40 64 bit hashes, 20 128 bit hashes
 *
 * Performance is only 32.5 QPS, so not worth it
 *
 * */
void hamming_top_n_default_simd_40(uint64_t* hashes, uint64_t* query, uint64_t* query_start,
                                   uint64_t num_hashes, uint64_t query_len, uint64_t* best_rows) {
  struct TopNQueue queue = create_queue(best_rows);

  assert(query_len == 40);

  /* Load query into registers */
  uint64x2_t q0 = vld1q_u64(query_start);
  uint64x2_t q1 = vld1q_u64(query_start + 2);
  uint64x2_t q2 = vld1q_u64(query_start + 4);
  uint64x2_t q3 = vld1q_u64(query_start + 6);
  uint64x2_t q4 = vld1q_u64(query_start + 8);
  uint64x2_t q5 = vld1q_u64(query_start + 10);
  uint64x2_t q6 = vld1q_u64(query_start + 12);
  uint64x2_t q7 = vld1q_u64(query_start + 14);
  uint64x2_t q8 = vld1q_u64(query_start + 16);
  uint64x2_t q9 = vld1q_u64(query_start + 18);
  uint64x2_t q10 = vld1q_u64(query_start + 20);
  uint64x2_t q11 = vld1q_u64(query_start + 22);
  uint64x2_t q12 = vld1q_u64(query_start + 24);
  uint64x2_t q13 = vld1q_u64(query_start + 26);
  uint64x2_t q14 = vld1q_u64(query_start + 28);
  uint64x2_t q15 = vld1q_u64(query_start + 30);
  uint64x2_t q16 = vld1q_u64(query_start + 32);
  uint64x2_t q17 = vld1q_u64(query_start + 34);
  uint64x2_t q18 = vld1q_u64(query_start + 36);
  uint64x2_t q19 = vld1q_u64(query_start + 38);

  /* accumulate register */

  for (uint64_t i = 0; i < num_hashes; i++) {
    uint64x2_t sum0 = vdupq_n_u64(0);

    /* current hashes into registers */
    uint64x2_t h0 = vld1q_u64(hashes);
    uint64x2_t h1 = vld1q_u64(hashes += 2);
    uint64x2_t h2 = vld1q_u64(hashes += 2);
    uint64x2_t h3 = vld1q_u64(hashes += 2);
    uint64x2_t h4 = vld1q_u64(hashes += 2);
    uint64x2_t h5 = vld1q_u64(hashes += 2);
    uint64x2_t h6 = vld1q_u64(hashes += 2);
    uint64x2_t h7 = vld1q_u64(hashes += 2);
    uint64x2_t h8 = vld1q_u64(hashes += 2);
    uint64x2_t h9 = vld1q_u64(hashes += 2);
    uint64x2_t h10 = vld1q_u64(hashes += 2);
    uint64x2_t h11 = vld1q_u64(hashes += 2);
    uint64x2_t h12 = vld1q_u64(hashes += 2);
    uint64x2_t h13 = vld1q_u64(hashes += 2);
    uint64x2_t h14 = vld1q_u64(hashes += 2);
    uint64x2_t h15 = vld1q_u64(hashes += 2);
    uint64x2_t h16 = vld1q_u64(hashes += 2);
    uint64x2_t h17 = vld1q_u64(hashes += 2);
    uint64x2_t h18 = vld1q_u64(hashes += 2);
    uint64x2_t h19 = vld1q_u64(hashes += 2);
    hashes += 2;

    /* XOR, popcount, then sum to sum registers */
    sum0 = vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h0, q0))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h1, q1)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h2, q2)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h3, q3)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h4, q4)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h5, q5)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h6, q6)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h7, q7)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h8, q8)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h9, q9)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h10, q10)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h11, q11)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h12, q12)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h13, q13)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h14, q14)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h15, q15)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h16, q16)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h17, q17)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h18, q18)))));
    sum0 = vaddq_u64(sum0, vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(veorq_u64(h19, q19)))));

    maybe_insert_into_queue(&queue,
                            vgetq_lane_u64(sum0, 0) + vgetq_lane_u64(sum0, 1),
                            i, 10);
  }

}


/*
 * Fastest simd thusfar, though it seems to have a bug as it doesn't get
 * same recall as the non-simd version.
 */
void hamming_top_n_default_simd(uint64_t* hashes, uint64_t* query, uint64_t* query_start,
                                uint64_t num_hashes, uint64_t query_len, uint64_t* best_rows) {
    struct TopNQueue queue = create_queue(best_rows);
    uint64x2_t sum = {0, 0};

    /* Fail if hashes are odd */
    if (query_len % 2 != 0) {
        return;
    }

    NPY_BEGIN_ALLOW_THREADS

    for (uint64_t i = 0; i < num_hashes; i++) {
        sum = vdupq_n_u64(0);
        query = query_start;
        for (uint64_t j = 0; j < query_len; j+=2) {
            /* Load into Neon registers */
            uint64x2_t q = vld1q_u64(query);
            uint64x2_t h = vld1q_u64(hashes);
            /* XOR */
            uint64x2_t x = veorq_u64(q, h);
            /* Popcount */
            uint8x16_t p = vcntq_u8(vreinterpretq_u8_u64(x));
            /* Get sum of each pop count byte */
            uint64x2_t s = vpaddlq_u8(p);

            /* Add to sum */
            sum = vaddq_u64(sum, s);

            /* Get the lower uint64 */
            /*uint64x1_t s1 = vget_low_u64(s);*/

            query += 2;
            hashes += 2;
        }
        /* Only add ot output if its better than nth_so_far. We don't care about sorting*/
        maybe_insert_into_queue(&queue,
                                vgetq_lane_u64(sum, 0) + vgetq_lane_u64(sum, 1),
                                i, 10);
    }
    NPY_END_ALLOW_THREADS
}

static void hamming_top_n(char **args, const npy_intp *dimensions,
                          const npy_intp *steps, void *data)
{

    /* npy_intp n = dimensions[0]; <<- not sure what this is */
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp query_len = dimensions[2];  /* appears to be size of second dimension */
    uint64_t *hashes = __builtin_assume_aligned((uint64_t*)args[0], 16);
    uint64_t *query =  __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *query_start = __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *best_rows = __builtin_assume_aligned((uint64_t*)args[2], 16);

    switch (query_len) {
        case 1:
            hamming_top_n_hash_1(hashes, query, num_hashes, best_rows);
            return;
        case 2:
            hamming_top_n_hash_2(hashes, query, num_hashes, best_rows);
            return;
        case 3:
            hamming_top_n_hash_3(hashes, query, num_hashes, best_rows);
            return;
        case 4:
            hamming_top_n_hash_4(hashes, query, num_hashes, best_rows);
            return;
        case 5:
            hamming_top_n_hash_5(hashes, query, num_hashes, best_rows);
            return;
        case 6:
            hamming_top_n_hash_6(hashes, query, num_hashes, best_rows);
            return;
        case 7:
            hamming_top_n_hash_7(hashes, query, num_hashes, best_rows);
            return;
        case 8:
            hamming_top_n_hash_8(hashes, query, num_hashes, best_rows);
            return;
        case 9:
            hamming_top_n_hash_9(hashes, query, num_hashes, best_rows);
            return;
        case 10:  /* start to get diminishing returns -- rolled: 101 unrolled: 111 */
            hamming_top_n_hash_10(hashes, query, num_hashes, best_rows);
            return;
        case 40:  /* experimental simd */
            hamming_top_n_default_simd(hashes, query, query_start, num_hashes, query_len, best_rows);
            return;
        default:
            hamming_top_n_default(hashes, query, query_start, num_hashes, query_len, best_rows);
            if (query_len % 2 == 0) {
                hamming_top_n_default_simd(hashes, query, query_start, num_hashes, query_len, best_rows);
                return;
            }
            else {
                hamming_top_n_default(hashes, query, query_start, num_hashes, query_len, best_rows);
                return;
            }
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&unshared_bits};
PyUFuncGenericFunction hamming_funcs[1] = {&hamming};
PyUFuncGenericFunction hamming_n_funcs[1] = {&hamming_top_n};

/* These are the input and return dtypes of logit.*/

static char types[3] = {NPY_UINT64, NPY_UINT64,
                        NPY_UINT8};

static char hamming_types[3] = {NPY_UINT64, NPY_UINT64,
                                NPY_UINT64};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufuncs",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ufuncs(void)
{
    PyObject *m, *num_unshared, *hamming_ufunc, *hamming_n_ufunc, *d;

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    num_unshared = PyUFunc_FromFuncAndData(funcs, NULL, types, 1, 2, 1,
                                           PyUFunc_None, "num_unshared_bits",
                                           "num_unshared_bits_docstr", 0);

    if (num_unshared == NULL) {
        printf("num_unshared is NULL!!\n");
    }

    hamming_ufunc = PyUFunc_FromFuncAndDataAndSignature(hamming_funcs, NULL, hamming_types, 1, 2, 1,
                                                  PyUFunc_None, "hamming",
                                                  "hamming_docstr", 0,
                                                  "(m,n),(n)->(m)");
    if (hamming_ufunc == NULL) {
        printf("hamming_ufunc is NULL!!\n");
    }

    hamming_n_ufunc = PyUFunc_FromFuncAndDataAndSignature(hamming_n_funcs, NULL, hamming_types, 1, 2, 1,
                                                          PyUFunc_None, "hamming_top_n",
                                                          "Gets the top 10 hashes by hamming similarity", 0,
                                                          "(m,n),(n)->(10)");
    if (hamming_n_ufunc == NULL) {
        printf("hamming_n_ufunc is NULL!!\n");
    }
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "num_unshared_bits", num_unshared);
    PyDict_SetItemString(d, "hamming", hamming_ufunc);
    PyDict_SetItemString(d, "hamming_top_n", hamming_n_ufunc);
    Py_DECREF(num_unshared);
    Py_DECREF(hamming_ufunc);

    return m;
}
