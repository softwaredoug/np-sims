#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/halffloat.h"
#include "numpy/npy_math.h"
#include <math.h>
#include <stdio.h>

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
const uint64_t NUM_CANDIDATES = 10;


struct TopNQueue {
    uint64_t out_queue_end;
    uint64_t top_n_sim_scores[N];
    uint64_t worst_in_queue;
    uint64_t worst_in_queue_index;

    uint64_t* best_rows;
};

struct CandidateQueue {
    uint64_t out_queue_end;
    uint64_t top_n_sim_scores[NUM_CANDIDATES];
    uint64_t* top_n_hashes;

    uint64_t worst_in_queue;
    uint64_t worst_in_queue_index;

    uint64_t* best_rows;
};

const struct TopNQueue topn_defaults = {
    .out_queue_end = 0,
    .top_n_sim_scores = {UINT64_MAX},
    .best_rows = NULL,
    .worst_in_queue = UINT64_MAX,
    .worst_in_queue_index = 0
};


const struct CandidateQueue cand_defaults = {
    .out_queue_end = 0,
    .top_n_sim_scores = {UINT64_MAX},
    .best_rows = NULL,
    .top_n_hashes = NULL,
    .worst_in_queue = UINT64_MAX,
    .worst_in_queue_index = 0
};

struct TopNQueue create_topn_queue(uint64_t* best_rows) {
    struct TopNQueue queue = topn_defaults;
    queue.best_rows = best_rows;
    return queue;
}

struct CandidateQueue create_candidate_queue(uint64_t* best_rows, uint64_t query_len) {
    struct CandidateQueue queue = cand_defaults;
    queue.best_rows = best_rows;
    /*printf("!! Allocating %llu bytes for top_n_hashes\n", sizeof(uint64_t) * NUM_CANDIDATES * query_len);*/
    queue.top_n_hashes = malloc(sizeof(uint64_t) * NUM_CANDIDATES * query_len);
    return queue;
}

void free_candidate_queue(struct CandidateQueue* queue) {
    free(queue->top_n_hashes);
}


void maybe_insert_as_cand(struct CandidateQueue* queue, uint64_t sim_score,
                          uint64_t* hashes,
                          uint64_t row_index,
                          uint64_t queue_len,
                          uint64_t hash_len) {
   if (sim_score < queue->worst_in_queue) {
     if (queue->out_queue_end < queue_len) {
       queue->best_rows[queue->out_queue_end] = row_index;
       queue->top_n_sim_scores[queue->out_queue_end] = sim_score;
       /*printf("!1 Inserting into queue at index %llu\n", queue->out_queue_end);
       printf("!1 hash_len: %llu\n", hash_len);
       printf("------------------------------------\n");*/

       memcpy(queue->top_n_hashes + (queue->out_queue_end * hash_len),
              hashes,
              sizeof(uint64_t) * hash_len);
       queue->out_queue_end++;
     }
     else {
       queue->best_rows[queue->worst_in_queue_index] = row_index;
       queue->top_n_sim_scores[queue->worst_in_queue_index] = sim_score;
       /*printf("!2 Inserting into queue at index %llu\n", queue->worst_in_queue_index);
       printf("!2 hash_len: %llu\n", hash_len);
       printf("------------------------------------\n");*/
       memcpy(queue->top_n_hashes + (queue->worst_in_queue_index * hash_len),
              hashes,
              sizeof(uint64_t) * hash_len);
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


/*#define EARLY_EXIT*/

#ifndef __builtin_assume_aligned
#define __builtin_assume_aligned(x, y) (x)
#endif

/* Seek candidates by looking at first hamming
 *
 * Give query
 *
 * Q0 Q1 .. QN
 *
 * Given hashes
 *
 * A0 A1 .. AN
 * B0 B1 .. BN
 *
 * We want just A0, B0, etc XOR'd with Q0
 *
 * */
void hamming_candidates(uint64_t* hashes, uint64_t* query,
                        uint64_t num_hashes, struct CandidateQueue* queue,
                        uint64_t hash_len) {
    uint64_t sum = 0;
    for (uint64_t i = 0; i < num_hashes; i++) {
      sum = popcount((*hashes) ^ (*query));
      maybe_insert_as_cand(queue, sum, hashes, i, NUM_CANDIDATES, hash_len);
      hashes += hash_len;
    }
}

/* loop unrolled versions of hamming sim computation */
#define HAMMING_TOP_N_HASH(N, BODY) \
void hamming_top_n_hash_##N(uint64_t* hashes, uint64_t* query, \
                            uint64_t num_hashes, uint64_t* best_rows, \
                            uint64_t output_len) { \
    struct TopNQueue queue = create_topn_queue(best_rows); \
    uint64_t sum = 0; \
    for (uint64_t i = 0; i < num_hashes; i++) { \
      BODY; \
      maybe_insert_into_queue(&queue, sum, i, output_len); \
    } \
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
    struct TopNQueue queue = create_topn_queue(best_rows);
    uint64_t sum = 0;

    for (uint64_t i = 0; i < num_hashes; i++) {
        sum = 0;
        query = query_start;
        #pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < query_len; j++) {
            sum += popcount((*hashes++) ^ (*query++));
        }
        maybe_insert_into_queue(&queue, sum, i, 10);
    }
}


static void hamming_top_n(char **args, const npy_intp *dimensions,
                          const npy_intp *steps, void *data)
{
  return;

    /* npy_intp n = dimensions[0]; <<- not sure what this is */
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp query_len = dimensions[2];  /* appears to be size of second dimension */
    const uint64_t output_len = N;
    uint64_t *hashes = __builtin_assume_aligned((uint64_t*)args[0], 16);
    uint64_t *query =  __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *query_start = __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *best_rows = __builtin_assume_aligned((uint64_t*)args[2], 16);

    uint64_t candidate_rows[NUM_CANDIDATES] = {0};


    /* It seems its best to malloc / free before saving the threadstate? */
    struct CandidateQueue queue; /* = create_candidate_queue(candidate_rows, query_len);*/
    NPY_BEGIN_ALLOW_THREADS
    NPY_END_ALLOW_THREADS
    return;

    /*hamming_candidates(hashes, query, num_hashes, &queue, query_len);*/

    switch (query_len) {
        case 1:
            hamming_top_n_hash_1(queue.top_n_hashes, query, NUM_CANDIDATES,
                                 best_rows, output_len);
            break;
        case 2:
            hamming_top_n_hash_2(queue.top_n_hashes, query, NUM_CANDIDATES,
                                 best_rows, output_len);
            break;
        case 3:
            hamming_top_n_hash_3(queue.top_n_hashes, query, NUM_CANDIDATES,
                                 best_rows, output_len);
            break;
        case 4:
            hamming_top_n_hash_4(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 5:
            hamming_top_n_hash_5(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 6:
            hamming_top_n_hash_6(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 7:
            hamming_top_n_hash_7(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 8:
            hamming_top_n_hash_8(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 9:
            hamming_top_n_hash_9(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        case 10:  /* start to get diminishing returns -- rolled: 101 unrolled: 111 */
            hamming_top_n_hash_10(queue.top_n_hashes, query, NUM_CANDIDATES, best_rows, output_len);
            break;
        default:
            /*hamming_top_n_default(queue.top_n_hashes, query, query_start,
                                    NUM_CANDIDATES, query_len, best_rows);*/
            break;
    }
    /*free_candidate_queue(&queue);*/
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
