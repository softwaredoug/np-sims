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

#define TOP_N_QUEUE(N) \
struct Top ## N ## Queue { \
    uint64_t size; \
    uint64_t out_queue_end; \
    uint64_t worst_in_queue; \
    uint64_t worst_in_queue_index; \
    uint64_t* best_rows; \
    uint64_t top_n_sim_scores[N]; \
}; \
const struct Top ## N ## Queue defaults_## N = { \
    .size = N, \
    .out_queue_end = 0, \
    .best_rows = NULL, \
    .worst_in_queue = UINT64_MAX, \
    .worst_in_queue_index = 0 \
}; \
\
struct Top ## N ## Queue create_ ## N ## _queue(uint64_t* best_rows) { \
    struct Top ## N ## Queue queue = defaults_## N; \
    queue.best_rows = best_rows; \
    /* Initialize to UINT64_MAX*/ \
    for (int i = 0; i < N; i++) { \
        queue.top_n_sim_scores[i] = UINT64_MAX; \
        queue.best_rows[i] = UINT64_MAX; \
    } \
    return queue; \
} \
\


TOP_N_QUEUE(10)
TOP_N_QUEUE(1000)



/*
 *
 *
 *
 *
 */

void maybe_insert_into_queue_10(struct Top10Queue* queue, uint64_t sim_score,
                                    uint64_t row_index) {
   if (sim_score < queue->worst_in_queue) {
     if (queue->out_queue_end < queue->size) {
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
     for (uint64_t output_idx = 0; output_idx < queue->size; output_idx++) {
       if (queue->top_n_sim_scores[output_idx] > queue->worst_in_queue) {
         queue->worst_in_queue = queue->top_n_sim_scores[output_idx];
         queue->worst_in_queue_index = output_idx;
       }
     }
   }
}


void maybe_insert_into_queue_1000(struct Top1000Queue* queue, uint64_t sim_score,
                                    uint64_t row_index) {
   if (sim_score < queue->worst_in_queue) {
     if (queue->out_queue_end < queue->size) {
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
     for (uint64_t output_idx = 0; output_idx < queue->size; output_idx++) {
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


/* loop unrolled versions of hamming sim computation */
#define HAMMING_TOP_N_HASH(N, QUEUE_LEN, BODY) \
void hamming_top_ ## QUEUE_LEN ## _hash_##N(uint64_t* hashes, uint64_t* query, \
                                      uint64_t num_hashes, uint64_t* best_rows) { \
    struct Top ## QUEUE_LEN ## Queue queue = create_ ## QUEUE_LEN ## _queue(best_rows); \
    uint64_t sum = 0; \
    for (uint64_t i = 0; i < num_hashes; i++) { \
      BODY; \
      maybe_insert_into_queue_## QUEUE_LEN (&queue, sum, i); \
    } \
}

/* 64 bits */
HAMMING_TOP_N_HASH(1, 10,
    sum = popcount((*hashes++) ^ (*query));
)

/* 128 bits */
HAMMING_TOP_N_HASH(2, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
)

/* 192 bits */
HAMMING_TOP_N_HASH(3, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
)

/* 256 bits */
HAMMING_TOP_N_HASH(4, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
)

/* 320 bits */
HAMMING_TOP_N_HASH(5, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
)

/* 384 bits */
HAMMING_TOP_N_HASH(6, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
)

/* 448 bits */
HAMMING_TOP_N_HASH(7, 10,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
)

/* 512 bits */
HAMMING_TOP_N_HASH(8, 10,
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
HAMMING_TOP_N_HASH(9, 10,
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
HAMMING_TOP_N_HASH(10, 10,
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

/* 64 bits */
HAMMING_TOP_N_HASH(1, 1000,
    sum = popcount((*hashes++) ^ (*query));
)

/* 128 bits */
HAMMING_TOP_N_HASH(2, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
)

/* 192 bits */
HAMMING_TOP_N_HASH(3, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
)

/* 256 bits */
HAMMING_TOP_N_HASH(4, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
)

/* 320 bits */
HAMMING_TOP_N_HASH(5, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
)

/* 384 bits */
HAMMING_TOP_N_HASH(6, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
)

/* 448 bits */
HAMMING_TOP_N_HASH(7, 1000,
    sum = popcount((*hashes++) ^ (*query));
    sum += popcount((*hashes++) ^ query[1]);
    sum += popcount((*hashes++) ^ query[2]);
    sum += popcount((*hashes++) ^ query[3]);
    sum += popcount((*hashes++) ^ query[4]);
    sum += popcount((*hashes++) ^ query[5]);
    sum += popcount((*hashes++) ^ query[6]);
)

/* 512 bits */
HAMMING_TOP_N_HASH(8, 1000,
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
HAMMING_TOP_N_HASH(9, 1000,
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
HAMMING_TOP_N_HASH(10, 1000,
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

void hamming_top_10_default(uint64_t* hashes, uint64_t* query, uint64_t* query_start,
                           uint64_t num_hashes, uint64_t query_len, uint64_t* best_rows) {
    struct Top10Queue queue = create_10_queue(best_rows);
    uint64_t sum = 0;

    for (uint64_t i = 0; i < num_hashes; i++) {
        sum = 0;
        query = query_start;
        #pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < query_len; j++) {
            sum += popcount((*hashes++) ^ (*query++));
        }
        maybe_insert_into_queue_10(&queue, sum, i);
    }
}

void hamming_top_1000_default(uint64_t* hashes, uint64_t* query, uint64_t* query_start,
                              uint64_t num_hashes, uint64_t query_len, uint64_t* best_rows) {
    struct Top1000Queue queue = create_1000_queue(best_rows);
    uint64_t sum = 0;

    for (uint64_t i = 0; i < num_hashes; i++) {
        sum = 0;
        query = query_start;
        #pragma clang loop vectorize(enable)
        for (uint64_t j = 0; j < query_len; j++) {
            sum += popcount((*hashes++) ^ (*query++));
        }
        maybe_insert_into_queue_1000(&queue, sum, i);
    }
}


static void hamming_top_10(char **args, const npy_intp *dimensions,
                           const npy_intp *steps, void *data)
{

    /* npy_intp n = dimensions[0]; <<- not sure what this is */
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp query_len = dimensions[2];  /* appears to be size of second dimension */
    uint64_t *hashes = __builtin_assume_aligned((uint64_t*)args[0], 16);
    uint64_t *query =  __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *query_start = __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *best_rows = __builtin_assume_aligned((uint64_t*)args[2], 16);

    PyThreadState *_save = NULL;
    if (PyGILState_Check()) { _save = PyEval_SaveThread(); }

    switch (query_len) {
        case 1:
            hamming_top_10_hash_1(hashes, query, num_hashes, best_rows);
            break;
        case 2:
            hamming_top_10_hash_2(hashes, query, num_hashes, best_rows);
            break;
        case 3:
            hamming_top_10_hash_3(hashes, query, num_hashes, best_rows);
            break;
        case 4:
            hamming_top_10_hash_4(hashes, query, num_hashes, best_rows);
            break;
        case 5:
            hamming_top_10_hash_5(hashes, query, num_hashes, best_rows);
            break;
        case 6:
            hamming_top_10_hash_6(hashes, query, num_hashes, best_rows);
            break;
        case 7:
            hamming_top_10_hash_7(hashes, query, num_hashes, best_rows);
            break;
        case 8:
            hamming_top_10_hash_8(hashes, query, num_hashes, best_rows);
            break;
        case 9:
            hamming_top_10_hash_9(hashes, query, num_hashes, best_rows);
            break;
        case 10:  /* start to get diminishing returns -- rolled: 101 unrolled: 111 */
            hamming_top_10_hash_10(hashes, query, num_hashes, best_rows);
            break;
        default:
            hamming_top_10_default(hashes, query, query_start, num_hashes, query_len, best_rows);
            break;
    }

    if (_save != NULL) { PyEval_RestoreThread(_save); }
}


static void hamming_top_cand(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{

    /* npy_intp n = dimensions[0]; <<- not sure what this is */
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp query_len = dimensions[2];  /* appears to be size of second dimension */
    uint64_t *hashes = __builtin_assume_aligned((uint64_t*)args[0], 16);
    uint64_t *query =  __builtin_assume_aligned((uint64_t*)args[1], 16);
    uint64_t *best_rows = __builtin_assume_aligned((uint64_t*)args[2], 16);
    uint64_t *query_start = __builtin_assume_aligned((uint64_t*)args[1], 16);

    PyThreadState *_save = NULL;
    if (PyGILState_Check()) { _save = PyEval_SaveThread(); }

    switch (query_len) {
        case 1:
            hamming_top_1000_hash_1(hashes, query, num_hashes, best_rows);
            break;
        case 2:
            hamming_top_1000_hash_2(hashes, query, num_hashes, best_rows);
            break;
        case 3:
            hamming_top_1000_hash_3(hashes, query, num_hashes, best_rows);
            break;
        case 4:
            hamming_top_1000_hash_4(hashes, query, num_hashes, best_rows);
            break;
        case 5:
            hamming_top_1000_hash_5(hashes, query, num_hashes, best_rows);
            break;
        case 6:
            hamming_top_1000_hash_6(hashes, query, num_hashes, best_rows);
            break;
        case 7:
            hamming_top_1000_hash_7(hashes, query, num_hashes, best_rows);
            break;
        case 8:
            hamming_top_1000_hash_8(hashes, query, num_hashes, best_rows);
            break;
        case 9:
            hamming_top_1000_hash_9(hashes, query, num_hashes, best_rows);
            break;
        case 10:  /* start to get diminishing returns -- rolled: 101 unrolled: 111 */
            hamming_top_1000_hash_10(hashes, query, num_hashes, best_rows);
            break;
        default:
            hamming_top_1000_default(hashes, query, query_start, num_hashes, query_len, best_rows);
            break;
    }

    if (_save != NULL) { PyEval_RestoreThread(_save); }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&unshared_bits};
PyUFuncGenericFunction hamming_funcs[1] = {&hamming};
PyUFuncGenericFunction hamming_10_funcs[1] = {&hamming_top_10};
PyUFuncGenericFunction hamming_cand_funcs[1] = {&hamming_top_cand};

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
    PyObject *m, *num_unshared, *hamming_ufunc, *hamming_10_ufunc, *hamming_cand_ufunc, *d;

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

    hamming_10_ufunc = PyUFunc_FromFuncAndDataAndSignature(hamming_10_funcs, NULL, hamming_types, 1, 2, 1,
                                                           PyUFunc_None, "hamming_top_10",
                                                           "Gets the top 10 hashes by hamming similarity", 0,
                                                           "(m,n),(n)->(10)");
    if (hamming_10_ufunc == NULL) {
        printf("hamming_n_ufunc is NULL!!\n");
    }

    hamming_cand_ufunc = PyUFunc_FromFuncAndDataAndSignature(hamming_cand_funcs, NULL, hamming_types, 1, 2, 1,
                                                            PyUFunc_None, "hamming_top_cand",
                                                            "Gets the top 1000 hashes by hamming similarity", 0,
                                                            "(m,n),(n)->(1000)");
    if (hamming_cand_ufunc == NULL) {
        printf("hamming_n_ufunc is NULL!!\n");
    }
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "num_unshared_bits", num_unshared);
    PyDict_SetItemString(d, "hamming", hamming_ufunc);
    PyDict_SetItemString(d, "hamming_top_10", hamming_10_ufunc);
    PyDict_SetItemString(d, "hamming_top_cand", hamming_cand_ufunc);
    Py_DECREF(num_unshared);
    Py_DECREF(hamming_ufunc);

    return m;
}
