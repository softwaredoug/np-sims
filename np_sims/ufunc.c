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

static uint8_t popcount(uint64_t val) {
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
    npy_intp n = dimensions[0];
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp n2 = dimensions[2];  /* appears to be size of second dimension */
    char *in1 = args[0], *in2 = args[1], *in1_start = args[0], *in2_start = args[1];
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

    uint64_t* best_rows
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
    for (int i = 0; i < N; i++) {
        best_rows[i] = UINT64_MAX;
    }
    return queue;
}

void maybe_insert_into_queue(struct TopNQueue* queue, uint64_t sim_score, uint64_t row_index) {
   if (sim_score < queue->worst_in_queue) {
     if (queue->out_queue_end < N) {
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
     for (int output_idx = 0; output_idx < 10; output_idx++) {
       if (queue->top_n_sim_scores[output_idx] > queue->worst_in_queue) {
         queue->worst_in_queue = queue->top_n_sim_scores[output_idx];
         queue->worst_in_queue_index = output_idx;
       }
     }
   }
}

static void hamming_top_n(char **args, const npy_intp *dimensions,
                          const npy_intp *steps, void *data)
{
    npy_intp i, j;
    /* npy_intp n = dimensions[0]; <<- not sure what this is */
    npy_intp num_hashes = dimensions[1];  /* appears to be size of first dimension */
    npy_intp hash_len = dimensions[2];  /* appears to be size of second dimension */
    char *in1 = args[0], *in2 = args[1], *in2_start = args[1], in1_next = args[0];


    struct TopNQueue queue = create_queue((uint64_t*)args[2]);

    uint64_t xord = 0;

    /* is dimension now what  we're accumulating over? */
    uint64_t sum = 0;
    for (i = 0; i < num_hashes; i++) {
        sum = 0;
        in2 = in2_start;
        for (j = 0; j < hash_len; j++) {
          xord = (*(uint64_t *)in1) ^ (*(uint64_t *)in2);
          /* perform popcount */

          sum += popcount(xord);

          /* OPTIMIZATION */
          /* break if we've already exceeded the worst in the queue */
          if (sum >= queue.worst_in_queue) {
            /* Fast forward in1 to the next row
            printf(">> (%ld) Maybe fast forwarding j: %lu hash_len: %lu diff: %lu\n", i, j, hash_len, (hash_len - j));
            printf(">>> Worst in queue: %lu\n", queue.worst_in_queue);
            printf(">>>            sum: %lu\n", sum); */
            in1 += (hash_len - j) * sizeof(uint64_t);
            break;
          }

          in2 += sizeof(uint64_t);
          in1 += sizeof(uint64_t);
        }

        /* Only add ot output if its better than nth_so_far. We don't care about sorting*/
        maybe_insert_into_queue(&queue, sum, i);
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
