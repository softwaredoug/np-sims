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


static void hamming(char **args, const npy_intp *dimensions,
                    const npy_intp *steps, void *data)
{
    npy_intp i, j;
    npy_intp n = dimensions[0];
    npy_intp n1 = dimensions[1];  /* appears to be size of first dimension */
    npy_intp n2 = dimensions[2];  /* appears to be size of second dimension */
    char *in1 = args[0], *in2 = args[1], *in1_start = args[0], *in2_start = args[1];
    char *out1 = args[2];

    uint64_t xord = 0;

    /* is dimension now what  we're accumulating over? */
    uint64_t sum = 0;
    for (i = 0; i < n1; i++) {
        sum = 0;
        in2 = in2_start;
        for (j = 0; j < n2; j++) {
          /* uint64_t value1 = (*(uint64_t *)in1);
          uint64_t value2 = (*(uint64_t *)in2); */
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

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&unshared_bits};
PyUFuncGenericFunction hamming_funcs[1] = {&hamming};

/* These are the input and return dtypes of logit.*/

static char types[3] = {NPY_UINT64, NPY_UINT64,
                        NPY_UINT8};

static char hamming_types[3] = {NPY_UINT64, NPY_UINT64,
                                NPY_UINT64};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    LogitMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *num_unshared, *hamming_ufunc, *d;

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
    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "num_unshared_bits", num_unshared);
    PyDict_SetItemString(d, "hamming", hamming_ufunc);
    Py_DECREF(num_unshared);
    Py_DECREF(hamming_ufunc);

    return m;
}
