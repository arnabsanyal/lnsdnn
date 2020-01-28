#include <Python.h>
#include <stdio.h>
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include <math.h>
#include <omp.h>

#define NUM_THREADS 20

struct module_state {
    PyObject *error;
};

typedef struct {
    int N;
    long granularity;
    double *delP;
    double *delM;
} appr_arithmetic;

static appr_arithmetic obj;

#if PY_MAJOR_VERSION >= 3
#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))
#else
#define GETSTATE(m) (&_state)
static struct module_state _state;
#endif

double* _fast_multiplier(double *_d1, double *_d2, int _m, int _n, int _p) {

    double *temp, *result;
    double diff, lv_max, lv_min;
    int _off1, _off2, _off3, i, j, k, t, index, t_num, t_ovf, chunk, start, end;
    result = (double *) malloc(sizeof(double) * 2 * _m * _n);
    _off1 = _m * _p;
    _off2 = _p * _n;
    _off3 = _m * _n;
    t_ovf = _off3 % NUM_THREADS;
    chunk = _off3 / NUM_THREADS;
    index = 0, i = 0, j = 0, k = 0, t = 0, start = 0, end = 0, lv_max = 0, lv_min = 0, t_num = 0;
    temp = NULL;
    #pragma omp parallel num_threads (NUM_THREADS)\
     private (temp, index, i, j, k, t, diff, lv_max, lv_min, t_num, start, end)\
     shared (_d1, _d2, _m, _n, _p, result, _off1, _off2, _off3, obj, t_ovf, chunk)\
     default (none)
    {
        temp = (double *) malloc(sizeof(double) * 2 * _p);
        t_num = omp_get_thread_num();
        start = t_num * chunk * _p;
        start = ((t_num < t_ovf) && (t_ovf != 0)) ? start + (t_num * _p) : start + (t_ovf * _p);
        end = start + (chunk * _p);
        end = ((t_num < t_ovf) && (t_ovf != 0)) ? end + _p : end;
        for (index = start; index < end; index++) {
            i = index / _off2;
            t = index % _off2;
            j = t / _p;
            k = t % _p;
            temp[k] = (_d1[(i * _p) + k] == _d2[(k * _n) + j]);
            temp[k + _p] = _d1[(i * _p) + k + _off1] + _d2[(k * _n) + j + _off2];
            if (k == 0) {
                result[(i * _n) + j] = temp[0];
                result[(i * _n) + j + _off3] = temp[_p];
            }
            else {
                lv_max = (result[(i * _n) + j + _off3] > temp[k + _p]) ? result[(i * _n) + j + _off3] : temp[k + _p];
                lv_min = (result[(i * _n) + j + _off3] <= temp[k + _p]) ? result[(i * _n) + j + _off3] : temp[k + _p];
                diff = lv_max - lv_min;
                diff = ((npy_isnan(diff)) || (diff > obj.N)) ? obj.N : diff;
                diff *= obj.granularity;
                if (temp[k] == result[(i * _n) + j])
                    result[(i * _n) + j + _off3] = lv_max + obj.delP[lround(diff)];
                else {
                    result[(i * _n) + j] = (result[(i * _n) + j + _off3] > temp[k + _p]) ? result[(i * _n) + j] : temp[k];
                    result[(i * _n) + j + _off3] = lv_max + obj.delM[lround(diff)];
                }
            }
        }
        free(temp);
        #pragma omp barrier
    }
    return result;
}

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

static PyObject *py_fast_multiplier(PyObject *self, PyObject *args)
{
    PyArrayObject *float_list1, *float_list2;
    PyObject *result, *capsule;
    int m, n, p;
    npy_intp dim[3];
    double *d1, *d2, *d3;

    if (!PyArg_ParseTuple(args, "OO", &float_list1, &float_list2))
        return NULL;
    d1 = (double *) float_list1->data;
    d2 = (double *) float_list2->data;
    m = (int)float_list1->dimensions[1];
    n = (int)float_list2->dimensions[2];
    p = (int)float_list1->dimensions[2];
    dim[0] = 2;
    dim[1] = m;
    dim[2] = n;
    d3 = _fast_multiplier(d1, d2, m, n, p);
    result = PyArray_SimpleNewFromData(3, dim, NPY_DOUBLE, (void *)d3);
    if (result == NULL)
        return NULL;
    capsule = PyCapsule_New(d3, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) result, capsule);
    return result;
}

static PyObject *py_init_error_obj(PyObject *self, PyObject *args)
{
    PyArrayObject *float_list1, *float_list2;

    if (!PyArg_ParseTuple(args, "ilOO", &obj.N, &obj.granularity, &float_list1, &float_list2))
        return NULL;
    obj.delP = (double *) float_list1->data;
    obj.delM = (double *) float_list2->data;
    return Py_BuildValue("i", 1);
}

static PyMethodDef symtab[] = {
    {"fast_multiplier", (PyCFunction)py_fast_multiplier, METH_VARARGS|METH_KEYWORDS},
    {"init_error_obj", (PyCFunction)py_init_error_obj,  METH_VARARGS|METH_KEYWORDS},
    {NULL,      NULL}       /* sentinel */
};

static int lst_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int lst_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
        PyModuleDef_HEAD_INIT,
        "native_matr_mult_wrapper",
        NULL,
        sizeof(struct module_state),
        symtab,
        NULL,
        lst_traverse,
        lst_clear,
        NULL
};

PyObject* PyInit_native_matr_mult_wrapper(void)
{
    /* Create the module and add the functions */
    PyObject* module = PyModule_Create(&moduledef);
    import_array();
    if (module == NULL)
        return NULL;

    struct module_state *st = GETSTATE(module);

    /* Add some symbolic constants to the module */
    st->error = PyErr_NewException("Dummy.Error", NULL, NULL);
    if (st->error == NULL) {
        Py_DECREF(module);
        return NULL;
    }

    return module;
}
