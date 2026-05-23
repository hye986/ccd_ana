/*
 * pnccd_ana/lib/_pattern_recognition_c.c
 * ========================================
 * C extension for hot-path pattern recognition.
 *
 * !! DO NOT add grade definitions here !!
 * The grade table and pattern pixel lists are generated from Python's
 * _GRADE_DEFS in pattern_recognition.py.  Run:
 *
 *   python -c "from pattern_recognition import write_grade_table_header; \
 *              write_grade_table_header()"
 *
 * to regenerate _grade_table_generated.h, then recompile.
 *
 * Exports:
 *
 *   local_max_5x5(frame)
 *       5×5 sliding-window maximum of a float32 2-D array.
 *
 *   find_events_c(frame, noise, lmax, seed_sigma, split_sigma, reject_extra)
 *       Full inner loop: seed detection + central 3×3 pattern classification.
 *       Returns (rows, cols, grades, signals) as 1-D numpy arrays.
 *
 *   check_grade_table(table_list)
 *       Accepts a Python list of 256 ints (the expected table from Python).
 *       Returns a list of (bitmask, c_grade, py_grade) triples where they
 *       differ.  Used by pattern_recognition.py to detect stale builds.
 *
 * Neighbour bit-encoding (central 3×3, 8 neighbours):
 *   bit 0 (0x01) : right      (dY= 0, dX=+1)
 *   bit 1 (0x02) : up         (dY=+1, dX= 0)
 *   bit 2 (0x04) : left       (dY= 0, dX=-1)
 *   bit 3 (0x08) : down       (dY=-1, dX= 0)
 *   bit 4 (0x10) : up-right   (dY=+1, dX=+1)
 *   bit 5 (0x20) : up-left    (dY=+1, dX=-1)
 *   bit 6 (0x40) : down-left  (dY=-1, dX=-1)
 *   bit 7 (0x80) : down-right (dY=-1, dX=+1)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <string.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

/*
 * Grade table and pattern pixel lists — generated from pattern_recognition.py.
 * Defines: GRADE_TABLE[256], PATTERN_PIXELS[N_GRADES][MAX_PATTERN_PIXELS][2],
 *          N_GRADE_DEFS, GRADE_OTHER, MAX_PATTERN_PIXELS.
 */
#include "_grade_table_generated.h"

/* ── 5×5 local maximum ────────────────────────────────────────────────────── */

static PyObject *py_local_max_5x5(PyObject *self, PyObject *args)
{
    PyArrayObject *frame_arr;
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &frame_arr))
        return NULL;

    if (PyArray_NDIM(frame_arr) != 2 ||
        PyArray_TYPE(frame_arr) != NPY_FLOAT32 ||
        !PyArray_IS_C_CONTIGUOUS(frame_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "frame must be a C-contiguous 2-D float32 array");
        return NULL;
    }

    npy_intp n_Y = PyArray_DIM(frame_arr, 0);
    npy_intp n_X = PyArray_DIM(frame_arr, 1);
    float *f = (float *)PyArray_DATA(frame_arr);

    npy_intp dims[2] = {n_Y, n_X};
    PyArrayObject *out_arr = (PyArrayObject *)PyArray_ZEROS(2, dims, NPY_FLOAT32, 0);
    if (!out_arr) return NULL;
    float *out = (float *)PyArray_DATA(out_arr);

    npy_intp y, x, dy, dx;
    for (y = 0; y < n_Y; y++) {
        npy_intp y0 = y - 2 < 0     ? 0      : y - 2;
        npy_intp y1 = y + 2 >= n_Y  ? n_Y-1  : y + 2;
        for (x = 0; x < n_X; x++) {
            npy_intp x0 = x - 2 < 0     ? 0      : x - 2;
            npy_intp x1 = x + 2 >= n_X  ? n_X-1  : x + 2;
            float mx = f[y0 * n_X + x0];
            for (dy = y0; dy <= y1; dy++)
                for (dx = x0; dx <= x1; dx++) {
                    float v = f[dy * n_X + dx];
                    if (v > mx) mx = v;
                }
            out[y * n_X + x] = mx;
        }
    }
    return (PyObject *)out_arr;
}

/* ── dynamic output buffer ────────────────────────────────────────────────── */

#define INIT_CAP 4096

typedef struct {
    uint16_t *rows;
    uint16_t *cols;
    uint8_t  *grades;
    float    *signals;
    Py_ssize_t len;
    Py_ssize_t cap;
} ResultBuf;

static int buf_push(ResultBuf *b,
                    uint16_t row, uint16_t col, uint8_t grade, float sig)
{
    if (b->len >= b->cap) {
        Py_ssize_t new_cap = b->cap * 2;
        uint16_t *r = realloc(b->rows,    new_cap * sizeof(uint16_t));
        uint16_t *c = realloc(b->cols,    new_cap * sizeof(uint16_t));
        uint8_t  *g = realloc(b->grades,  new_cap * sizeof(uint8_t));
        float    *s = realloc(b->signals, new_cap * sizeof(float));
        if (!r || !c || !g || !s) return -1;
        b->rows = r; b->cols = c; b->grades = g; b->signals = s;
        b->cap = new_cap;
    }
    b->rows[b->len]    = row;
    b->cols[b->len]    = col;
    b->grades[b->len]  = grade;
    b->signals[b->len] = sig;
    b->len++;
    return 0;
}

static void buf_free(ResultBuf *b)
{
    free(b->rows); free(b->cols); free(b->grades); free(b->signals);
}

/* ── full inner recognition loop ─────────────────────────────────────────── */

static PyObject *py_find_events_c(PyObject *self, PyObject *args)
{
    PyArrayObject *frame_arr, *noise_arr, *lmax_arr;
    double seed_sigma, split_sigma;
    int reject_extra;

    if (!PyArg_ParseTuple(args, "O!O!O!ddi",
                          &PyArray_Type, &frame_arr,
                          &PyArray_Type, &noise_arr,
                          &PyArray_Type, &lmax_arr,
                          &seed_sigma, &split_sigma,
                          &reject_extra))
        return NULL;

    if (PyArray_NDIM(frame_arr)  != 2 || PyArray_TYPE(frame_arr)  != NPY_FLOAT32 ||
        PyArray_NDIM(noise_arr)  != 2 || PyArray_TYPE(noise_arr)  != NPY_FLOAT32 ||
        PyArray_NDIM(lmax_arr)   != 2 || PyArray_TYPE(lmax_arr)   != NPY_FLOAT32 ||
        !PyArray_IS_C_CONTIGUOUS(frame_arr) ||
        !PyArray_IS_C_CONTIGUOUS(noise_arr) ||
        !PyArray_IS_C_CONTIGUOUS(lmax_arr)) {
        PyErr_SetString(PyExc_ValueError,
                        "All arrays must be C-contiguous 2-D float32");
        return NULL;
    }

    npy_intp n_Y = PyArray_DIM(frame_arr, 0);
    npy_intp n_X = PyArray_DIM(frame_arr, 1);
    float *f     = (float *)PyArray_DATA(frame_arr);
    float *noise = (float *)PyArray_DATA(noise_arr);
    float *lmax  = (float *)PyArray_DATA(lmax_arr);
    float seed_s  = (float)seed_sigma;
    float split_s = (float)split_sigma;

    ResultBuf buf;
    buf.len  = 0;
    buf.cap  = INIT_CAP;
    buf.rows    = malloc(INIT_CAP * sizeof(uint16_t));
    buf.cols    = malloc(INIT_CAP * sizeof(uint16_t));
    buf.grades  = malloc(INIT_CAP * sizeof(uint8_t));
    buf.signals = malloc(INIT_CAP * sizeof(float));
    if (!buf.rows || !buf.cols || !buf.grades || !buf.signals) {
        buf_free(&buf);
        return PyErr_NoMemory();
    }

    npy_intp y, x;
    for (y = 2; y < n_Y - 2; y++) {
        for (x = 2; x < n_X - 2; x++) {
            float centre = f[y * n_X + x];

            /* Seed threshold */
            if (centre <= seed_s * noise[y * n_X + x])
                continue;

            /* Centre must be the 5×5 local maximum.
             * Use strict less-than: if another pixel in the window equals
             * centre, that pixel will also be a candidate and will pass its
             * own lmax test at its own position. */
            if (centre < lmax[y * n_X + x])
                continue;

            /* Build 8-bit neighbour bitmask for the central 3×3. */
            uint8_t nbr = 0;

#define TEST_NBR(dY, dX, bit)                                           \
    do {                                                                 \
        float st = split_s * noise[(y+(dY)) * n_X + (x+(dX))];         \
        if (f[(y+(dY)) * n_X + (x+(dX))] > st) nbr |= (uint8_t)(bit); \
    } while(0)

            TEST_NBR( 0,+1, 0x01);   /* right      */
            TEST_NBR(+1, 0, 0x02);   /* up          */
            TEST_NBR( 0,-1, 0x04);   /* left        */
            TEST_NBR(-1, 0, 0x08);   /* down        */
            TEST_NBR(+1,+1, 0x10);   /* up-right    */
            TEST_NBR(+1,-1, 0x20);   /* up-left     */
            TEST_NBR(-1,-1, 0x40);   /* down-left   */
            TEST_NBR(-1,+1, 0x80);   /* down-right  */
#undef TEST_NBR

            uint8_t grade = GRADE_TABLE[nbr];

            if (reject_extra && grade == GRADE_OTHER)
                continue;

            /* Sum signal pixels for this grade.
             * PATTERN_PIXELS[grade] lists (dY,dX) pairs; {99,99} terminates. */
            float sig = 0.0f;
            int k;
            for (k = 0; k < MAX_PATTERN_PIXELS && PATTERN_PIXELS[grade][k][0] != 99; k++) {
                int8_t dy = PATTERN_PIXELS[grade][k][0];
                int8_t dx = PATTERN_PIXELS[grade][k][1];
                sig += f[(y + dy) * n_X + (x + dx)];
            }

            if (buf_push(&buf, (uint16_t)y, (uint16_t)x, grade, sig) < 0) {
                buf_free(&buf);
                return PyErr_NoMemory();
            }
        }
    }

    /* Convert buffers → numpy arrays */
    npy_intp n = (npy_intp)buf.len;
    PyObject *arr_rows = NULL, *arr_cols = NULL,
             *arr_grades = NULL, *arr_sigs = NULL;

#define MAKE_ARRAY(name, ctype, data, dtype)                             \
    name = PyArray_SimpleNew(1, &n, dtype);                              \
    if (!name) goto oom;                                                 \
    if (n > 0) memcpy(PyArray_DATA((PyArrayObject*)(name)),             \
                      data, (size_t)n * sizeof(ctype));

    MAKE_ARRAY(arr_rows,   uint16_t, buf.rows,    NPY_UINT16)
    MAKE_ARRAY(arr_cols,   uint16_t, buf.cols,    NPY_UINT16)
    MAKE_ARRAY(arr_grades, uint8_t,  buf.grades,  NPY_UINT8)
    MAKE_ARRAY(arr_sigs,   float,    buf.signals,  NPY_FLOAT32)
#undef MAKE_ARRAY

    buf_free(&buf);
    return Py_BuildValue("(OOOO)", arr_rows, arr_cols, arr_grades, arr_sigs);

oom:
    Py_XDECREF(arr_rows); Py_XDECREF(arr_cols);
    Py_XDECREF(arr_grades); Py_XDECREF(arr_sigs);
    buf_free(&buf);
    return PyErr_NoMemory();
}

/* ── grade table self-check ───────────────────────────────────────────────── */
/*
 * check_grade_table(py_table) -> list of (bitmask, c_grade, py_grade) triples
 * where C and Python disagree.  Called at import time by pattern_recognition.py
 * to catch stale builds.
 */
static PyObject *py_check_grade_table(PyObject *self, PyObject *args)
{
    PyObject *py_table;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &py_table))
        return NULL;
    if (PyList_Size(py_table) != 256) {
        PyErr_SetString(PyExc_ValueError, "table must have exactly 256 entries");
        return NULL;
    }
    PyObject *mismatches = PyList_New(0);
    if (!mismatches) return NULL;
    int i;
    for (i = 0; i < 256; i++) {
        long py_grade = PyLong_AsLong(PyList_GetItem(py_table, i));
        if (py_grade == -1 && PyErr_Occurred()) {
            Py_DECREF(mismatches);
            return NULL;
        }
        if ((int)GRADE_TABLE[i] != (int)py_grade) {
            PyObject *triple = Py_BuildValue("(iii)", i,
                                             (int)GRADE_TABLE[i], (int)py_grade);
            if (!triple) { Py_DECREF(mismatches); return NULL; }
            PyList_Append(mismatches, triple);
            Py_DECREF(triple);
        }
    }
    return mismatches;
}

/* ── module definition ────────────────────────────────────────────────────── */

static PyMethodDef methods[] = {
    {"local_max_5x5",
     py_local_max_5x5, METH_VARARGS,
     "local_max_5x5(frame) -> ndarray\n"
     "5×5 sliding-window maximum of a float32 2-D array."},
    {"find_events_c",
     py_find_events_c, METH_VARARGS,
     "find_events_c(frame, noise, lmax, seed_sigma, split_sigma, reject_extra)"
     " -> (rows, cols, grades, signals)\n"
     "Full C inner loop for photon-event recognition."},
    {"check_grade_table",
     py_check_grade_table, METH_VARARGS,
     "check_grade_table(py_table) -> list of (mask, c_grade, py_grade) mismatches"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_pattern_recognition_c", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__pattern_recognition_c(void)
{
    import_array();
    /* Grade table is static (from generated header) — no runtime build needed. */
    return PyModule_Create(&moduledef);
}
