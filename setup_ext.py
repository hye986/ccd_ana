"""
Build the C pattern-recognition extension in-place.

Usage
-----
  python setup_ext.py build_ext --inplace

The compiled .so will appear in  pnccd_ana/lib/
and is imported automatically by pattern_recognition.py.

This script automatically regenerates _grade_table_generated.h from the
_GRADE_DEFS in pattern_recognition.py before compiling the C extension.
"""

from setuptools import setup, Extension
import numpy as np
import sys

# Regenerate the grade table header before compilation.
# This ensures the C code picks up any changes to _GRADE_DEFS.
sys.path.insert(0, str(__file__).rsplit("/", 1)[0])
from pnccd_ana.lib.pattern_recognition import write_grade_table_header
write_grade_table_header()
print("[setup_ext] Regenerated _grade_table_generated.h")

ext = Extension(
    name="pnccd_ana.lib._pattern_recognition_c",
    sources=["pnccd_ana/lib/_pattern_recognition_c.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        "-O3",
        "-march=native",
        "-ffast-math",
        "-Wall",
    ],
)

setup(
    name="pnccd_ana",
    version="0.1.0",
    ext_modules=[ext],
)
