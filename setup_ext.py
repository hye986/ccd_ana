"""
Build the C pattern-recognition extension in-place.

Usage
-----
  python setup_ext.py build_ext --inplace

The compiled .so will appear in  pnccd_ana/lib/
and is imported automatically by pattern_recognition.py.
"""

from setuptools import setup, Extension
import numpy as np

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
