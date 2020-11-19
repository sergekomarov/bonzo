# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_module = Extension(
    "surf_bright",
    ["surf_bright.pyx"]
)
setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    include_dirs = [np.get_include()]
    )
