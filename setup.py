# -*- coding: utf-8 -*-

import numpy as np
import sys, os, glob

from scipy._build_utils import numpy_nodepr_api

from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

from shutil import copy,move, copytree, rmtree
from distutils.spawn import find_executable

import re
import argparse
from fnmatch import filter


# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

# ===========================================================================

# Clean up the build.

def clean_c_files(dir):
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if os.path.isfile(path) and path.endswith(".c"):
            # check if the .c file is a cythonized .pyx and then remove it
            if os.path.exists('.'.join(path.split('.')[:-1] +['pyx'])):
                os.remove(path)
        elif os.path.isdir(path):
            clean_c_files(path)



if 'clean' in sys.argv:

    # delete the build folder
    if os.path.isdir('build'):
        rmtree('build')
    if os.path.isdir('src_c'):
        rmtree('src_c')
    if os.path.isdir('bnz'):
        rmtree('bnz')

    if os.path.isdir('src'):
        clean_c_files('src')

    sys.exit()


# ==========================================================================

# Move shared libraries to bnz/

def move_out_so_files(src_dir, bnz_dir):

    if not os.path.exists(bnz_dir):
        os.mkdir(bnz_dir)

    def move_so_files(dir):

        for file in os.listdir(dir):

            path = os.path.join(dir, file)
            path_new = path.replace(path[:path.index(os.path.sep)], bnz_dir)

            if os.path.isfile(path):

                if path.endswith(".so"):
                    move(path, path_new)
                if path.endswith(".py"):
                    copy(path, path_new)

            elif os.path.isdir(path):
                if not os.path.exists(path_new): os.mkdir(path_new)
                move_so_files(path)

    move_so_files(src_dir)



def move_out_c_files(src_dir, src_dir_c, ext_names_c):


    if not os.path.exists(src_dir_c):
        os.mkdir(src_dir_c)


    names_c = [name.split('.')[-1]+'.c' for name in ext_names_c]

    def move_c_files(dir):

        for file in os.listdir(dir):

            path = os.path.join(dir, file)
            path_new = path.replace(path[:path.index(os.path.sep)], src_dir_c)

            if os.path.isfile(path):

                if path.endswith(".c"):
                    name = path.split(os.path.sep)[-1]
                    if name in names_c: copy(path, path_new)
                    else: move(path, path_new)

                if path.endswith(".h"):
                    copy(path, path_new)

                if path.endswith("__init__.py"):
                    copy(path, path_new)

            elif os.path.isdir(path):
                if not os.path.exists(path_new): os.mkdir(path_new)
                move_c_files(path)

    move_c_files(src_dir)


#--------------------------------------------------------------------

def move_in_so_files(bnz_dir, src_dir):

    def move_so_files_(dir):

        for file in os.listdir(dir):

            path = os.path.join(dir, file)
            path_new = path.replace(path[:path.index(os.path.sep)], src_dir)

            if os.path.isfile(path):
                if path.endswith(".so"):
                    move(path, path_new)

            elif os.path.isdir(path):
                if not os.path.exists(path_new): os.mkdir(path_new)
                move_so_files_(path)

    if os.path.exists(bnz_dir):
        move_so_files_(bnz_dir)


def move_in_c_files(src_dir_c, ext_names_c, bnz_dir, src_dir):

    names_c = [name.split('.')[-1]+'.c' for name in ext_names_c]

    def move_c_files(dir):

        for file in os.listdir(dir):

            path = os.path.join(dir, file)
            path_new = path.replace(path[:path.index(os.path.sep)], src_dir)

            if os.path.isfile(path):
                if path.endswith(".c"):
                    name = path.split(os.path.sep)[-1]
                    if not (name in names_c): move(path, path_new)

            elif os.path.isdir(path):
                if not os.path.exists(path_new): os.mkdir(path_new)
                move_c_files(path)

    if os.path.exists(src_dir_c):
        move_c_files(src_dir_c)
    if os.path.exists(bnz_dir):
        move_c_files(bnz_dir)




# ============================================================================


# Parse command line arguments.

parser = argparse.ArgumentParser()

parser.add_argument("cmd", choices=['build_ext', 'clean'])
parser.add_argument("--inplace", action="store_true", help="build in-place")

parser.add_argument("-mf", action="store_true", help="magnetic fields on")

parser.add_argument("-cgl", action="store_true", help="solve CGL equations")

parser.add_argument("-ion-tc", action="store_true",
                help="anisotropic ion thermal conduction on"+
                     "(enables magnetic field and CGL)")

parser.add_argument("-mhdpic", action="store_true",
                help="MHDPIC on (enables magnetic fields)")

parser.add_argument("-pic", action="store_true", help="particle-in-cell on")

parser.add_argument("-two-temp", action="store_true",
                help="separate ion and electron temperatures")


parser.add_argument("-d1", action="store_true", help="1 space dimension")

parser.add_argument("-d2", action="store_true", help="2 space dimensions")

parser.add_argument("-d3", action="store_true", help="3 space dimensions")


parser.add_argument("-fixed-dt", action="store_true", help="fixed timestep")


parser.add_argument("-sprec", action="store_true", help="use single precision")


parser.add_argument("-mpi", action="store_true",
                help="use MPI (enables double precision)")

parser.add_argument("-omp", action="store_true", help="use OpenMP")

parser.add_argument("-omp-nt", type=int, help="number of OpenMP threads")


parser.add_argument("--dir", help="user output directory")

parser.add_argument("--prob", help="problem name")

# compiler choices
cc_choices = [
    'gcc',
    'gcc-simd',
    'icc',
    'icc-debug',
    'icc-phi',
    'clang',
    'clang-simd',
    'clang-apple'
    ]

parser.add_argument('--cc', choices=cc_choices, help='C compiler')


args = parser.parse_args()


# now remove all arguments except 'build_ext' and '--inplace'
setup_args = sys.argv[:2]
if '--inplace' in sys.argv:
    setup_args.append('--inplace')
sys.argv = setup_args[:]


# Default compile-time options.

compile_time_env = {
    'MFIELD':0,
    'MHDPIC':0,
    'PIC':0,

    'CGL':0,
    'IONTC':0,
    'TWOTEMP':0,

    'MPI':0,
    'OMP':0,

    'SPREC':0,
    'FIXDT':0,
    # 'USR_DIR':'.',

    'D2D':0,
    'D3D':0,
    'OMP_NT':1,

    'XNONUNI':0,
    'YNONUNI':0,
    'ZNONUNI':0,
    'SPHER_COORD':0,
    'CYLIND_COORD':0
    }


# Set user compile-time options.

# physics

if args.mf:
    compile_time_env['MFIELD'] = 1

if args.cgl:
    compile_time_env['CGL'] = 1
    compile_time_env['MFIELD'] = 1

if args.ion_tc:
    compile_time_env['IONTC'] = 1
    compile_time_env['CGL'] = 1
    compile_time_env['MFIELD'] = 1

if args.mhdpic:
    compile_time_env['MHDPIC'] = 1
    compile_time_env['MFIELD'] = 1

if args.pic: compile_time_env['PIC'] = 1

if args.two_temp:
    compile_time_env['TWOTEMP'] = 1


# dimensionality

if (args.d1 and args.d2) or (args.d1 and args.d3) or (args.d2 and args.d3):
    print('error: use a single dimensionality flag')
    sys.exit(1)

if args.d1:
    compile_time_env['D2D'] = 0
    compile_time_env['D3D'] = 0

if args.d2:
    compile_time_env['D2D'] = 1
    compile_time_env['D3D'] = 0

if args.d3:
    compile_time_env['D2D'] = 1
    compile_time_env['D3D'] = 1


# fixed timestep

if args.fixed_dt:
    compile_time_env['FIXDT'] = 1


# precision

if args.sprec:
    compile_time_env['SPREC'] = 1


# parallelization

if args.mpi:
    compile_time_env['MPI'] = 1
    compile_time_env['SPREC'] = 1

if args.omp:
    compile_time_env['OMP'] = 1

if args.omp_nt != None:
    compile_time_env['OMP'] = 1
    compile_time_env['OMP_NT'] = args.omp_nt


# ==============================================================================


# set user directory

# if args.dir != None:
#     compile_time_env['USR_DIR'] = args.dir
#     usr_dir = args.dir
# else:
#     usr_dir = '.'


# assign a problem

if args.prob != None:
    problem_name = args.prob
else:
    problem_name = 'otvortex'

# make a copy of the problem file and name it problem.pyx

os.chdir('src/problem')

problem_fname = problem_name+'.pyx'

if not os.path.isfile(problem_fname):
    print "error: problem file "+problem_fname+" doesn't exist"
    sys.exit(1)

copy(problem_fname, 'problem.pyx')

os.chdir('../..')



#===============================================================================

# Set up compiler flags.

# choose default compiler depending on the platform

if args.cc == None:

    if sys.platform.startswith('darwin'):
        # use clang included in MacOS
        args.cc = 'clang-apple'
    else:
        if find_executable('icc') != None:
            # use Intel C compiler if available
            args.cc = 'icc'
        else:
            # try to use GCC on other platforms
            args.cc = 'gcc-simd'

print args.cc


compiler_command = []
compiler_flags = []
include_flags = []
link_flags = []
library_flags = []


# convert cython definitions to c macros
macros_c = [(key, str(value)) for (key,value) in compile_time_env.items()]


if args.cc == 'gcc' or args.cc == 'gcc-simd':
    compiler_command = 'gcc'

if args.cc == 'icc' or args.cc == 'icc-debug' or args.cc == 'icc-phi':
    compiler_command = 'icc'
    macros_c.append(('__PURE_INTEL_C99_HEADERS__', 1))

if args.cc == 'clang' or args.cc == 'clang-simd' or args.cc == 'clang-apple':
    compiler_command = 'clang'


if args.cc == 'gcc':
    compiler_flags += ['-O2', '-fopenmp-simd',
    '-march=native',
    '-fopt-info-vec-all']

if args.cc == 'gcc-simd':
    # GCC version >= 4.9, for OpenMP 4.0; version >= 6.1 for OpenMP 4.5 support
    compiler_flags += [
        '-O3', '-fopenmp-simd',
        '-fwhole-program', '-flto',
        '-freciprocal-math',
        '-fprefetch-loop-arrays',
        '-march=native',
        '-ffast-math',

        # '-march=skylake-avx512', 'skylake', 'core-avx2',
        # '-mprefer-vector-width=128',  # available in gcc-8, but not gcc-7
        # '-mtune=native, generic, broadwell',
        # '-mprefer-avx128',
        # '-m64' # (default)
        ]


if args.cc == 'icc':
    compiler_flags += [
      '-O3', '-xhost', '-inline-forceinline', #'-ipo',
      '-qopenmp-simd', '-qopt-prefetch=4', '-qoverride-limits'
      # '-qopt-report-phase=ipo', # (does nothing without -ipo)
      # '-qopt-zmm-usage=high'  # typically harms multi-core performance on Skylake Xeon
    ]


if args.cc == 'icc-debug':
    # Disable IPO, forced inlining, and fast math. Enable vectorization reporting.
    # Useful for testing symmetry, SIMD-enabled functions and loops with OpenMP 4.5
    compiler_flags += [
      '-O3', '-xhost', '-qopenmp-simd', '-fp-model precise',
      '-qopt-prefetch=4', '-qopt-report=5', '-qopt-report-phase=openmp,vec',
      '-g', '-qoverride-limits'
    ]

if args.cc == 'icc-phi':
    # Cross-compile for Intel Xeon Phi x200 KNL series (unique AVX-512ER and AVX-512FP)
    # -xMIC-AVX512: generate AVX-512F, AVX-512CD, AVX-512ER and AVX-512FP
    compiler_flags += [
      '-O3', '-ipo', '-xMIC-AVX512', '-inline-forceinline', '-qopenmp-simd',
      '-qopt-prefetch=4', '-qoverride-limits'
    ]

if args.cc == 'clang++':
    compiler_flags += ['-O3']

if args.cc == 'clang-simd':
    # LLVM/Clang version >= 3.9 for most of OpenMP 4.0 and 4.5 (still incomplete; no
    # offloading, target/declare simd directives). OpenMP 3.1 fully supported in LLVM 3.7
    compiler_flags += ['-O3', '-fopenmp-simd']

if args.cc == 'clang-apple':
    # Apple LLVM/Clang: forked version of the open-source LLVM project bundled in macOS
    compiler_flags += ['-O3']


# -debug argument
# if args['debug']:
#     definitions['DEBUG_OPTION'] = 'DEBUG'
#     # Completely replace the --cxx= sets of default compiler flags, disable optimization,
#     # and emit debug symbols in the compiled binaries
#     if (args.cc == 'gcc' or args.cc == 'gcc-simd'
#             or args.cc == 'icc' or args.cc == 'icc-debug'
#             or args.cc == 'clang' or args.cc == 'clang-simd'
#             or args.cc == 'clang-apple'):
#         compiler_flags = '-O0 --std=c11 -g'  # -Og
#
#     if args.cc == 'icc-phi':
#         compiler_flags = '-O0 --std=c11 -g -xMIC-AVX512'
# else:
#     definitions['DEBUG_OPTION'] = 'NOT_DEBUG'


# -mpi argument
if args.mpi:
    compiler_command = 'mpicc'

# HDF5 flags
library_flags += ['-lhdf5']


# -omp argument
if args.omp:

    if (args.cc == 'gcc' or args.cc == 'gcc-simd' or args.cc == 'clang'
            or args.cc == 'clang-simd'):
        compiler_flags += ['-fopenmp']

    if (args.cc == 'clang-apple'):
        # Apple Clang disables the front end OpenMP driver interface; enable it via the
        # preprocessor. Must install LLVM's OpenMP runtime library libomp beforehand
        compiler_flags += ['-Xpreprocessor', '-fopenmp']
        library_flags += ['-lomp']

    if args.cc == 'icc' or args.cc == 'icc-debug' or args.cc == 'icc-phi':
        compiler_flags += ['-qopenmp']

else:

    if args.cc == 'icc' or args.cc == 'icc-debug' or args.cc == 'icc-phi':
        # suppressed messages:
        #   3180: pragma omp not recognized
        compiler_flags += ['-diag-disable 3180']



# assemble all flags of any sort given to compiler
compiler_flags = (compiler_flags + library_flags +
                  include_flags + link_flags)

# set the compiler environment variable used by setup
os.environ['CC'] = compiler_command


print compiler_command
print compiler_flags




# ==============================================================================

# Check cython, h5py, and mpi4py installations.

if compile_time_env['MPI']:

    try:
        import mpi4py
    except ImportError as err:
        print "mpi4py import error: {0}".format(err)
        sys.exit(1)

try:
    import h5py
except ImportError as err:
    print "h5py import error: {0}".format(err)
    sys.exit(1)

try:
    import cython
except ImportError as err:
    print "cython import error: {0}".format(err)
    sys.exit(1)



#===============================================================================

# List module and package names.


if not compile_time_env['PIC']:

    ext_names_c = ["mhd.eos", "mhd.reconstr", "mhd.fluxes", "dissipation.heat_fluxes_e"]

    ext_names_pyx = ["defs_cy", "data_struct",  "coord",  "mhd.diagnostics_mhd",
                 "utils", "bc.utils_bc", "dissipation.utils_diffuse", #"utils_particle",
                 "mhd.new_dt", "mhd.eos_cy", "gravity", "turb_driv",

                 "problem.problem",

                 "bc.bc_grid", #"bc.bc_prt", "bc.bc_mhdpic_exch",

                 "dissipation.thcond_elec", #"dissipation.thcond_ion",
                 #"dissipation.visc", "dissipation.visc4",
                 #"dissipation.resist", "dissipation.resist4",

                 "dissipation.diffuse", #"dissipation.collisions",

                 "read_config", #"init_mpi_blocks",

                 "output", "restart",

                 "mhd.init_mhd", "init",

                 "mhd.godunov", #"mhd.ct",

                 "mhd.integr_mhd_jobs",
                 "mhd.integr_mhd_vl", "mhd.integr_mhd_rk3", "integrate",

                 "simulation"]

    if compile_time_env['MFIELD']:
        ext_names_pyx.insert(21, "mhd.ct")

    if compile_time_env['MPI']:
        ext_names_pyx.insert(16, "init_mpi_blocks")

    if compile_time_env['TWOTEMP'] or compile_time_env['CGL']:
        ext_names_pyx.insert(17, "dissipation.collisions")

    if compile_time_env['IONTC']:
        ext_names_pyx.insert(18, "dissipation.thcond_ion")

    if compile_time_env['MHDPIC']:
        ext_names_pyx.insert(13, "bc.bc_mhdpic_exch")
        ext_names_pyx.insert(13, "bc.bc_prt")
        ext_names_pyx.insert(7, "utils_particle")

    packages = ["", "mhd", "bc", "dissipation", "problem"]

    if compile_time_env['MHDPIC']: packages.append(".mhdpic")

else:

    ext_names_c = []

    ext_names_pyx = ["defs_cy", "data_struct",  "coord",  "pic.diagnostics_pic",
                 "utils", "bc.utils_bc", "utils_particle",

                 "problem.problem",

                 "bc.bc_grid", "bc.bc_prt", "bc.bc_pic_exch",

                 "read_config", #"init_mpi_blocks",

                 "output", "restart",

                 "pic.init_pic", "init",

                 "pic.field", "pic.deposit", "pic.move",
                 "pic.integr_pic", "integrate",

                 "simulation"]

    if compile_time_env['MPI']:
        ext_names_pyx.insert(12, "init_mpi_blocks")

    packages = ["", "pic", "bc", "problem"]



# ================================================================================

# Build shared libraries from .c source files.

# .pyx and .c source directory
src_dir = 'src'

# cythonized .c source directory
src_dir_c = 'src_c'

# .so directory
bnz_dir = 'bnz'


# check if previously compiled modules exist and move them back into src/
# this way the setup script can detect which sources need to be recompiled
if os.path.exists(bnz_dir) or os.path.exists(src_dir_c):
    move_in_so_files(bnz_dir, src_dir)
    move_in_c_files(src_dir_c, ext_names_c, bnz_dir, src_dir)
if os.path.exists(bnz_dir):
    rmtree(bnz_dir)
if os.path.exists(src_dir_c):
    rmtree(src_dir_c)


# rename src/ folder and compile modules in-place, then move the source files
# to a separate new folder src/
# os.rename('src', 'bnz')

copytree(src_dir, bnz_dir)


packages_bnz = ['bnz.'+package for package in packages]

# absolute paths to directories containing C libraries and headers
include_dirs = [package.replace('.',os.path.sep) for package in packages_bnz]
library_dirs = include_dirs
# names of C libraries
libraries_c = [ext_name_c.split('.')[-1] for ext_name_c in ext_names_c]


def makeExtension_c(name):
    # need to prepend 'lib' to the names of shared C libs
    name_split = name.split('.')
    ext_name =  '.'.join([ bnz_dir ] + name_split[:-1] + [ 'lib'+name_split[-1] ])
    ext_path = ('.'.join([ bnz_dir, name])).replace('.',os.path.sep) + '.c'
    return Extension(
        ext_name,
        [ext_path],
        include_dirs=include_dirs,
        extra_compile_args=compiler_flags,
        extra_link_args=library_flags+link_flags,
        define_macros = macros_c
        )

cextensions_c  = [makeExtension_c(name) for name in ext_names_c]


# setup C libraries
setup(
  name="bnz",
  packages=packages_bnz,
  ext_modules=cextensions_c,
  cmdclass = {'build_ext': build_ext}
)



# =============================================================================

# Make cython extensions.

# set environment variable to link against the C libraries

if os.environ.get("LD_LIBRARY_PATH") != None:
    os.environ["LD_LIBRARY_PATH"] += os.pathsep + os.pathsep.join(library_dirs)
else:
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(library_dirs)


def makeExtension_pyx(name):

    ext_name =  '.'.join([bnz_dir, name])
    ext_path = ext_name.replace('.',os.path.sep) + '.pyx'

    return Extension(
        ext_name,
        [ext_path],
        include_dirs=[np.get_include(), '.'] + include_dirs,
        libraries=libraries_c,
        library_dirs=library_dirs,
        runtime_library_dirs=library_dirs,
        extra_compile_args=compiler_flags,
        extra_link_args=library_flags+link_flags,
        define_macros = macros_c
        )

# make extnsions
extensions_pyx = [makeExtension_pyx(name) for name in ext_names_pyx]


# directives for compiling .pyx files
cy_compiler_directives = {
    'boundscheck' : False,
    'wraparound' : False,
    'initializedcheck' : False,
    'cdivision' : True,
    'language_level' : 2
    }


cextensions_pyx = cythonize(extensions_pyx,
                       compile_time_env=compile_time_env,
                       compiler_directives=cy_compiler_directives)


# setup Cython modules
setup(
  name="bnz",
  packages=packages_bnz,
  ext_modules=cextensions_pyx,
  cmdclass = {'build_ext': build_ext}
)

# move share libraries and cythonized .pyx out of /src
# move shared libraries to bnz/
# move .c files including cythonized .pyx files to src_c/

rmtree(src_dir)
os.rename(bnz_dir, src_dir)
move_out_so_files(src_dir, bnz_dir)
move_out_c_files(src_dir, src_dir_c, ext_names_c)
