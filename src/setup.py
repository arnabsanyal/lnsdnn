from distutils.core import setup, Extension
import numpy

module1 = Extension('native_matr_mult_wrapper',
        sources = ['log_multiplier_kernel.c'],
        # define_macros = [('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
        include_dirs=[numpy.get_include()],
        extra_compile_args = ['-fopenmp'],
        extra_link_args = ['-lgomp'])

setup (name = 'native_matr_mult_wrapper',
        version = '1.0',
        description = 'This is a cache inefficient log-matrix multiplier wrapper',
        ext_modules = [module1])

