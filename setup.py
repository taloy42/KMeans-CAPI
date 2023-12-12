from setuptools import setup, Extension

# we want to setup with the following mandatory parameters:
setup(
    name="mykmeanssp",  # the name we want the module to be called (does not have to be the same as in the C API)
    version="1.1",  # just the version, pretty self-explanatory
    ext_modules=[Extension("mykmeanssp", sources=["kmeans.c"])]  # all the modules we want to export to
    # our new module.
    # every module is from the form : Extension(module name as stated in the C file, sources=[the C file itself])
)