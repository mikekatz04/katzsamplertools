# from future.utils import iteritems
from setuptools import setup

setup(
    name="katzsamplertools",
    # Random metadata. there's more you can supply
    author="Michael Katz",
    version="0.1",
    packages=[
        "katzsamplertools",
        "katzsamplertools.utils",
        "katzsamplertools.samplers",
    ],
    py_modules=["katzsamplertools.sampler_main"],
)
