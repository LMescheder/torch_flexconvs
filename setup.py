from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='_cuda_ext',
    ext_modules=[
        CUDAExtension('_cuda_ext', [

        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })