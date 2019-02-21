from setuptools import setup
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


src_files = [
    'src/extension.cc',
    'src/flex_conv_cpu.cc',
    'src/flex_conv_cuda.cu',
    'src/flex_deconv_cpu.cc',
    'src/flex_deconv_cuda.cu',
]

setup(
    name='torch_flexconv',
    ext_modules=[
        CUDAExtension('_cuda_ext', src_files,
            extra_compile_args={
                'cxx': [],
                'nvcc': ['-arch=sm_61'],
        }),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
