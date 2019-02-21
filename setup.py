from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


cuda_extension = CUDAExtension(
    'torch_flexconv._cuda_ext',
    sources=[
        'torch_flexconv/src/extension.cc',
        'torch_flexconv/src/flex_conv_cpu.cc',
        'torch_flexconv/src/flex_conv_cuda.cu',
        'torch_flexconv/src/flex_deconv_cpu.cc',
        'torch_flexconv/src/flex_deconv_cuda.cu',
        'torch_flexconv/src/flex_pool_cpu.cc',
        'torch_flexconv/src/flex_pool_cuda.cu',
    ],
    extra_compile_args={
        'cxx': [],
        'nvcc': ['-arch=sm_61'],
    }
)

setup(
    name='torch_flexconv',
    version='0.1',
    description="This is a PyTorch adaptation for FlexConvolutions.",
    authors='Lars Mescheder',
    author_email='larsmescheder@gmx.net',
    packages=find_packages(),
    ext_modules=[cuda_extension],
    cmdclass={
        'build_ext': BuildExtension
    })
