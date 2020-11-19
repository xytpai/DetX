from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='detx_ext_cuda',
    ext_modules=[
        CUDAExtension(
            'detx_ext_cuda', ['detx_ext_cuda.cpp',
            'assign_bbox_kernel.cu', 'sigmoid_focal_loss_kernel.cu',
            'roi_align_corners_kernel.cu'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
})
