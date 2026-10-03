from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="w4a16_ext",
    ext_modules=[
        CUDAExtension(
            name="w4a16_ext",
            sources=["extension.cpp", "kernel.cu"],   # both files
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)