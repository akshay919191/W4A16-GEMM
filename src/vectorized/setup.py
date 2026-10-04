from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="w4a16_opt",
    ext_modules=[
        CUDAExtension(
            name="w4a16_opt",
            sources=["extension.cpp", "kernel.cu"],
            extra_compile_args={"cxx": ["-O3"], "nvcc": ["-O3", "-arch=sm_86"]},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)