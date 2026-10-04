from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="w4a16_ext",
    ext_modules=[
        CUDAExtension(
            name="w4a16_opt",
            sources=["extension.cpp", "kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-arch=sm_80"],   # sm_80 A100, sm_86 3090/A6000, sm_89 4090, sm_90 H100
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)