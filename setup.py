import os
import shutil

import torch
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

if torch.version.cuda is None:
    raise SystemExit("This PyTorch build has no CUDA support.")
if shutil.which("nvcc") is None and not os.environ.get("CUDA_HOME"):
    raise SystemExit(
        "nvcc not found. Install the CUDA toolkit and put nvcc on PATH "
        f"(PyTorch was built for CUDA {torch.version.cuda})."
    )

if "TORCH_CUDA_ARCH_LIST" not in os.environ:
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    else:
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

cxx_flags = ["-O3", "-std=c++17"]
nvcc_flags = ["-O3", "-std=c++17", "-lineinfo"]

setup(
    name="w4a16",
    version="0.1.0",
    ext_modules=[
        CUDAExtension(
            name="w4a16_ext",
            sources=["src/baseline/extension.cpp", "src/baseline/kernel.cu"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
        CUDAExtension(
            name="w4a16_opt",
            sources=["src/vectorized/extension.cpp", "src/vectorized/kernel.cu"],
            extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
        ),
    ],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
)