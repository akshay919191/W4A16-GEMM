"""W4A16 decode GEMM (sm_80+, tuned for sm_86). JIT-compiles on first call."""
import os

import torch
from torch.utils.cpp_extension import load

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ext = None


def _load():
    global _ext
    if _ext is None:
        cc = torch.cuda.get_device_capability()
        if cc < (8, 0):
            raise RuntimeError(f"w4a16 needs cp.async / mma.sync (sm_80+), got sm_{cc[0]}{cc[1]}")
        arch = f"{cc[0]}{cc[1]}"
        _ext = load(
            name="w4a16_ext",
            sources=[os.path.join(_ROOT, "csrc", "binding.cpp"),
                     os.path.join(_ROOT, "csrc", "w4a16_gemm.cu")],
            extra_cflags=["-O3", "-std=c++17"],
            extra_cuda_cflags=["-O3", "-std=c++17", "-lineinfo",
                               f"-gencode=arch=compute_{arch},code=sm_{arch}"],
            verbose=bool(int(os.environ.get("W4A16_VERBOSE", "0"))),
        )
    return _ext


def w4a16_gemm(Wp: torch.Tensor, x: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """y[M,N] = x[M,K] @ W^T.  Wp: int32 packed (see README), x: fp16 [M,K], scales: fp16 [K/G,N]."""
    return _load().gemm(Wp, x, scales)