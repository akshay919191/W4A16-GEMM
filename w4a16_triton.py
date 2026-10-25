import importlib.util
import os
import sys

_kernel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "triton", "kernel.py")
_spec = importlib.util.spec_from_file_location("_w4a16_triton_kernel", _kernel_path)
_kernel = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _kernel
_spec.loader.exec_module(_kernel)

W4Weights = _kernel.W4Weights
quantize_w4 = _kernel.quantize_w4
dequantize_w4 = _kernel.dequantize_w4
unpack_w4_codes = _kernel.unpack_w4_codes
w4a16_gemm = _kernel.w4a16_gemm
