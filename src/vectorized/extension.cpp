#include <torch/extension.h>

torch::Tensor w4a16_gemm_cuda(torch::Tensor Wp, torch::Tensor x, torch::Tensor scales);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm_cuda, "W4A16 group-wise tensor-core GEMM");
}