#include <torch/extension.h>

torch::Tensor w4a16_gemv_cuda(torch::Tensor W, torch::Tensor A,
                              torch::Tensor sW, torch::Tensor zW);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemv", &w4a16_gemv_cuda, "W4A16 GEMV (CUDA)");
}