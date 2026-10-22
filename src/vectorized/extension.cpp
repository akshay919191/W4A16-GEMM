#include <torch/extension.h>

torch::Tensor w4a16_gemm_cuda(torch::Tensor Wp, torch::Tensor x, torch::Tensor scales, int64_t variant);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_gemm", &w4a16_gemm_cuda,
          "W4A16 group-wise tensor-core GEMM (decode M<=16 fast path + tiled prefill for any M)",
          py::arg("Wp"), py::arg("x"), py::arg("scales"), py::arg("variant") = -1);
}
