#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

__global__ void w4a16_gemv(
    const uint32_t* __restrict__ W,
    const half* __restrict__ A,
    const half* __restrict__ sW,
    const half* __restrict__ zW,
    half* __restrict__ C,
    int N, int K
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    const uint32_t* W_row = W + (size_t)row * (K / 8);   
    float acc = 0.0f;

    for (int k = 0; k < K / 8; k++) {
        uint32_t w = W_row[k];
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            int wv = (w >> (4 * j)) & 0xF;
            if (wv >= 8) wv -= 16;
            acc += (float)wv * __half2float(A[k * 8 + j]);
        }
    }
    float s = __half2float(sW[row]);
    float z = __half2float(zW[row]);
    C[row] = __float2half((acc - z) * s);
}

torch::Tensor w4a16_gemv_cuda(torch::Tensor W, torch::Tensor A,
                              torch::Tensor sW, torch::Tensor zW) {
    // 1. validate
    TORCH_CHECK(W.is_cuda() && A.is_cuda() && sW.is_cuda() && zW.is_cuda(),
                "all tensors must be CUDA");
    TORCH_CHECK(W.is_contiguous() && A.is_contiguous() &&
                sW.is_contiguous() && zW.is_contiguous(),
                "all tensors must be contiguous");
    TORCH_CHECK(W.dtype() == torch::kInt32, "W must be int32 (packed)");
    TORCH_CHECK(A.dtype() == torch::kHalf && sW.dtype() == torch::kHalf &&
                zW.dtype() == torch::kHalf, "A, sW, zW must be float16");

    int N = W.size(0);
    int K = A.size(0);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    TORCH_CHECK(W.size(1) == K / 8, "W must be [N, K/8]");
    TORCH_CHECK(sW.numel() == N && zW.numel() == N, "sW, zW must be [N]");

    const at::cuda::CUDAGuard guard(W.device());
    auto stream = at::cuda::getCurrentCUDAStream();

    auto C = torch::empty({N}, A.options());

    // 4. launch
    int threads = 128;
    int blocks = (N + threads - 1) / threads;
    w4a16_gemv<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const uint32_t*>(W.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(sW.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(zW.data_ptr<at::Half>()),
        reinterpret_cast<half*>(C.data_ptr<at::Half>()),
        N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return C;
}