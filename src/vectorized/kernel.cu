#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ uint32_t smem_u32_ptr(const void* ptr) {
    uint32_t addr;
    asm volatile(
        "{ .reg .u64 smem_addr;\n"
        "  cvta.to.shared.u64 smem_addr, %1;\n"
        "  cvt.u32.u64 %0, smem_addr;\n"
        "}\n"
        : "=r"(addr) : "l"(ptr));
    return addr;
}

__device__ __forceinline__ void ldmatrix_x2(uint32_t (&frag)[2], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(frag[0]), "=r"(frag[1]) : "r"(addr));
}

__device__ __forceinline__ void cp_async16(uint32_t smem_addr, const void* gptr, int src_bytes) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(smem_addr), "l"(gptr), "r"(src_bytes));
}

__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
}

__device__ __forceinline__ void mma_m16n8k16(float (&c)[4], const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__device__ __forceinline__ void dequant8(uint32_t q, uint32_t (&a)[4],
                                         const __half2 s_lo, const __half2 s_hi) {
    const __half2 sub = __float2half2_rn(1032.0f);
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        uint32_t bits = ((q >> (4 * p)) & 0x000F000Fu) | 0x64006400u;
        __half2 h = __hsub2(*reinterpret_cast<__half2*>(&bits), sub);
        h = __hmul2(h, (p & 1) ? s_hi : s_lo);
        a[p] = *reinterpret_cast<uint32_t*>(&h);
    }
}

constexpr int WARPS   = 8;
constexpr int BN      = 64;
constexpr int XS      = 72;
constexpr int RS      = 68;
constexpr int W_BYTES = 4 * 32 * 16;
constexpr int S_BYTES = BN * 2;
constexpr int X_OFF   = W_BYTES + S_BYTES;

constexpr int ilog2(int v) { return v <= 1 ? 0 : 1 + ilog2(v >> 1); }

template <int M_TILES, int STAGES>
constexpr int smem_bytes() {
    return WARPS * STAGES * (X_OFF + M_TILES * 8 * XS * 2);
}

template <int G, int M_TILES, int STAGES>
__global__ void __launch_bounds__(WARPS * 32)
w4a16_gemm_mma(const uint4* __restrict__ Wp,
               const half*  __restrict__ x,
               const half*  __restrict__ scales,
               half*        __restrict__ y,
               int M, int N, int K)
{
    static_assert(G >= 64 && (G & (G - 1)) == 0, "G must be a power of two >= 64");
    static_assert(STAGES >= 2, "need at least 2 stages");
    constexpr int GSHIFT      = ilog2(G / 64);
    constexpr int M_TOK       = M_TILES * 8;
    constexpr int X_BYTES     = M_TOK * XS * 2;
    constexpr int STAGE_BYTES = X_OFF + X_BYTES;
    static_assert(WARPS * M_TOK * RS * 4 <= WARPS * STAGES * STAGE_BYTES,
                  "reduction buffer must fit in the aliased pipeline smem");

    extern __shared__ __align__(16) char smem_raw[];

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int g = lane >> 2;
    const int t = lane & 3;

    const int n_base = blockIdx.x * BN;
    const int nb0    = blockIdx.x * (BN / 16);
    const int KC     = K / 64;
    const int ITERS  = K / (64 * WARPS);

    char* warp_smem = smem_raw + (size_t)warp * STAGES * STAGE_BYTES;

    auto issue = [&](int it, int stage) {
        const int kc = warp * ITERS + it;
        char* sbase  = warp_smem + stage * STAGE_BYTES;
        const uint32_t sWa = smem_u32_ptr(sbase);
        const uint32_t sSa = smem_u32_ptr(sbase + W_BYTES);
        const uint32_t sXa = smem_u32_ptr(sbase + X_OFF);
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const uint4* src = Wp + ((size_t)(nb0 + r) * KC + kc) * 32 + lane;
            cp_async16(sWa + (r * 32 + lane) * 16, src, 16);
        }
        if (lane < BN / 8) {
            const half* src = scales + (size_t)(kc >> GSHIFT) * N + n_base + lane * 8;
            cp_async16(sSa + lane * 16, src, 16);
        }
        #pragma unroll
        for (int i = 0; i < M_TOK * 8 / 32; ++i) {
            const int sidx = lane + 32 * i;
            const int tok  = sidx >> 3;
            const int part = sidx & 7;
            const bool ok  = tok < M;
            const half* src = x + (ok ? (size_t)tok * K + (size_t)kc * 64 + part * 8 : 0);
            cp_async16(sXa + (tok * XS + part * 8) * 2, src, ok ? 16 : 0);
        }
    };

    float c[4][M_TILES][4];
    #pragma unroll
    for (int r = 0; r < 4; ++r)
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) c[r][mt][i] = 0.f;

    #pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < ITERS) issue(s, s);
        cp_async_commit();
    }

    for (int it = 0; it < ITERS; ++it) {
        cp_async_wait<STAGES - 2>();
        __syncwarp();
        const int nxt = it + STAGES - 1;
        if (nxt < ITERS) issue(nxt, nxt % STAGES);
        cp_async_commit();

        const char* sbase = warp_smem + (it % STAGES) * STAGE_BYTES;
        const uint4* sw   = reinterpret_cast<const uint4*>(sbase);
        const half*  ss   = reinterpret_cast<const half*>(sbase + W_BYTES);
        const uint32_t xs = smem_u32_ptr(sbase + X_OFF);

        uint32_t wq[4][4];
        __half2 s_lo[4], s_hi[4];
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            uint4 v = sw[r * 32 + lane];
            wq[r][0] = v.x; wq[r][1] = v.y; wq[r][2] = v.z; wq[r][3] = v.w;
            s_lo[r] = __half2half2(ss[r * 16 + g]);
            s_hi[r] = __half2half2(ss[r * 16 + g + 8]);
        }

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t b[M_TILES][2];
            #pragma unroll
            for (int mt = 0; mt < M_TILES; ++mt) {
                const int tok = mt * 8 + (lane & 7);
                const int kk  = s * 16 + ((lane >> 3) & 1) * 8;
                ldmatrix_x2(b[mt], xs + (tok * XS + kk) * 2);
            }
            #pragma unroll
            for (int r = 0; r < 4; ++r) {
                uint32_t a[4];
                dequant8(wq[r][s], a, s_lo[r], s_hi[r]);
                #pragma unroll
                for (int mt = 0; mt < M_TILES; ++mt)
                    mma_m16n8k16(c[r][mt], a, b[mt]);
            }
        }
    }

    cp_async_wait<0>();
    __syncthreads();
    float* red = reinterpret_cast<float*>(smem_raw);

    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        #pragma unroll
        for (int mt = 0; mt < M_TILES; ++mt) {
            const int n0 = r * 16 + g;
            const int m0 = mt * 8 + 2 * t;
            float* base = red + (size_t)warp * M_TOK * RS;
            base[(m0    ) * RS + n0    ] = c[r][mt][0];
            base[(m0 + 1) * RS + n0    ] = c[r][mt][1];
            base[(m0    ) * RS + n0 + 8] = c[r][mt][2];
            base[(m0 + 1) * RS + n0 + 8] = c[r][mt][3];
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < M_TOK * BN; e += WARPS * 32) {
        const int m = e / BN, n = e % BN;
        if (m < M) {
            float sum = 0.f;
            #pragma unroll
            for (int w = 0; w < WARPS; ++w) sum += red[(w * M_TOK + m) * RS + n];
            y[(size_t)m * N + n_base + n] = __float2half(sum);
        }
    }
}

template <int G, int MT, int ST>
static bool try_launch(const torch::Tensor& Wp, const torch::Tensor& x, const torch::Tensor& scales,
                       torch::Tensor& y, int M, int N, int K, cudaStream_t stream, int max_optin) {
    constexpr int smem = smem_bytes<MT, ST>();
    if (smem > max_optin) return false;
    auto kfn = w4a16_gemm_mma<G, MT, ST>;
    C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
    kfn<<<N / BN, WARPS * 32, smem, stream>>>(
        reinterpret_cast<const uint4*>(Wp.data_ptr<int32_t>()),
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(scales.data_ptr<at::Half>()),
        reinterpret_cast<half*>(y.data_ptr<at::Half>()),
        M, N, K);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return true;
}

template <int G>
static void dispatch(const torch::Tensor& Wp, const torch::Tensor& x, const torch::Tensor& scales,
                     torch::Tensor& y, int M, int N, int K, cudaStream_t stream) {
    int dev, max_optin;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    bool ok;
    if (M <= 8) {
        ok = try_launch<G, 1, 4>(Wp, x, scales, y, M, N, K, stream, max_optin)
          || try_launch<G, 1, 3>(Wp, x, scales, y, M, N, K, stream, max_optin)
          || try_launch<G, 1, 2>(Wp, x, scales, y, M, N, K, stream, max_optin);
    } else {
        ok = try_launch<G, 2, 4>(Wp, x, scales, y, M, N, K, stream, max_optin)
          || try_launch<G, 2, 3>(Wp, x, scales, y, M, N, K, stream, max_optin)
          || try_launch<G, 2, 2>(Wp, x, scales, y, M, N, K, stream, max_optin);
    }
    TORCH_CHECK(ok, "no pipeline configuration fits in ", max_optin, " B of shared memory");
}

torch::Tensor w4a16_gemm_cuda(torch::Tensor Wp, torch::Tensor x, torch::Tensor scales) {
    TORCH_CHECK(Wp.is_cuda() && x.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(Wp.is_contiguous() && x.is_contiguous() && scales.is_contiguous(), "contiguous only");
    TORCH_CHECK(Wp.dtype() == torch::kInt32, "Wp must be int32");
    TORCH_CHECK(x.dtype() == torch::kHalf && scales.dtype() == torch::kHalf, "x, scales must be fp16");
    TORCH_CHECK(x.dim() == 2 && scales.dim() == 2, "x must be [M, K], scales must be [K/G, N]");

    const int M = x.size(0), K = x.size(1), N = scales.size(1);
    const int groups = scales.size(0);
    TORCH_CHECK(M >= 1 && M <= 16, "M must be in [1, 16]");
    TORCH_CHECK(K % (64 * WARPS) == 0, "K must be a multiple of 512");
    TORCH_CHECK(N % BN == 0, "N must be a multiple of 64");
    TORCH_CHECK(groups > 0 && K % groups == 0, "K must be divisible by the number of groups");
    TORCH_CHECK(Wp.numel() == (int64_t)N * K / 8, "Wp must hold N*K/8 words");

    const int G = K / groups;
    const at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto y = torch::empty({M, N}, x.options());

    switch (G) {
        case 64:  dispatch<64>(Wp, x, scales, y, M, N, K, stream);  break;
        case 128: dispatch<128>(Wp, x, scales, y, M, N, K, stream); break;
        default:  TORCH_CHECK(false, "unsupported group size ", G, ", supported: 64, 128");
    }
    return y;
}