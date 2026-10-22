#pragma once
// Decode kernel (1 <= M <= 16). Body is unchanged from the original kernel.cu.
#include "common.cuh"

constexpr int WARPS   = 8;
constexpr int BN      = 64;
constexpr int XS      = 72;
constexpr int RS      = 68;
constexpr int W_BYTES = 4 * 32 * 16;
constexpr int S_BYTES = BN * 2;
constexpr int X_OFF   = W_BYTES + S_BYTES;

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
