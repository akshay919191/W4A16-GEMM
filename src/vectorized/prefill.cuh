#pragma once
// Prefill kernel: any M >= 1, N % 16 == 0, K % 64 == 0, G in {64, 128}.
//
// Same weight format as the decode kernel (each lane's 32-bit word is already an mma fragment), but
// the roles are swapped so tokens are the mma "M" side and weights the "N" side:
//
//     C[token, n] += X[token, k] (A, via ldmatrix.x4)  *  Wdeq[n, k]^T (B, dequantized in registers)
//
// The lane's 8 nibbles (n = g and g+8, k = 2t,2t+1,2t+8,2t+9) are exactly the B fragments of two
// m16n8k16 n8-tiles, so the packed layout needs no change. The output fragment is already y[token, n],
// so no transpose / smem reduction is needed in the epilogue.
//
// CTA tile BM x BN, K walked in steps of 64 (one packed chunk per n16 block), ST-stage cp.async pipeline,
// one __syncthreads per k-step. Accumulation over K is sequential in fp32 inside each output element, so
// all prefill tile configs give bit-identical results.
#include "common.cuh"

namespace pf {

constexpr int XPAD = 72;   // x smem row stride in halves (144 B): 16B-aligned, ldmatrix conflict-free

template <int BM, int BN, int ST>
constexpr int smem_bytes() {
    return ST * (BM * XPAD * 2 + BN * 32 + BN * 2);
}

template <int G, int BM, int BN, int WARPS_M, int WARPS_N, int ST, int MINB>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, MINB)
w4a16_gemm_prefill(const uint4* __restrict__ Wp,
                   const half*  __restrict__ x,
                   const half*  __restrict__ scales,
                   half*        __restrict__ y,
                   int M, int N, int K)
{
    static_assert(G >= 64 && (G & (G - 1)) == 0, "G must be a power of two >= 64");
    static_assert(ST >= 2, "need at least 2 stages");
    constexpr int T  = WARPS_M * WARPS_N * 32;
    constexpr int WM = BM / WARPS_M;          // warp tile (tokens)
    constexpr int WN = BN / WARPS_N;          // warp tile (output channels)
    constexpr int MT = WM / 16;               // m16 tiles per warp
    constexpr int NB = WN / 16;               // packed n16 blocks per warp
    constexpr int NT = WN / 8;                // n8 mma tiles per warp
    static_assert(BM % WARPS_M == 0 && BN % WARPS_N == 0 && WM % 16 == 0 && WN % 16 == 0, "bad warp tiling");

    constexpr int X_BYTES     = BM * XPAD * 2;
    constexpr int W_BYTES     = BN * 32;          // (BN/16) blocks * 512 B
    constexpr int S_BYTES     = BN * 2;
    constexpr int STAGE_BYTES = X_BYTES + W_BYTES + S_BYTES;
    constexpr int X_SEGS      = BM * 8;           // 16-byte segments
    constexpr int W_SEGS      = BN * 2;           // (BN/16) blocks * 32 lanes
    constexpr int S_SEGS      = BN / 8;
    static_assert(X_SEGS % T == 0 && W_SEGS % T == 0 && S_SEGS <= T, "load mapping needs T | segments");
    constexpr int GSHIFT      = ilog2(G / 64);

    extern __shared__ __align__(16) char smem_raw[];

    const int tid    = threadIdx.x;
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int g      = lane >> 2;
    const int t      = lane & 3;
    const int warp_m = warp / WARPS_N;
    const int warp_n = warp % WARPS_N;

    const int n0   = blockIdx.x * BN;
    const int m0   = blockIdx.y * BM;
    const int nb0  = n0 >> 4;
    const int NBLK = N >> 4;
    const int KT   = K >> 6;                      // k-steps of 64 == packed chunks per n16 block

    const uint32_t smem_base = smem_u32_ptr(smem_raw);

    auto issue = [&](int kt, int stage) {
        const uint32_t sX = smem_base + stage * STAGE_BYTES;
        const uint32_t sW = sX + X_BYTES;
        const uint32_t sS = sW + W_BYTES;
        #pragma unroll
        for (int r = 0; r < W_SEGS / T; ++r) {
            const int i  = tid + r * T;
            const int nb = nb0 + (i >> 5);
            const bool ok = nb < NBLK;
            const uint4* src = Wp + ((size_t)(ok ? nb : 0) * KT + kt) * 32 + lane;
            cp_async16(sW + i * 16, src, ok ? 16 : 0);
        }
        if (tid < S_SEGS) {
            const int n = n0 + tid * 8;
            const bool ok = n < N;
            const half* src = scales + (size_t)(kt >> GSHIFT) * N + (ok ? n : 0);
            cp_async16(sS + tid * 16, src, ok ? 16 : 0);
        }
        #pragma unroll
        for (int r = 0; r < X_SEGS / T; ++r) {
            const int i    = tid + r * T;
            const int row  = i >> 3;
            const int part = i & 7;
            const int tok  = m0 + row;
            const bool ok  = tok < M;
            const half* src = x + (ok ? (size_t)tok * K + (size_t)kt * 64 + part * 8 : 0);
            cp_async16(sX + (row * XPAD + part * 8) * 2, src, ok ? 16 : 0);
        }
    };

    float c[MT][NT][4];
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt)
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) c[mt][nt][i] = 0.f;

    #pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (s < KT) issue(s, s);
        cp_async_commit();
    }

    int rd = 0;
    int wr = ST - 1;
    for (int it = 0; it < KT; ++it) {
        cp_async_wait<ST - 2>();
        __syncthreads();                      // stage `rd` landed for everyone; stage `wr` is free
        if (it + ST - 1 < KT) issue(it + ST - 1, wr);
        cp_async_commit();

        const char*    sbase = smem_raw + rd * STAGE_BYTES;
        const uint4*   sw    = reinterpret_cast<const uint4*>(sbase + X_BYTES);
        const half*    ss    = reinterpret_cast<const half*>(sbase + X_BYTES + W_BYTES);
        const uint32_t xa     = smem_base + rd * STAGE_BYTES
                              + ((warp_m * WM + (lane & 15)) * XPAD + (lane >> 4) * 8) * 2;

        uint4 wv[NB];
        __half2 s_lo[NB], s_hi[NB];
        #pragma unroll
        for (int j = 0; j < NB; ++j) {
            const int jb = warp_n * NB + j;
            wv[j]   = sw[jb * 32 + lane];
            s_lo[j] = __half2half2(ss[jb * 16 + g]);
            s_hi[j] = __half2half2(ss[jb * 16 + g + 8]);
        }

        #pragma unroll
        for (int s = 0; s < 4; ++s) {
            uint32_t a[MT][4];
            #pragma unroll
            for (int mt = 0; mt < MT; ++mt)
                ldmatrix_x4(a[mt], xa + (mt * 16 * XPAD + s * 16) * 2);
            #pragma unroll
            for (int j = 0; j < NB; ++j) {
                const uint32_t q = (s == 0) ? wv[j].x : (s == 1) ? wv[j].y : (s == 2) ? wv[j].z : wv[j].w;
                uint32_t d[4];
                dequant8(q, d, s_lo[j], s_hi[j]);
                // d[p] = (n = g + 8*(p&1), k = 2t + 8*(p>>1) + {0,1}); a B fragment is {k 2t.., k 2t+8..} of one n
                const uint32_t b0[2] = {d[0], d[2]};   // n8 tile 2j   : n = 16*jb + g
                const uint32_t b1[2] = {d[1], d[3]};   // n8 tile 2j+1 : n = 16*jb + 8 + g
                #pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    mma_m16n8k16(c[mt][2 * j],     a[mt], b0);
                    mma_m16n8k16(c[mt][2 * j + 1], a[mt], b1);
                }
            }
        }
        rd = (rd + 1 == ST) ? 0 : rd + 1;
        wr = (wr + 1 == ST) ? 0 : wr + 1;
    }
    cp_async_wait<0>();

    // c0,c1: token g, n = 2t,2t+1 ; c2,c3: token g+8. N % 16 == 0 so n8 tiles are all-in or all-out.
    #pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row = m0 + warp_m * WM + mt * 16 + g;
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const int col = n0 + warp_n * WN + nt * 8 + 2 * t;
            if (col < N) {
                if (row < M)
                    *reinterpret_cast<__half2*>(y + (size_t)row * N + col) =
                        __floats2half2_rn(c[mt][nt][0], c[mt][nt][1]);
                if (row + 8 < M)
                    *reinterpret_cast<__half2*>(y + (size_t)(row + 8) * N + col) =
                        __floats2half2_rn(c[mt][nt][2], c[mt][nt][3]);
            }
        }
    }
}

}  // namespace pf
