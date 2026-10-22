#pragma once
// Shared device helpers for the W4A16 kernels (moved verbatim out of kernel.cu).
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

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&frag)[4], uint32_t addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(frag[0]), "=r"(frag[1]), "=r"(frag[2]), "=r"(frag[3]) : "r"(addr));
}

constexpr int ilog2(int v) { return v <= 1 ? 0 : 1 + ilog2(v >> 1); }
