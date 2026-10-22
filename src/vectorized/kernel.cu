#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <map>
#include <mutex>
#include <tuple>

#include "common.cuh"
#include "decode.cuh"
#include "prefill.cuh"

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


// ============================================================================================ prefill
// Tile configs. Variant ids (also accepted by the optional `variant` argument): 0 = decode kernel,
// 1..4 = the prefill configs below.
template <int CFG> struct PfCfg;
template <> struct PfCfg<1> { static constexpr int BM = 16,  BN = 64,  WM = 1, WN = 4, MINB = 4, ST_MAX = 4; };
template <> struct PfCfg<2> { static constexpr int BM = 32,  BN = 128, WM = 1, WN = 8, MINB = 2, ST_MAX = 4; };
template <> struct PfCfg<3> { static constexpr int BM = 64,  BN = 128, WM = 2, WN = 4, MINB = 2, ST_MAX = 4; };
template <> struct PfCfg<4> { static constexpr int BM = 64,  BN = 128, WM = 1, WN = 8, MINB = 2, ST_MAX = 4; };

constexpr int PF_NCFG = 4;
constexpr int PF_BM[PF_NCFG + 1] = {0, PfCfg<1>::BM, PfCfg<2>::BM, PfCfg<3>::BM, PfCfg<4>::BM};
constexpr int PF_BN[PF_NCFG + 1] = {0, PfCfg<1>::BN, PfCfg<2>::BN, PfCfg<3>::BN, PfCfg<4>::BN};

struct PfInfo { bool ok = false; int stages = 0; int occ = 0; };

// Sets the smem attributes and measures occupancy once per (device, G, cfg, stages) -- never per call.
template <int G, int CFG, int ST>
static bool pf_probe_st(int max_optin, PfInfo& info) {
    using C = PfCfg<CFG>;
    if constexpr (ST > C::ST_MAX) {
        return false;
    } else {
        constexpr int smem = pf::smem_bytes<C::BM, C::BN, ST>();
        if (smem > max_optin) return false;
        auto kfn = pf::w4a16_gemm_prefill<G, C::BM, C::BN, C::WM, C::WN, ST, C::MINB>;
        C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        C10_CUDA_CHECK(cudaFuncSetAttribute(kfn, cudaFuncAttributePreferredSharedMemoryCarveout,
                                            cudaSharedmemCarveoutMaxShared));
        int occ = 0;
        C10_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kfn, C::WM * C::WN * 32, smem));
        if (occ < 1) return false;
        info.ok = true; info.stages = ST; info.occ = occ;
        return true;
    }
}

template <int G, int CFG>
static PfInfo pf_get_info(int dev, int max_optin) {
    static std::mutex mu;
    static std::map<std::tuple<int, int>, PfInfo> cache;
    std::lock_guard<std::mutex> lk(mu);
    auto key = std::make_tuple(dev, CFG);
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    PfInfo info;
    // Probe every viable stage count. Prefer the highest occupancy, then the deeper pipeline.
    // The old short-circuit expression picked ST=4 whenever it fit, even when ST=3 admitted
    // another resident CTA per SM.
    PfInfo candidate;
    if (pf_probe_st<G, CFG, 4>(max_optin, candidate)) info = candidate;
    candidate = PfInfo{};
    if (pf_probe_st<G, CFG, 3>(max_optin, candidate) && candidate.occ > info.occ) info = candidate;
    candidate = PfInfo{};
    if (pf_probe_st<G, CFG, 2>(max_optin, candidate) && candidate.occ > info.occ) info = candidate;
    if (std::getenv("W4A16_DEBUG_PREFILL"))
        std::fprintf(stderr, "prefill G=%d cfg=%d selected_stage=%d occupancy=%d\n", G, CFG,
                     info.stages, info.occ);
    cache[key] = info;
    return info;
}

template <int G, int CFG, int ST>
static void pf_launch_st(const uint4* Wp, const half* x, const half* sc, half* y,
                         int M, int N, int K, cudaStream_t stream) {
    using C = PfCfg<CFG>;
    if constexpr (ST <= C::ST_MAX) {
        constexpr int smem = pf::smem_bytes<C::BM, C::BN, ST>();
        const dim3 grid((N + C::BN - 1) / C::BN, (M + C::BM - 1) / C::BM);
        pf::w4a16_gemm_prefill<G, C::BM, C::BN, C::WM, C::WN, ST, C::MINB>
            <<<grid, C::WM * C::WN * 32, smem, stream>>>(Wp, x, sc, y, M, N, K);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int G, int CFG>
static void pf_launch(int stages, const uint4* Wp, const half* x, const half* sc, half* y,
                      int M, int N, int K, cudaStream_t stream) {
    switch (stages) {
        case 4: pf_launch_st<G, CFG, 4>(Wp, x, sc, y, M, N, K, stream); break;
        case 3: pf_launch_st<G, CFG, 3>(Wp, x, sc, y, M, N, K, stream); break;
        default: pf_launch_st<G, CFG, 2>(Wp, x, sc, y, M, N, K, stream); break;
    }
}

// Generic estimated selector for GPUs without a local profile. This fallback is not a performance
// claim; use `variant=` to compare configs on an unprofiled device.
static int pf_select(int M, int N, int num_sms, const PfInfo (&info)[PF_NCFG + 1]) {
    int best = 0;
    double best_cost = std::numeric_limits<double>::infinity();
    for (int cfg = PF_NCFG; cfg >= 1; --cfg) {          // descending: ties go to the larger BM
        if (!info[cfg].ok) continue;
        const double gy = std::ceil((double)M / PF_BM[cfg]);
        if (gy > 65535.0) continue;                      // grid.y limit
        const double ctas  = std::ceil((double)N / PF_BN[cfg]) * gy;
        const double occ   = info[cfg].occ;
        const double waves = std::ceil(ctas / (num_sms * occ));
        const double cost  = waves * occ * PF_BN[cfg] * (PF_BM[cfg] + 16.0);
        if (cost < best_cost) { best_cost = cost; best = cfg; }
    }
    return best;
}

// Measured routing for the RTX 3050 6GB Laptop GPU (sm_86, 20 SMs), fit from the committed Llama 3
// projection sweep. Config IDs: 1=16x64 1x4, 2=32x128 1x8, 3=64x128 2x4, 4=64x128 1x8.
// Keep this device-specific: other GPUs use the generic estimate until they have their own sweep.
static int pf_select_rtx3050(int M, int N, int K) {
    if (M <= 17) return N >= 16384 ? 2 : 1;
    if (M <= 32) return (N >= 16384 || K >= 8192) ? 2 : 1;
    if (M <= 48) return N > 4096 ? 2 : 1;
    if (M <= 64) return N <= 4096 ? 3 : 2;
    if (M < 512) {
        if (N >= 16384) return 3;
        if (N >= 6000) return 4;
        return 2;
    }
    return 3;
}

template <int G>
static void dispatch_prefill(const torch::Tensor& Wp, const torch::Tensor& x, const torch::Tensor& scales,
                             torch::Tensor& y, int M, int N, int K, cudaStream_t stream, int variant) {
    int dev, max_optin, num_sms, major, minor;
    C10_CUDA_CHECK(cudaGetDevice(&dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&max_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev));
    C10_CUDA_CHECK(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev));

    PfInfo info[PF_NCFG + 1];
    info[1] = pf_get_info<G, 1>(dev, max_optin);
    info[2] = pf_get_info<G, 2>(dev, max_optin);
    info[3] = pf_get_info<G, 3>(dev, max_optin);
    info[4] = pf_get_info<G, 4>(dev, max_optin);

    int cfg = variant;                                   // 1..4 forced, or -1 = auto
    if (cfg < 0) {
        const bool profiled_rtx3050 = major == 8 && minor == 6 && num_sms == 20 && max_optin == 101376;
        const int profiled_cfg = (profiled_rtx3050 && M >= 17) ? pf_select_rtx3050(M, N, K) : 0;
        cfg = (profiled_cfg > 0 && info[profiled_cfg].ok) ? profiled_cfg : pf_select(M, N, num_sms, info);
    }
    TORCH_CHECK(cfg >= 1 && cfg <= PF_NCFG && info[cfg].ok,
                "no prefill configuration fits (shared memory limit ", max_optin, " B, M=", M, ")");
    TORCH_CHECK(((M + PF_BM[cfg] - 1) / PF_BM[cfg]) <= 65535, "M too large for prefill config ", cfg);
    if (std::getenv("W4A16_DEBUG_PREFILL"))
        std::fprintf(stderr, "prefill dispatch G=%d M=%d N=%d K=%d cfg=%d occ=%d stage=%d\n",
                     G, M, N, K, cfg, info[cfg].occ, info[cfg].stages);

    const uint4* wp = reinterpret_cast<const uint4*>(Wp.data_ptr<int32_t>());
    const half*  xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
    const half*  sp = reinterpret_cast<const half*>(scales.data_ptr<at::Half>());
    half*        yp = reinterpret_cast<half*>(y.data_ptr<at::Half>());
    const int st = info[cfg].stages;
    switch (cfg) {
        case 1: pf_launch<G, 1>(st, wp, xp, sp, yp, M, N, K, stream); break;
        case 2: pf_launch<G, 2>(st, wp, xp, sp, yp, M, N, K, stream); break;
        case 3: pf_launch<G, 3>(st, wp, xp, sp, yp, M, N, K, stream); break;
        default: pf_launch<G, 4>(st, wp, xp, sp, yp, M, N, K, stream); break;
    }
}

// ============================================================================================ entry
// variant: -1 = auto, 0 = decode kernel, 1..4 = prefill tile configs (BM = 16/32/64/128).
torch::Tensor w4a16_gemm_cuda(torch::Tensor Wp, torch::Tensor x, torch::Tensor scales, int64_t variant) {
    TORCH_CHECK(Wp.is_cuda() && x.is_cuda() && scales.is_cuda(), "all tensors must be CUDA");
    TORCH_CHECK(Wp.is_contiguous() && x.is_contiguous() && scales.is_contiguous(), "contiguous only");
    TORCH_CHECK(Wp.dtype() == torch::kInt32, "Wp must be int32");
    TORCH_CHECK(x.dtype() == torch::kHalf && scales.dtype() == torch::kHalf, "x, scales must be fp16");
    TORCH_CHECK(x.dim() == 2 && scales.dim() == 2, "x must be [M, K], scales must be [K/G, N]");
    TORCH_CHECK(variant >= -1 && variant <= PF_NCFG, "variant must be -1 (auto), 0 (decode) or 1..", PF_NCFG);

    const int64_t M64 = x.size(0), K64 = x.size(1), N64 = scales.size(1), groups = scales.size(0);
    TORCH_CHECK(M64 >= 1, "M must be at least 1");
    TORCH_CHECK(M64 < (1LL << 31) && K64 < (1LL << 31) && N64 < (1LL << 31), "dimensions must fit in int32");
    const int M = (int)M64, K = (int)K64, N = (int)N64;
    TORCH_CHECK(K % 64 == 0, "K must be a multiple of 64");
    TORCH_CHECK(N % 16 == 0, "N must be a multiple of 16");
    TORCH_CHECK(groups > 0 && K % groups == 0, "K must be divisible by the number of groups");
    TORCH_CHECK(Wp.numel() == (int64_t)N * K / 8, "Wp must hold N*K/8 words");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(Wp.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(x.data_ptr()) % 16 == 0 &&
                reinterpret_cast<uintptr_t>(scales.data_ptr()) % 16 == 0,
                "Wp, x and scales must be 16-byte aligned");

    const int G = K / (int)groups;
    TORCH_CHECK(G == 64 || G == 128, "unsupported group size ", G, ", supported: 64, 128");

    // The decode kernel keeps its original (stricter) shape contract; everything else goes to prefill.
    const bool decode_ok = (M <= 16) && (N % 64 == 0) && (K % 512 == 0);
    int v = (int)variant;
    if (v == -1 && decode_ok) v = 0;
    if (v == 0) {
        TORCH_CHECK(decode_ok, "decode kernel needs M in [1, 16], N a multiple of 64 and K a multiple of 512 "
                               "(use variant=-1 or 1..4 for other shapes)");
    }

    const at::cuda::CUDAGuard guard(x.device());
    auto stream = at::cuda::getCurrentCUDAStream();
    auto y = torch::empty({M, N}, x.options());

    if (v == 0) {
        switch (G) {
            case 64:  dispatch<64>(Wp, x, scales, y, M, N, K, stream);  break;
            default:  dispatch<128>(Wp, x, scales, y, M, N, K, stream); break;
        }
    } else {
        switch (G) {
            case 64:  dispatch_prefill<64>(Wp, x, scales, y, M, N, K, stream, v < 0 ? -1 : v);  break;
            default:  dispatch_prefill<128>(Wp, x, scales, y, M, N, K, stream, v < 0 ? -1 : v); break;
        }
    }
    return y;
}
