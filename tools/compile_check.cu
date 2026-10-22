// Torch-free compile harness: instantiates every kernel config so register/spill/smem usage can be
// inspected without building the extension:
//   nvcc --expt-relaxed-constexpr -arch=sm_86 -std=c++17 -Xptxas -v -cubin -I src/vectorized tools/compile_check.cu -o /dev/null
#include "decode.cuh"
#include "prefill.cuh"

#define DEC(G, MT, ST) template __global__ void w4a16_gemm_mma<G, MT, ST>(const uint4*, const half*, const half*, half*, int, int, int);
#define PRE(G, BM, BN, WM, WN, ST, MINB) \
    template __global__ void pf::w4a16_gemm_prefill<G, BM, BN, WM, WN, ST, MINB>(const uint4*, const half*, const half*, half*, int, int, int);

#define DEC_ALL(G) DEC(G,1,4) DEC(G,1,3) DEC(G,1,2) DEC(G,2,4) DEC(G,2,3) DEC(G,2,2)
#define PRE_ALL(G) \
    PRE(G,16,64,1,4,4,4)  PRE(G,16,64,1,4,3,4)  PRE(G,16,64,1,4,2,4) \
    PRE(G,32,128,1,8,4,2) PRE(G,32,128,1,8,3,2) PRE(G,32,128,1,8,2,2) \
    PRE(G,64,128,2,4,4,2) PRE(G,64,128,2,4,3,2) PRE(G,64,128,2,4,2,2) \
    PRE(G,64,128,1,8,4,2)  PRE(G,64,128,1,8,3,2)  PRE(G,64,128,1,8,2,2)

DEC_ALL(64) DEC_ALL(128)
PRE_ALL(64) PRE_ALL(128)
