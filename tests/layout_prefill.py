# numpy-only emulation of the *prefill* kernel's index math (cp.async tile gather, ldmatrix addressing,
# B-fragment reuse of the packed A-fragment layout, scale lookup, C-fragment store, M/N tails) vs a plain matmul.
#   python tests/layout_prefill.py
# It follows src/vectorized/prefill.cuh line by line; it does not model fp16 rounding or timing.
import itertools
import numpy as np

import importlib.util, os
_spec = importlib.util.spec_from_file_location("layout", os.path.join(os.path.dirname(__file__), "layout.py"))
# layout.py runs its own self-test on import (prints 3 lines); reuse its pack().
import contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    _layout = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_layout)
pack = _layout.pack

XPAD = 72
rng = np.random.default_rng(1)


def emulate(Wp, x, scales, N, K, G, BM, BN, WARPS_M, WARPS_N):
    M = x.shape[0]
    WM, WN = BM // WARPS_M, BN // WARPS_N
    MT, NB, NT = WM // 16, WN // 16, WN // 8
    KT, NBLK = K // 64, N // 16
    GSHIFT = int(np.log2(G // 64))
    Wp = Wp.reshape(NBLK, KT, 32, 4)
    y = np.zeros((M, N))
    for by in range(-(-M // BM)):
        for bx in range(-(-N // BN)):
            m0, n0, nb0 = by * BM, bx * BN, bx * BN // 16
            # accumulators as 16x8 matrices per (warp, mt, nt)
            C = np.zeros((WARPS_M * WARPS_N, MT, NT, 16, 8))
            for kt in range(KT):
                # ---- issue(): global -> smem, zero fill outside M / N
                sX = np.zeros((BM, 64))
                for row in range(BM):
                    if m0 + row < M:
                        sX[row] = x[m0 + row, kt * 64:(kt + 1) * 64]
                sW = np.zeros((BN // 16, 32, 4), dtype=np.uint64)
                for j in range(BN // 16):
                    if nb0 + j < NBLK:
                        sW[j] = Wp[nb0 + j, kt]
                sS = np.zeros(BN)
                for i in range(BN):
                    if n0 + i < N:
                        sS[i] = scales[kt >> GSHIFT, n0 + i]
                for warp in range(WARPS_M * WARPS_N):
                    wm, wn = warp // WARPS_N, warp % WARPS_N
                    wv = {}; slo = {}; shi = {}
                    for lane in range(32):
                        g = lane >> 2
                        for j in range(NB):
                            jb = wn * NB + j
                            wv[lane, j] = sW[jb, lane]
                            slo[lane, j] = sS[jb * 16 + g]; shi[lane, j] = sS[jb * 16 + g + 8]
                    for s in range(4):
                        # ---- ldmatrix.x4 per mt: lane l supplies row (lane&15), col (lane>>4)*8
                        A = np.zeros((MT, 16, 16))
                        for mt in range(MT):
                            addr = lambda l: (wm * WM + mt * 16 + (l & 15), s * 16 + (l >> 4) * 8)
                            reg = np.zeros((32, 4, 2))
                            for l in range(32):
                                for mi in range(4):
                                    r0, c0 = addr(8 * mi + (l >> 2))          # row r'=l>>2 of matrix mi
                                    reg[l, mi] = sX[r0, c0 + 2 * (l & 3): c0 + 2 * (l & 3) + 2]
                            for l in range(32):
                                g, t = l >> 2, l & 3
                                A[mt, g, 2 * t:2 * t + 2] = reg[l, 0]
                                A[mt, g + 8, 2 * t:2 * t + 2] = reg[l, 1]
                                A[mt, g, 2 * t + 8:2 * t + 10] = reg[l, 2]
                                A[mt, g + 8, 2 * t + 8:2 * t + 10] = reg[l, 3]
                        for j in range(NB):
                            B0 = np.zeros((16, 8)); B1 = np.zeros((16, 8))
                            for l in range(32):
                                g, t = l >> 2, l & 3
                                q = int(wv[l, j][s])
                                d = []
                                for p in range(4):                              # dequant8
                                    bits = (q >> (4 * p)) & 0x000F000F
                                    lo, hi = bits & 0xFFFF, bits >> 16
                                    sc = slo[l, j] if p % 2 == 0 else shi[l, j]
                                    d.append(((lo - 8) * sc, (hi - 8) * sc))
                                # b0 = d[0], b1 = d[2]  -> n8 tile 2j ; b0 = d[1], b1 = d[3] -> tile 2j+1
                                B0[2 * t:2 * t + 2, g] = d[0]; B0[2 * t + 8:2 * t + 10, g] = d[2]
                                B1[2 * t:2 * t + 2, g] = d[1]; B1[2 * t + 8:2 * t + 10, g] = d[3]
                            for mt in range(MT):
                                C[warp, mt, 2 * j] += A[mt] @ B0
                                C[warp, mt, 2 * j + 1] += A[mt] @ B1
            # ---- epilogue: per-lane C fragments stored exactly like the kernel
            for warp in range(WARPS_M * WARPS_N):
                wm, wn = warp // WARPS_N, warp % WARPS_N
                for lane in range(32):
                    g, t = lane >> 2, lane & 3
                    for mt in range(MT):
                        row = m0 + wm * WM + mt * 16 + g
                        for nt in range(NT):
                            col = n0 + wn * WN + nt * 8 + 2 * t
                            if col < N:
                                Cm = C[warp, mt, nt]
                                if row < M:
                                    y[row, col], y[row, col + 1] = Cm[g, 2 * t], Cm[g, 2 * t + 1]
                                if row + 8 < M:
                                    y[row + 8, col], y[row + 8, col + 1] = Cm[g + 8, 2 * t], Cm[g + 8, 2 * t + 1]
    return y


CFGS = {1: (16, 64, 1, 4), 2: (32, 128, 1, 8), 3: (64, 128, 2, 4),
        4: (64, 128, 1, 8)}
# (M, N, K, G): ragged M, N not a multiple of BN / 64, K not a multiple of 512, both group sizes
CASES = [(1, 16, 64, 64), (5, 48, 192, 64), (17, 80, 256, 64), (33, 144, 640, 128), (70, 208, 384, 64), (130, 272, 256, 128)]

if __name__ == "__main__":
    worst = 0.0
    for (M, N, K, G), cfg in itertools.product(CASES, CFGS):
        q = rng.integers(0, 16, (N, K), dtype=np.uint8)
        scales = rng.random((K // G, N)) * 0.05 + 0.001
        x = rng.standard_normal((M, K))
        Wdeq = (q.astype(np.float64) - 8) * np.repeat(scales.T, G, axis=1)
        ref = x @ Wdeq.T
        got = emulate(pack(q), x, scales, N, K, G, *CFGS[cfg])
        err = np.abs(got - ref).max() / np.abs(ref).max()
        worst = max(worst, err)
        assert err < 1e-12, (M, N, K, G, cfg, err)
    print(f"prefill index-math emulation OK: {len(CASES) * len(CFGS)} (shape, config) cases, worst rel err {worst:.1e}")
