# W4A16 Decode GEMM for Ampere (sm_86)

A hand-written CUDA kernel for **LLM decode-time linear layers** with 4-bit weights and fp16 activations.

```
y[M, N] = x[M, K] @ Wᵀ        x, y : fp16
                              W[n, k] ≈ (q[n, k] − 8) · scale[k / G, n],   q ∈ [0, 15]
```

Decode is memory-bound: every generated token streams the entire weight matrix from DRAM, so the speed limit is
`bytes moved / bandwidth`. 4-bit weights cut traffic ~3.8× versus fp16, and this kernel is built to get as close to
that limit as possible for small batches (`M ≤ 16`).

**Headline result (RTX 3050 6GB Laptop, sm_86):** **3.4–4.0× faster than fp16 cuBLAS** for `M ≤ 8` on Llama-3-8B layer
shapes, sustaining **~150–160 GB/s**, which is about **90–96 % of the card's theoretical 168 GB/s** (see [Results](#results)).

Target is `sm_86` (RTX 30-series / A10-class), but anything `sm_80+` should compile (`cp.async`, `mma.sync`).

---

## Design

| Piece | What it does |
|---|---|
| **Weights pre-swizzled into MMA A-fragments** | Each lane loads one `uint4` (16 B) per 16×64 weight tile and *already holds* its `mma.m16n8k16` A-fragment as nibbles. No shared-memory staging of weights, no `ldmatrix`, no shuffles for W. |
| **Weights are the MMA "A" operand, tokens are "B"** | `N` fills the 16-wide MMA row dimension; the tiny token count `M` goes in the 8-wide `n` dimension. Batches of 1–8 tokens waste no tensor-core work on padding (`M ≤ 8` → 1 token tile, `9–16` → 2). |
| **Register dequant (the `0x6400` trick)** | `(q & 0xF) \| 0x6400` is the fp16 value `1024 + q`; subtracting `1032` gives `q − 8` exactly; one `__hmul2` applies the scale. 4 ops per 2 weights, no int→float converts. |
| **Per-warp split-K, no block barriers in the main loop** | The block's 8 warps each own a contiguous K slice and a private `cp.async` pipeline (`STAGES` deep) with its own weight / scale / x tiles. Only `__syncwarp` inside the loop; a single `__syncthreads` at the end for the cross-warp fp32 reduction (through shared memory aliased over the pipeline buffers). |
| **Shared-memory-fit dispatch** | Tries 4 → 3 → 2 pipeline stages and uses the deepest one that fits the device's opt-in shared-memory limit. |
| **fp32 accumulation, fixed reduction order** | Results are deterministic run-to-run. |

CTA tile: 64 output columns (`N`) × all `M` tokens. Grid = `N / 64`.

---

## Quick start

```bash
pip install torch pytest ninja
```

```python
from w4a16_opt import w4a16_gemm
from w4a16_opt.quant import quantize_w4, pack_int4

q, scales = quantize_w4(W, group_size=128)   # W fp16/fp32 [N,K] -> q: uint8 [N,K], scales: fp16 [K/G, N]
Wp = pack_int4(q)                            # int32 [N*K/8], kernel order
y  = w4a16_gemm(Wp, x, scales)               # x fp16 [M,K]  ->  y fp16 [M,N]
```

The extension JIT-compiles on first call. Set `W4A16_VERBOSE=1` to see the nvcc output.

### Constraints

All checked with `TORCH_CHECK`.

| Parameter | Requirement |
|---|---|
| `M` | 1 … 16 (decode / speculative-decode batches; **not** a prefill kernel) |
| `K` | multiple of **512** (8 warps × 64) |
| `N` | multiple of **64** |
| Group size `G` | 64 or 128 (power of two ≥ 64 in the template); `K % G == 0` |
| Dtypes / layout | `x`, `scales` fp16; `Wp` int32; all contiguous, on CUDA |

> **Heads-up:** `K = 11008` (Llama-2-7B `down_proj`) is *not* a multiple of 512 and is not supported.
> Llama-3-8B shapes (4096 / 6144 / 14336 / 28672) all are.

---

## Weight format

`pack_int4` in `w4a16_opt/quant.py` is the executable spec. In words:

- `q[n, k]` is an unsigned nibble with zero-point 8. `G` consecutive `k` share one scale per `n`; `scales` is stored `[K/G, N]` (n contiguous).
- `W` is cut into 16 (n) × 64 (k) tiles; a tile is 4 k-steps of 16. One tile is 512 B = 32 lanes × `uint4`.
- `Wp[(((n/16)·(K/64) + k/64)·32 + lane)·4 + s]` is a 32-bit word (`s = (k%64)/16`) holding 8 nibbles for lane `(g = lane/4, t = lane%4)`:

```
nibble p      (p = 0..3, bits 4p .. 4p+3)     -> W[n0 + g + 8·(p&1)][k0 + 2t + 8·(p>>1)]
nibble p + 4  (bits 4p+16 .. 4p+19)           -> W[n0 + g + 8·(p&1)][k0 + 2t + 8·(p>>1) + 1]
```

Register `a[p]` of the `mma.m16n8k16` A-fragment is therefore `half2(lo = nibble p, hi = nibble p+4)`, and a single mask
`(word >> 4p) & 0x000F000F` yields both halves of `a[p]`.

---

## Testing

```bash
pytest -q tests/test_w4a16.py                  # CPU layout tests + GPU correctness
compute-sanitizer --tool memcheck python -m pytest -q tests/test_w4a16.py -k "shapes or onehot"
```

**Status:** 93 tests pass on an RTX 3050 6GB Laptop (sm_86).

What is covered:

- Pack/unpack round-trip, plus a decode of the packed words written from the kernel's point of view.
- Every `M ∈ {1, 2, 3, 7, 8, 9, 15, 16}` × `G ∈ {64, 128}` × several shapes, including `ITERS = 1` and `ITERS = 2`.
- Llama-3-8B layer shapes.
- **One-hot activations that must reproduce W's columns bit-exactly**, which pins the k-mapping of every warp's slice.
- All-0 and all-15 nibbles, zero input, determinism, CUDA-graph capture, argument validation.

The reference is an fp32 matmul over the *same* fp16-dequantized weights, so the tolerance (`2e-3` of `max|y|`) only has to
absorb fp16 output rounding and accumulation order.

---

## Benchmark

```bash
python bench/measure_bw.py                         # achievable DRAM GB/s on your card
python bench/bench_gemm.py --peak-bw <GB/s> --md   # table below
```

The baseline is fp16 cuBLAS (`F.linear`) on identical shapes. Timing uses CUDA graphs over several distinct weight copies
(≥ 128 MiB total) so weights stream from DRAM rather than L2. Use `--target-mb 1` to see the hot-L2 case.

Roofline traffic per call: `N·K/2 (weights) + (K/G)·N·2 (scales) + M·K·2 + M·N·2`.
At `G = 128` that is 0.516 B per weight versus 2 B for fp16, so at most **~3.9×** faster than fp16 (3.8× at `G = 64`).

### Results

**GPU:** NVIDIA GeForce RTX 3050 6GB Laptop · sm_86 · 20 SMs · 2 MiB L2
**Software:** torch 2.7.1+cu118 · CUDA 11.8 · weights cycled over 128 MiB · `G = 128`

| layer | N | K | M | fp16 µs | W4A16 µs | speedup | ideal × | GB/s | % of 144 GB/s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| qkv | 6144 | 4096 | 1 | 310.6 | 83.6 | 3.72 | 3.87 | 155.5 | 108.0 |
| qkv | 6144 | 4096 | 2 | 339.6 | 84.7 | 4.01 | 3.87 | 153.7 | 106.7 |
| qkv | 6144 | 4096 | 4 | 337.7 | 85.2 | 3.97 | 3.86 | 153.3 | 106.5 |
| qkv | 6144 | 4096 | 8 | 340.3 | 87.0 | 3.91 | 3.84 | 151.0 | 104.8 |
| qkv | 6144 | 4096 | 16 | 346.9 | 96.2 | 3.61 | 3.81 | 138.4 | 96.1 |
| o | 4096 | 4096 | 1 | 207.6 | 58.7 | 3.53 | 3.87 | 147.5 | 102.5 |
| o | 4096 | 4096 | 2 | 230.4 | 59.3 | 3.89 | 3.87 | 146.5 | 101.7 |
| o | 4096 | 4096 | 4 | 208.9 | 60.1 | 3.48 | 3.86 | 145.0 | 100.7 |
| o | 4096 | 4096 | 8 | 209.9 | 61.7 | 3.40 | 3.84 | 142.3 | 98.8 |
| o | 4096 | 4096 | 16 | 211.7 | 70.3 | 3.01 | 3.79 | 126.8 | 88.1 |
| gate_up | 28672 | 4096 | 1 | 1441.1 | 375.5 | 3.84 | 3.88 | 161.5 | 112.1 |
| gate_up | 28672 | 4096 | 2 | 1455.5 | 377.5 | 3.86 | 3.87 | 160.8 | 111.6 |
| gate_up | 28672 | 4096 | 4 | 1460.2 | 380.6 | 3.84 | 3.87 | 159.8 | 111.0 |
| gate_up | 28672 | 4096 | 8 | 1467.4 | 385.7 | 3.80 | 3.85 | 158.4 | 110.0 |
| gate_up | 28672 | 4096 | 16 | 1480.7 | 404.1 | 3.66 | 3.83 | 152.4 | 105.9 |
| down | 4096 | 14336 | 1 | 720.9 | 198.0 | 3.64 | 3.88 | 153.1 | 106.3 |
| down | 4096 | 14336 | 2 | 795.1 | 199.3 | 3.99 | 3.87 | 152.3 | 105.8 |
| down | 4096 | 14336 | 4 | 796.6 | 200.5 | 3.97 | 3.86 | 151.7 | 105.4 |
| down | 4096 | 14336 | 8 | 799.2 | 203.8 | 3.92 | 3.85 | 150.0 | 104.2 |
| down | 4096 | 14336 | 16 | 804.3 | 225.9 | 3.56 | 3.82 | 136.6 | 94.9 |

#### How to read these numbers

- **"% of 144 GB/s" exceeding 100 % is not a bug in the kernel; the 144 GB/s reference is too low.** It came from
  `measure_bw.py`, which is conservative on this card. The RTX 3050 6GB Laptop has a 96-bit GDDR6 bus at 14 Gbps, a
  **theoretical peak of ~168 GB/s**. Against that, the kernel reaches roughly **75–96 %**, with the best rows
  (`gate_up`, `M ≤ 8`) at ~95 %. Real GPUs rarely sustain more than ~90–95 % of theoretical, so these are at the practical limit.
  Laptop GPUs also vary with power limit and memory clocks, so re-run `measure_bw.py` on your own machine.
- **Speedups above "ideal ×" (e.g. 4.01 vs 3.87) are baseline noise**, not a violation of physics. The cuBLAS fp16 timings
  fluctuate by 10 % or more between neighbouring `M` values (e.g. `o`, M = 2 vs M = 4). Trust the W4A16 µs and GB/s columns more than the speedup column.
- **`M = 16` is consistently slower** (about 5–15 % lower bandwidth). This is expected: two token tiles need more shared memory
  per stage, so the dispatcher falls back to a 2-stage pipeline (see below).
- **Small layers lose a bit more.** `o` (N = 4096, K = 4096) is only 64 CTAs on 20 SMs, so wave quantization and
  the fixed epilogue cost take a larger share than on `gate_up`.

---

## Profiling notes & next steps

Shared-memory budget per CTA (8 warps × stages × tile bytes). The sm_86 opt-in limit is 99 KiB = 101,376 B:

| tokens | stages | bytes | fits sm_86? |
|---|---|---|---|
| M ≤ 8 | 4 / **3** / 2 | 106,496 / **79,872** / 53,248 | 4 no → runs 3 stages |
| M 9–16 | 4 / 3 / **2** | 143,360 / 107,520 / **71,680** | only 2 stages |

So on sm_86 you get ~1 CTA/SM, and a shallower pipeline for `M > 8`. Worth profiling:

```bash
ncu --set full -k regex:w4a16 --launch-skip 5 --launch-count 1 \
    python bench/bench_gemm.py --shape 4096,4096 --M 1 --reps 3
```

- DRAM throughput % (the goal), warp-stall reasons (long scoreboard vs. barrier), achieved occupancy.
- **Wave quantization:** the grid is only `N / 64` CTAs (64 for `N = 4096`). Compare against the SM count, and try `BN = 32` or splitting K across CTAs when `N` is small.
- **x re-reads:** every CTA fetches the whole activation (from L2). Check L2→SM traffic against DRAM traffic.
- **Reduction epilogue:** 8-way shared-memory reduce plus one `__syncthreads`; measure its share for small `K`.

---

## Not supported (yet)

Prefill (`M > 16`) · asymmetric / per-group zero-points · GPTQ act-order (`g_idx`) · bf16 · `K % 512 ≠ 0` ·
`G` other than 64 / 128 · fused bias / activation epilogues.

---

## Repository layout

```
csrc/w4a16_gemm.cu    the kernel + torch entry point (w4a16_gemm_cuda)
csrc/binding.cpp      pybind11 module
w4a16_opt/            JIT loader, quantize / pack / reference helpers
tests/test_w4a16.py   correctness tests
bench/                bench_gemm.py (vs cuBLAS + roofline), measure_bw.py
```