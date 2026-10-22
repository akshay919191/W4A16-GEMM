# W4A16 GEMM

CUDA extension for multiplying fp16 activations by group-scaled, packed 4-bit weights:

```text
y[M, N] = x[M, K] @ Wᵀ
W[n, k] = (q[n, k] - 8) * scale[k / G, n]
```

One entry point, `w4a16_gemm(Wp, x, scales, variant=-1)`, with two kernels behind it:

| Path | Used when | Shape contract |
|---|---|---|
| **decode** (`decode.cuh`) | `M` in 1–16 **and** `N % 64 == 0` **and** `K % 512 == 0` | unchanged from before |
| **prefill** (`prefill.cuh`) | everything else (any `M >= 1`) | `N % 16 == 0`, `K % 64 == 0` |

Both take fp16 activations and scales, int32 packed weights (the same format, no repacking), and group size 64 or 128 (`K / groups` must be exactly 64 or 128). `variant` is for testing and tuning: `-1` auto, `0` decode kernel, `1..4` force a prefill tile config (`BM` = 16 / 32 / 64 / 128).

The decode algorithm and shape-specific launch configurations are preserved in `decode.cuh`; the new prefill path is selected on the host for batches and dimensions outside the decode contract.

### How prefill works

The packed weight word each lane holds is already an `mma.m16n8k16` A fragment. Prefill swaps the operand roles (tokens become the mma M side, weights the N side). The same eight nibbles then form the B fragments of two n8 tiles, so **the weight format does not change** and the output fragment is already `y[token, n]` (no transpose, no smem reduction). Each CTA computes a `BM x BN` tile over a 3–4 stage `cp.async` pipeline with one `__syncthreads()` per 64-wide k-step; weights are dequantized in registers (one fp16 subtract + one fp16 multiply, bit-identical to the reference). `K` is accumulated sequentially in fp32, so **all four prefill configs produce bit-identical output**, and a row's result does not depend on how many other rows are in the batch (as long as both batches use the prefill path; decode vs prefill differ only by fp32 summation order).

| cfg | tile (BM x BN) | warps | intended range |
|---|---|---|---|
| 1 | 16 x 64 | 1 x 4 | M <= 16 on shapes the decode kernel cannot take (e.g. `K = 11008`) |
| 2 | 32 x 128 | 1 x 8 | M ~ 17–48 |
| 3 | 64 x 128 | 2 x 4 | M ~ 48–128 |
| 4 | 128 x 128 | 2 x 4 | large M |

The host picks the config from a small cost model (tile work, waves over the SM count, occupancy queried once per device). Its constants are **estimates that have not been tuned on hardware**; use `--variant` in the benchmark to check the choice on your GPU.

## Environment and build

The tested reference setup is Ubuntu 24.04 x86_64, Python 3.10.22, PyTorch 2.7.1+cu118, CUDA Toolkit 11.8.89, GCC/G++ 11.5, and sm_86. The GPU used for the results below is an NVIDIA GeForce RTX 3050 6GB Laptop GPU with driver 595.104.02. `environment.yml` and [`requirements-repro.txt`](requirements-repro.txt) pin the Conda and Python packages; `.python-version` records Python.

Install CUDA Toolkit 11.8 and GCC/G++ 11 on the host. From the repository root, create the Python environment and build both extensions:

```bash
conda env create -f environment.yml
conda activate w4a16  # or use your existing akku environment
python -m pip install -r requirements-repro.txt
export CUDA_HOME=/usr/local/cuda-11.8
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11
export TORCH_CUDA_ARCH_LIST=8.6
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install -e . --no-build-isolation --no-deps
```

The editable build compiles `w4a16_opt` from `src/vectorized/` and `w4a16_ext` from `src/baseline/`. This setup was rebuilt successfully with CUDA 11.8.89 and GCC/G++ 11.5. Keep the same Python, PyTorch, toolkit, compiler, driver, and GPU architecture when comparing performance.

Run correctness tests from the root:

```bash
python -m pytest -q tests/test_w4a16.py
python tests/layout_prefill.py
```

The GEMM correctness cases execute the CUDA extension on an NVIDIA GPU. Separate host-side checks cover packing and layout helpers; they do not exercise the GEMM kernel. Decode and prefill GEMM cases run on the CUDA GPU and are skipped when CUDA is unavailable. The prefill tests cover ragged batches, output and K tails, both group sizes, all forced tile configs, CUDA graph capture, and comparison with the reference GEMM.

Validation on the RTX 3050: both extensions rebuilt successfully, and `conda run -n akku python -m pytest -q tests/test_w4a16.py` reported 420 passed and 12 skipped. The 12 skipped parameter cases use a K dimension that is not divisible by their requested group size. The numpy-only prefill layout emulator and all-config CUDA compile check are also included; see below for their results.

## Benchmark

Run the default benchmark from the root and save both its machine-readable rows and console environment header:

```bash
mkdir -p benchmarks/results
python benchmarks/bench_gemm.py --peak-bw 168 --md --json benchmarks/results/current-run.json | tee benchmarks/results/current-run.log
```

It compares the W4A16 extension with fp16 `torch.nn.functional.linear` over the Llama 3 8B projection dimensions, with `M` in `{1, 2, 4, 8, 16}` and group size 128. Timings use CUDA graphs. The default cycles through at least 128 MiB of packed weights (up to 64 copies). Each run saves its rows as JSON under `benchmarks/results/`. Choose a path with `--json path/to/results.json`.

`--peak-bw` sets the denominator for the reported percentage and ideal-time estimate. Use 168 for the RTX 3050 tested here; use the appropriate value for another GPU.

Other shapes and batches can be selected, for example:

```bash
python benchmarks/bench_gemm.py --shape 4096,4096 --shape 8192,4096 --M 1,4,16 --G 64,128 --md
```

### Prefill benchmark

```bash
python benchmarks/bench_gemm.py --prefill --md --json benchmarks/results/prefill-run.json
python benchmarks/bench_gemm.py --shape 4096,4096 --M 17,48,100,256 --variant 2 --md   # compare cfgs: --variant 1..4
```

Prefill is mostly compute-bound, so read the TFLOP/s columns and the speedup against fp16 `linear`; the GB/s and `% roofline` columns model decode traffic and are not meaningful there.

| M | fp16 linear (µs) | W4A16 auto (µs) | speedup | W4A16 TFLOP/s |
|---:|---:|---:|---:|---:|
| 17 | 240.6 | 95.2 | 2.53× | 6.0 |
| 64 | 309.2 | 198.7 | 1.56× | 10.8 |
| 256 | 627.7 | 684.0 | 0.92× |

Source: [`benchmarks/results/prefill-auto-rtx3050.json`](benchmarks/results/prefill-auto-rtx3050.json). This run shows that the measured advantage shrinks with batch size and W4A16 is slower at M=256. The GB/s/% roofline fields are modeled using decode-style traffic and should not be used to infer prefill DRAM bandwidth.

Forced prefill tile latency (µs) for the same shape and batch sizes:

| M | cfg 1 (BM=16) | cfg 2 (BM=32) | cfg 3 (BM=64) | cfg 4 (BM=128) | auto |
|---:|---:|---:|---:|---:|---:|
| 17 | 97.3 | 96.0 | 194.4 | 365.6 | 95.2 |
| 64 | 184.3 | **181.2** | 198.6 | 366.6 | 198.7 |
| 256 | 744.3 | **562.2** | 683.0 | 738.1 | 684.0 |

The results show the current auto cost model misses the faster cfg 2 at M=64 and M=256 on this GPU. The per-config measurements are saved as [`cfg1`](benchmarks/results/prefill-cfg1-rtx3050.json), [`cfg2`](benchmarks/results/prefill-cfg2-rtx3050.json), [`cfg3`](benchmarks/results/prefill-cfg3-rtx3050.json), and [`cfg4`](benchmarks/results/prefill-cfg4-rtx3050.json) JSON files. This is evidence to tune the model for this device; it is not yet a general rule for other shapes or GPUs. All runs use a 1 MiB weight working set and 50 repetitions. Treat them as local reference values, not cross-GPU performance claims.

### Diagnostic runs

Compare output-grid sizes while keeping the weight element count constant (16,777,216 elements in each case). This helps check whether a small number of CTAs limits the smaller-N cases:

```bash
python benchmarks/bench_gemm.py --shape 2048,8192 --shape 4096,4096 --shape 8192,2048 --M 8 --G 128 --target-mb 128 --reps 50 --md
```

Compare one packed-weight copy with a larger weight working set for a large layer:

```bash
python benchmarks/bench_gemm.py --shape 28672,4096 --M 8 --target-mb 1 --reps 50 --md
python benchmarks/bench_gemm.py --shape 28672,4096 --M 8 --target-mb 128 --reps 50 --md
```

For this shape, even one packed-weight copy is about 56 MiB, far larger than the laptop GPU's 2 MiB L2 cache. So `--target-mb 1` does **not** make it a hot-cache run; it only reduces the number of distinct copies from three to one. Use hardware counters to determine DRAM traffic. These timing comparisons can show working-set and grid-size effects, but they cannot establish measured bandwidth by themselves.

### Diagnostic results

- For `N=28672, K=4096, M=8`, one weight copy measured 387.1 µs; three copies measured 386.7 µs. Since even one copy is much larger than L2, these runs do not indicate cache residency. The per-call time is essentially unchanged as the copied working set grows from about 56 MiB to 168 MiB.
- With the weight element count held constant, the 32-, 64-, and 128-CTA cases measured 59.3, 63.2, and 58.0 µs. This is not a monotonic improvement with CTA count. K changes between these shapes too, so the sweep does not isolate grid size as the cause.
- Nsight Compute could not collect counters on this machine: it returned `ERR_NVGPUCTRPERM`, and administrator access was unavailable. There is no counter-based occupancy, DRAM throughput, or warp-stall result yet.

## Recorded benchmark

The fresh full run at [`benchmarks/results/baseline-rtx3050-168gbps.json`](benchmarks/results/baseline-rtx3050-168gbps.json) contains 20 rows: four projection dimensions and five batch sizes. Values below summarize that run; latency is the measured W4A16 range, speedup compares against fp16 `linear`, and bandwidth is estimated from the benchmark's modeled traffic and latency.

| Projection | N × K | W4A16 latency (µs) | Speedup range | Implied bandwidth* (GB/s) | of 168 GB/s |
|---|---:|---:|---:|---:|
| qkv | 6144 × 4096 | 83.6–95.5 | 3.64–3.98× | 139.3–155.5 | 82.9–92.6% |
| o | 4096 × 4096 | 58.6–70.0 | 3.02–3.91× | 127.3–147.8 | 75.8–88.0% |
| gate_up | 28672 × 4096 | 375.4–403.6 | 3.67–3.84× | 152.6–161.5 | 90.8–96.1% |
| down | 4096 × 14336 | 198.2–225.9 | 3.56–4.00× | 136.6–153.0 | 81.3–91.1% |

*The benchmark's bandwidth is modeled bytes divided by elapsed time, not a hardware DRAM counter. `nvidia-smi` on this laptop reports a maximum memory clock of 7001 MHz. With its 96-bit memory interface and GDDR6's double data rate, that gives about 168 GB/s theoretical peak. The highest modeled value here, 161.5 GB/s, is about 96% of peak. The older checked-in JSON used `--peak-bw 144`, so its `pct_peak` field is relative to 144 GB/s, not the theoretical peak; the fresh run uses `--peak-bw 168`.

The benchmark prints the GPU, CUDA runtime, PyTorch version, and device details before the table. The fresh JSON stores per-row results; its environment is recorded above. The PyTorch/CUDA wheel combination follows [PyTorch's official installation instructions](https://pytorch.org/get-started/previous-versions/).

## Repository files

- `src/vectorized/kernel.cu` — host-side validation, dispatch and tile-config selection.
- `src/vectorized/decode.cuh` — decode kernel (M <= 16), unchanged.
- `src/vectorized/prefill.cuh` — prefill kernel (any M).
- `src/vectorized/common.cuh` — shared PTX helpers and the dequant routine.
- `tools/compile_check.cu` — torch-free instantiation of every kernel config.
- `src/baseline/kernel.cu` — baseline CUDA kernel.
- `src/*/extension.cpp` — PyTorch extension bindings.
- `tests/` — correctness and packing tests.
- `benchmarks/` — benchmark scripts and saved JSON results.
