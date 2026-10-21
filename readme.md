# W4A16 GEMM

CUDA extension for multiplying fp16 activations by group-scaled, packed 4-bit weights:

```text
y[M, N] = x[M, K] @ Wᵀ
W[n, k] = (q[n, k] - 8) * scale[k / G, n]
```

The vectorized kernel is decode-only: `M` must be 1–16. It does not implement prefill. It supports fp16 activations and scales, int32 packed weights, `N` divisible by 64, `K` divisible by 512, and group size 64 or 128.

## Environment and build

Use Linux with an NVIDIA GPU, CUDA-enabled PyTorch, and CUDA Toolkit 12.0 or newer (`nvcc` available on `PATH`). PyTorch must match the installed CUDA toolchain. The build selects the visible GPU architecture, or defaults to sm_86 if no GPU is visible.

From the repository root, install test/build tools and build both CUDA extensions in the active Python environment:

```bash
python -m pip install torch pytest ninja
python -m pip install -e . --no-build-isolation
```

The editable build compiles `w4a16_opt` from `src/vectorized/` and `w4a16_ext` from `src/baseline/`. If you already have a compatible CUDA-enabled PyTorch installed, avoid replacing it with a different PyTorch build when installing dependencies.

Run correctness tests from the root:

```bash
python -m pytest -q tests/test_w4a16.py
```

The GEMM correctness cases execute the CUDA extension on an NVIDIA GPU. Separate host-side checks cover packing and layout helpers; they do not exercise the GEMM kernel. GPU cases are skipped when CUDA is unavailable. Prefill-sized batches are tested on GPU and should be rejected by the extension.

## Benchmark

Run the default benchmark from the root:

```bash
python benchmarks/bench_gemm.py --md
```

It compares the W4A16 extension with fp16 `torch.nn.functional.linear` over the Llama 3 8B projection dimensions, with `M` in `{1, 2, 4, 8, 16}` and group size 128. Timings use CUDA graphs. The default cycles through 128 MiB of weights; `--target-mb 1` measures a hot-cache case. Each run saves its rows as JSON under `benchmarks/results/`. Choose a path with `--json path/to/results.json`.

Other shapes and batches can be selected, for example:

```bash
python benchmarks/bench_gemm.py --shape 4096,4096 --shape 8192,4096 --M 1,4,16 --G 64,128 --md
```

## Recorded benchmark

The checked-in run at [`benchmarks/results/bench-20261021-182748.json`](benchmarks/results/bench-20261021-182748.json) contains 20 rows: four projection dimensions and five batch sizes. Values below summarize the rows for each layer; latency is the measured W4A16 range, speedup compares against fp16 `linear`, and bandwidth is estimated from the benchmark's modeled traffic and latency.

| Projection | N × K | W4A16 latency (µs) | Speedup range | Estimated bandwidth (GB/s) |
|---|---:|---:|---:|---:|
| qkv | 6144 × 4096 | 83.7–95.4 | 3.64–4.04× | 139.5–155.3 |
| o | 4096 × 4096 | 58.7–70.0 | 3.02–3.91× | 127.3–147.7 |
| gate_up | 28672 × 4096 | 375.5–403.6 | 3.67–3.85× | 152.6–161.5 |
| down | 4096 × 14336 | 197.9–225.7 | 3.56–3.99× | 136.8–153.2 |

The benchmark's default roofline reference is 144 GB/s, so its reported percentage can exceed 100%. The JSON stores per-row timings and calculated metrics, but does not store the GPU model, software versions, or complete command line. Preserve the benchmark's console header with the JSON when recording an environment-specific result.

## Repository files

- `src/vectorized/kernel.cu` — optimized CUDA kernel.
- `src/baseline/kernel.cu` — baseline CUDA kernel.
- `src/*/extension.cpp` — PyTorch extension bindings.
- `tests/` — correctness and packing tests.
- `benchmarks/` — benchmark scripts and saved JSON results.
