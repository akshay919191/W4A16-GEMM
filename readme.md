# W4A16 GEMM

CUDA extension for multiplying fp16 activations by group-scaled, packed 4-bit weights:

```text
y[M, N] = x[M, K] @ Wᵀ
W[n, k] = (q[n, k] - 8) * scale[k / G, n]
```

The vectorized kernel is decode-only: `M` must be 1–16. It does not implement prefill. It supports fp16 activations and scales, int32 packed weights, `N` divisible by 64, `K` divisible by 512, and group size 64 or 128.

## Environment and build

The reference environment is Ubuntu 22.04 x86_64, Python 3.10.16, PyTorch 2.7.1 with CUDA 12.6, and NVIDIA CUDA Toolkit 12.6.3. It targets sm_86. You need an NVIDIA driver compatible with CUDA 12.6, `nvcc`, and a CUDA-supported C++ compiler. CUDA 12.6 supports GCC 6.x–13.2 on x86_64. The pinned Python packages are in [`requirements-repro.txt`](requirements-repro.txt); `.python-version` records the Python version.

Install CUDA Toolkit 12.6.x on the host first. Then, from the repository root, create an isolated environment and build both extensions:

```bash
pyenv install 3.10.16  # skip if this Python version is already installed
pyenv local 3.10.16
python -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1
python -m pip install -r requirements-repro.txt
export CUDA_HOME=/usr/local/cuda-12.6
export TORCH_CUDA_ARCH_LIST=8.6
nvcc --version
python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
python -m pip install -e . --no-build-isolation --no-deps
```

The editable build compiles `w4a16_opt` from `src/vectorized/` and `w4a16_ext` from `src/baseline/`. Keep the same Python, PyTorch, CUDA Toolkit, compiler, driver, and GPU architecture when comparing results. PyTorch provides the CUDA 12.6 runtime wheel, while the separate Toolkit provides `nvcc` for compiling the extensions.

Run correctness tests from the root:

```bash
python -m pytest -q tests/test_w4a16.py
```

The GEMM correctness cases execute the CUDA extension on an NVIDIA GPU. Separate host-side checks cover packing and layout helpers; they do not exercise the GEMM kernel. GPU cases are skipped when CUDA is unavailable. Prefill-sized batches are tested on GPU and should be rejected by the extension.

## Benchmark

Run the default benchmark from the root and save both its machine-readable rows and console environment header:

```bash
mkdir -p benchmarks/results
python benchmarks/bench_gemm.py --md --json benchmarks/results/current-run.json | tee benchmarks/results/current-run.log
```

It compares the W4A16 extension with fp16 `torch.nn.functional.linear` over the Llama 3 8B projection dimensions, with `M` in `{1, 2, 4, 8, 16}` and group size 128. Timings use CUDA graphs. The default cycles through at least 128 MiB of packed weights (up to 64 copies). Each run saves its rows as JSON under `benchmarks/results/`. Choose a path with `--json path/to/results.json`.

Other shapes and batches can be selected, for example:

```bash
python benchmarks/bench_gemm.py --shape 4096,4096 --shape 8192,4096 --M 1,4,16 --G 64,128 --md
```

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

## Recorded benchmark

The checked-in run at [`benchmarks/results/bench-20261021-182748.json`](benchmarks/results/bench-20261021-182748.json) contains 20 rows: four projection dimensions and five batch sizes. Values below summarize the rows for each layer; latency is the measured W4A16 range, speedup compares against fp16 `linear`, and bandwidth is estimated from the benchmark's modeled traffic and latency.

| Projection | N × K | W4A16 latency (µs) | Speedup range | Implied bandwidth* (GB/s) |
|---|---:|---:|---:|---:|
| qkv | 6144 × 4096 | 83.7–95.4 | 3.64–4.04× | 139.5–155.3 |
| o | 4096 × 4096 | 58.7–70.0 | 3.02–3.91× | 127.3–147.7 |
| gate_up | 28672 × 4096 | 375.5–403.6 | 3.67–3.85× | 152.6–161.5 |
| down | 4096 × 14336 | 197.9–225.7 | 3.56–3.99× | 136.8–153.2 |

*The benchmark calculates this as modeled bytes divided by elapsed time; it is not a hardware DRAM counter. The run used a 144 GB/s reference, also the RTX 3050 6GB Laptop GPU's theoretical bandwidth (96-bit bus at 12 Gb/s). The maximum implied value is 161.5 GB/s, or 112% of that limit. A DRAM-bound kernel cannot sustain more than physical peak, so this points to a discrepancy in effective-bandwidth estimation or timing—not performance above the roofline. Do not treat the estimate or `pct_peak` as measured DRAM bandwidth. Recheck device SKU, timings, and traffic accounting before making roofline claims. The desktop RTX 3050 6GB is a different SKU: NVIDIA lists its 96-bit interface, and MSI specifies 14 Gb/s memory for its desktop 6GB model (168 GB/s).

The checked-in JSON predates this pinned environment and does not include GPU or software metadata. Re-run the benchmark with the commands above to save its GPU and software header alongside the results. See [PyTorch's PyTorch 2.7.1 CUDA 12.6 install command](https://pytorch.org/get-started/previous-versions/) and [NVIDIA CUDA Toolkit 12.6 documentation](https://docs.nvidia.com/cuda/archive/12.6.3/).

## Repository files

- `src/vectorized/kernel.cu` — optimized CUDA kernel.
- `src/baseline/kernel.cu` — baseline CUDA kernel.
- `src/*/extension.cpp` — PyTorch extension bindings.
- `tests/` — correctness and packing tests.
- `benchmarks/` — benchmark scripts and saved JSON results.
