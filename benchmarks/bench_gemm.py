#!/usr/bin/env python3
"""Benchmark the W4A16 decode GEMM against fp16 cuBLAS and the memory roofline.

Method: each timed unit is a CUDA graph that launches the op over several *distinct* weight copies
(total >= --target-mb) back-to-back, so weights always stream from DRAM instead of L2, and there is no
Python/launch overhead in the measurement. Reported time = graph time / launches (median of --reps).
Weights are random bits: runtime does not depend on values; correctness lives in tests/.

    python bench/bench_gemm.py                               # Llama-3-8B layers, M in 1,2,4,8,16
    python bench/bench_gemm.py --peak-bw 130 --md > results.md
    python bench/bench_gemm.py --shape 4096,4096 --M 1 --G 128 --target-mb 1   # hot-L2 (1 copy)
"""
import argparse
import json
import math
import os
import statistics
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from w4a16_opt import w4a16_gemm  # noqa: E402

LLAMA3_8B = {"qkv": (6144, 4096), "o": (4096, 4096), "gate_up": (28672, 4096), "down": (4096, 14336)}


def graph_time_us(launches, reps):
    for f in launches:                      # eager warm-up (JIT load, smem attribute, cuBLAS heuristics)
        f()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for f in launches:
            f()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); g.replay(); b.record(); b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / len(launches))
    return statistics.median(ts), min(ts)


def n_copies(bytes_per_copy, target_mb):
    return max(1, min(64, math.ceil(target_mb * 2 ** 20 / bytes_per_copy)))


def traffic_bytes(M, N, K, G):
    return N * K // 2 + (K // G) * N * 2 + M * K * 2 + M * N * 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", action="append", help="N,K (repeatable). Default: Llama-3-8B layers")
    ap.add_argument("--M", default="1,2,4,8,16")
    ap.add_argument("--G", default="128")
    ap.add_argument("--peak-bw", type=float, default=144.0, help="GB/s roofline (measure with measure_bw.py)")
    ap.add_argument("--target-mb", type=float, default=128, help="distinct weight bytes to cycle through")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--md", action="store_true", help="print a markdown table")
    ap.add_argument("--json", help="also write rows to this file")
    a = ap.parse_args()

    shapes = ([(f"{n}x{k}", tuple(map(int, n_k.split(",")))) for n_k in a.shape for n, k in [n_k.split(",")]]
              if a.shape else list(LLAMA3_8B.items()))
    Ms, Gs = [int(v) for v in a.M.split(",")], [int(v) for v in a.G.split(",")]
    dev = "cuda"
    p = torch.cuda.get_device_properties(0)
    print(f"# {p.name}  sm_{p.major}{p.minor}  SMs={p.multi_processor_count}  "
          f"L2={getattr(p, 'L2_cache_size', 0) / 2**20:.0f} MiB  roofline={a.peak_bw} GB/s  "
          f"torch={torch.__version__} cuda={torch.version.cuda}  weights cycled: {a.target_mb:.0f} MiB")

    rows = []
    for name, (N, K) in shapes:
        W16 = torch.randn(N, K, device=dev, dtype=torch.half) * 0.02
        for G in Gs:
            cp = n_copies(N * K // 2, a.target_mb)
            Wp = [torch.randint(-2 ** 31, 2 ** 31 - 1, (N * K // 8,), dtype=torch.int32, device=dev)
                  for _ in range(cp)]
            sc = (torch.rand(K // G, N, device=dev) * 0.01 + 0.001).half()
            for M in Ms:
                x = torch.randn(M, K, device=dev, dtype=torch.half)
                try:
                    t4, t4min = graph_time_us([(lambda w=w: w4a16_gemm(w, x, sc)) for w in Wp], a.reps)
                except RuntimeError as e:
                    print(f"# skip {name} M={M} G={G}: {str(e).splitlines()[0]}")
                    continue
                t16, _ = graph_time_us([lambda: F.linear(x, W16)] * n_copies(N * K * 2, a.target_mb), a.reps) \
                    if cp else (float("nan"), 0)
                by = traffic_bytes(M, N, K, G)
                gbs = by / (t4 * 1e-6) / 1e9
                rows.append(dict(layer=name, N=N, K=K, M=M, G=G, fp16_us=t16, w4_us=t4, w4_min_us=t4min,
                                 speedup=t16 / t4, gbps=gbs, pct_peak=100 * gbs / a.peak_bw,
                                 ideal_us=by / (a.peak_bw * 1e9) * 1e6,
                                 ideal_speedup=(N * K * 2 + M * K * 2 + M * N * 2) / by))
            del Wp
        del W16
        torch.cuda.empty_cache()

    cols = ["layer", "N", "K", "G", "M", "fp16_us", "w4_us", "speedup", "ideal_speedup", "gbps", "pct_peak"]
    hdr = ["layer", "N", "K", "G", "M", "fp16 µs", "W4A16 µs", "speedup", "ideal ×", "GB/s", "% roofline"]
    fmt = lambda c, v: (f"{v:.1f}" if c in ("fp16_us", "w4_us", "gbps", "pct_peak") else
                        f"{v:.2f}" if c in ("speedup", "ideal_speedup") else str(v))
    if a.md:
        print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
        for r in rows:
            print("| " + " | ".join(fmt(c, r[c]) for c in cols) + " |")
    else:
        print("  ".join(f"{h:>10}" for h in hdr))
        for r in rows:
            print("  ".join(f"{fmt(c, r[c]):>10}" for c in cols))
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1)


if __name__ == "__main__":
    main()