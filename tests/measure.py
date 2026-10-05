#!/usr/bin/env python3
"""Measure achievable DRAM bandwidth (use this, not the spec sheet, as the roofline denominator).

    python bench/measure_bw.py            # then:  python bench/bench_gemm.py --peak-bw <GB/s printed>
"""
import argparse

import torch


def best_ms(fn, iters=30):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); b.synchronize()
        best = min(best, a.elapsed_time(b))
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mb", type=int, default=256, help="buffer size in MiB (must be >> L2)")
    n = ap.parse_args().mb * 2 ** 20
    src = torch.empty(n, dtype=torch.uint8, device="cuda").random_()
    dst = torch.empty_like(src)
    print(torch.cuda.get_device_name())
    print(f"copy (read+write) : {2 * n / best_ms(lambda: dst.copy_(src)) / 1e6:8.1f} GB/s")
    print(f"sum  (read only)  : {n / best_ms(lambda: src.view(torch.int32).sum()) / 1e6:8.1f} GB/s")


if __name__ == "__main__":
    main()