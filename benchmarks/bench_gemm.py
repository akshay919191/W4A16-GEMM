#!/usr/bin/env python3
"""Compare PyTorch fp16 GEMM with the CUDA and Triton W4A16 kernels."""
import argparse
import html
import json
import math
import os
import statistics
import sys
from datetime import datetime

import torch
import torch.nn.functional as F
import triton

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from tests.quant import pack_int4

cuda_w4a16_gemm = None
triton_kernel = None

LLAMA3_8B = {
    "qkv": (6144, 4096),
    "o": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
}
BACKENDS = (
    ("PyTorch fp16", "pytorch_tflops", "#2563eb"),
    ("CUDA W4A16", "cuda_tflops", "#ea580c"),
    ("Triton W4A16", "triton_tflops", "#16a34a"),
)


def graph_time_us(calls, reps):
    for call in calls:
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for call in calls:
            call()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / len(calls))
    return statistics.median(samples), min(samples)


def n_copies(bytes_per_copy, target_mb):
    return max(1, min(64, math.ceil(target_mb * 2**20 / bytes_per_copy)))


def tflops(M, N, K, time_us):
    return 2 * M * N * K / (time_us * 1e6)


def load_weights(N, K, G, target_mb, device):
    w16 = torch.randn((N, K), device=device, dtype=torch.float16) * 0.02
    triton_weights = triton_kernel.quantize_w4(w16, G=G, pack_bk=128)
    q = triton_kernel.unpack_w4_codes(triton_weights)
    cuda_wp = pack_int4(q)
    del q

    cuda_count = n_copies(
        cuda_wp.numel() * cuda_wp.element_size()
        + triton_weights.scales.numel() * triton_weights.scales.element_size(),
        target_mb,
    )
    triton_count = n_copies(
        triton_weights.wp.numel() * triton_weights.wp.element_size()
        + triton_weights.scales.numel() * triton_weights.scales.element_size(),
        target_mb,
    )
    torch_count = n_copies(w16.numel() * w16.element_size(), target_mb)

    cuda_weights = [cuda_wp]
    cuda_scales = [triton_weights.scales]
    for _ in range(cuda_count - 1):
        cuda_weights.append(cuda_wp.clone())
        cuda_scales.append(triton_weights.scales.clone())

    triton_weight_copies = [triton_weights]
    for _ in range(triton_count - 1):
        triton_weight_copies.append(
            triton_kernel.W4Weights(
                triton_weights.wp.clone(),
                triton_weights.scales.clone(),
                N,
                K,
                G,
                triton_weights.pack_bk,
            )
        )

    torch_weights = [w16]
    for _ in range(torch_count - 1):
        torch_weights.append(w16.clone())

    return torch_weights, cuda_weights, cuda_scales, triton_weight_copies


def write_tflops_svg(path, rows, gpu_name):
    layers = []
    for row in rows:
        if row["layer"] not in layers:
            layers.append(row["layer"])
    width = 1080
    panel_height = 250
    top = 110
    height = top + panel_height * len(layers) + 34
    left, right = 100, 1035
    plot_top_offset, plot_height = 56, 150
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#1f2937}.small{font-size:12px}.title{font-size:22px;font-weight:bold}.panel{font-size:15px;font-weight:bold}</style>',
        '<text x="40" y="36" class="title">W4A16 GEMM performance</text>',
        f'<text x="40" y="59" class="small">{html.escape(gpu_name)} · TFLOP/s = 2MNK / elapsed time</text>',
    ]
    legend_x = 40
    for label, _, color in BACKENDS:
        svg.append(f'<line x1="{legend_x}" y1="83" x2="{legend_x + 24}" y2="83" stroke="{color}" stroke-width="3"/>')
        svg.append(f'<circle cx="{legend_x + 12}" cy="83" r="4" fill="{color}"/>')
        svg.append(f'<text x="{legend_x + 31}" y="87" class="small">{html.escape(label)}</text>')
        legend_x += 170

    for panel_index, layer in enumerate(layers):
        panel_y = top + panel_index * panel_height
        group = sorted((row for row in rows if row["layer"] == layer), key=lambda row: row["M"])
        max_value = max(
            (row[key] for row in group for _, key, _ in BACKENDS if row.get(key) is not None),
            default=1.0,
        )
        y_max = max(1.0, max_value * 1.12)
        plot_top = panel_y + plot_top_offset
        plot_bottom = plot_top + plot_height
        svg.append(f'<text x="{left}" y="{panel_y + 20}" class="panel">{html.escape(layer)}  (N×K = {group[0]["N"]}×{group[0]["K"]})</text>')
        svg.append(f'<line x1="{left}" y1="{plot_top}" x2="{left}" y2="{plot_bottom}" stroke="#374151"/>')
        svg.append(f'<line x1="{left}" y1="{plot_bottom}" x2="{right}" y2="{plot_bottom}" stroke="#374151"/>')
        svg.append(f'<text x="24" y="{plot_top + plot_height / 2}" class="small" transform="rotate(-90 24 {plot_top + plot_height / 2})">TFLOP/s</text>')
        for tick in range(5):
            value = y_max * tick / 4
            y = plot_bottom - plot_height * tick / 4
            svg.append(f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
            svg.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.1f}</text>')

        m_values = [row["M"] for row in group]
        min_m, max_m = min(m_values), max(m_values)
        log_min, log_max = math.log2(min_m), math.log2(max_m)
        for row in group:
            if log_max == log_min:
                x = (left + right) / 2
            else:
                x = left + (math.log2(row["M"]) - log_min) / (log_max - log_min) * (right - left)
            svg.append(f'<line x1="{x:.1f}" y1="{plot_bottom}" x2="{x:.1f}" y2="{plot_bottom + 4}" stroke="#374151"/>')
            svg.append(f'<text x="{x:.1f}" y="{plot_bottom + 18}" text-anchor="middle" class="small">{row["M"]}</text>')

        for _, key, color in BACKENDS:
            points = []
            for row in group:
                value = row.get(key)
                if value is None:
                    points.append(None)
                    continue
                if log_max == log_min:
                    x = (left + right) / 2
                else:
                    x = left + (math.log2(row["M"]) - log_min) / (log_max - log_min) * (right - left)
                y = plot_bottom - (value / y_max) * plot_height
                points.append((x, y, value))
            segment = []
            segments = []
            for point in points:
                if point is None:
                    if segment:
                        segments.append(segment)
                        segment = []
                else:
                    segment.append(point)
            if segment:
                segments.append(segment)
            for segment in segments:
                coords = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in segment)
                svg.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.5"/>')
                for x, y, value in segment:
                    svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"><title>{value:.3f} TFLOP/s</title></circle>')
        svg.append(f'<text x="{(left + right) / 2}" y="{plot_bottom + 36}" text-anchor="middle" class="small">Batch size M (log₂ spacing)</text>')

    svg.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        file.write("\n".join(svg) + "\n")


def main():
    global cuda_w4a16_gemm, triton_kernel

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="N,K (repeatable); defaults to Llama 3 8B projection shapes")
    parser.add_argument("--M", default="1,2,4,8,16", help="comma-separated batch sizes")
    parser.add_argument("--G", type=int, default=128, help="group size (64 or 128)")
    parser.add_argument("--target-mb", type=float, default=128, help="weight working set for each backend")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--json", default=None, help="result JSON path")
    parser.add_argument("--plot", default=None, help="TFLOP/s SVG path")
    parser.add_argument("--md", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    json_path = args.json or os.path.join(results_dir, f"three-backend-{stamp}.json")
    plot_path = args.plot or os.path.splitext(json_path)[0] + ".svg"
    shapes = (
        [(f"{int(n)}x{int(k)}", (int(n), int(k))) for n, k in (value.split(",") for value in args.shape)]
        if args.shape
        else list(LLAMA3_8B.items())
    )
    Ms = [int(value) for value in args.M.split(",")]

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch environment with an accessible GPU is required.")
    if args.G not in (64, 128):
        raise ValueError("--G must be 64 or 128")

    from w4a16_opt import w4a16_gemm as cuda_w4a16_gemm
    import w4a16_triton as triton_kernel

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    print(
        f"# {gpu_name}  SMs={props.multi_processor_count}  "
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}  "
        f"weight working set={args.target_mb:.0f} MiB"
    )

    rows = []
    for layer, (N, K) in shapes:
        torch_weights, cuda_weights, cuda_scales, triton_weights = load_weights(
            N, K, args.G, args.target_mb, device
        )
        for M in Ms:
            x = torch.randn((M, K), device=device, dtype=torch.float16)
            torch_calls = [lambda w=w: F.linear(x, w) for w in torch_weights]
            cuda_calls = [
                lambda w=w, s=s: cuda_w4a16_gemm(w, x, s, variant=-1)
                for w, s in zip(cuda_weights, cuda_scales)
            ]
            triton_calls = [
                lambda pw=pw: triton_kernel.w4a16_gemm(x, pw)
                for pw in triton_weights
            ]

            torch_us, torch_min = graph_time_us(torch_calls, args.reps)
            cuda_us, cuda_min = graph_time_us(cuda_calls, args.reps)
            triton_us, triton_min = graph_time_us(triton_calls, args.reps)
            rows.append(
                {
                    "layer": layer,
                    "N": N,
                    "K": K,
                    "M": M,
                    "G": args.G,
                    "pytorch_us": torch_us,
                    "pytorch_min_us": torch_min,
                    "cuda_us": cuda_us,
                    "cuda_min_us": cuda_min,
                    "triton_us": triton_us,
                    "triton_min_us": triton_min,
                    "pytorch_tflops": tflops(M, N, K, torch_us),
                    "cuda_tflops": tflops(M, N, K, cuda_us),
                    "triton_tflops": tflops(M, N, K, triton_us),
                }
            )
            print(f"finished {layer} M={M}")
        del torch_weights, cuda_weights, cuda_scales, triton_weights
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    result = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "gpu": gpu_name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "triton": triton.__version__,
        "parameters": {
            "group_size": args.G,
            "target_weight_mib": args.target_mb,
            "repetitions": args.reps,
            "shapes": [{"name": name, "N": dims[0], "K": dims[1]} for name, dims in shapes],
            "M": Ms,
            "timing": "CUDA graph median; per-operation time with distinct weight copies",
        },
        "rows": rows,
        "plot": os.path.basename(plot_path),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    write_tflops_svg(plot_path, rows, gpu_name)

    headers = ["Layer", "M", "PyTorch TFLOP/s", "CUDA TFLOP/s", "Triton TFLOP/s"]
    if args.md:
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for row in rows:
            print(
                f'| {row["layer"]} | {row["M"]} | {row["pytorch_tflops"]:.2f} '
                f'| {row["cuda_tflops"]:.2f} | {row["triton_tflops"]:.2f} |'
            )
    else:
        print("  ".join(f"{header:>20}" for header in headers))
        for row in rows:
            print(
                f'{row["layer"]:>20}  {row["M"]:>20}  {row["pytorch_tflops"]:>20.2f}  '
                f'{row["cuda_tflops"]:>20.2f}  {row["triton_tflops"]:>20.2f}'
            )
    print(f"\nSaved {len(rows)} rows to {json_path}")
    print(f"Saved TFLOP/s graph to {plot_path}")


if __name__ == "__main__":
    main()
