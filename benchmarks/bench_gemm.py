#!/usr/bin/env python3
"""Compare PyTorch fp16 GEMM with the CUDA and Triton W4A16 kernels."""
import argparse
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


def write_tflops_chart(path, rows, gpu_name):
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    batch_sizes = sorted({row["M"] for row in rows})[-3:]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/w4a16-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#138b7d", "#2f6db3", "#df812d"]
    columns = 2
    panel_rows = math.ceil(len(layers) / columns)
    fig, axes = plt.subplots(panel_rows, columns, figsize=(13, 4.4 * panel_rows), squeeze=False)

    for panel_index, layer in enumerate(layers):
        ax = axes[panel_index // columns][panel_index % columns]
        layer_rows = {row["M"]: row for row in rows if row["layer"] == layer}
        shape = next(row for row in rows if row["layer"] == layer)
        positions = list(range(len(batch_sizes)))
        bar_width = 0.23
        offsets = [(index - (len(colors) - 1) / 2) * bar_width for index in range(len(colors))]

        for backend_index, ((label, key, _), color) in enumerate(zip(BACKENDS, colors)):
            values = [layer_rows[m][key] for m in batch_sizes]
            ax.bar(
                [position + offsets[backend_index] for position in positions],
                values,
                width=bar_width,
                color=color,
                label=label,
            )

        ax.set_title(f'{layer}  (N × K = {shape["N"]:,} × {shape["K"]:,})')
        ax.set_ylabel("TFLOP/s")
        ax.set_xlabel("Batch size M")
        ax.set_xticks(positions, [f"{m:,}" for m in batch_sizes])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for panel_index in range(len(layers), panel_rows * columns):
        axes[panel_index // columns][panel_index % columns].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("W4A16 GEMM performance", fontsize=17, y=0.995)
    fig.text(0.5, 0.945, f"{gpu_name} · largest batch sizes M={', '.join(map(str, batch_sizes))}", ha="center")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.895), ncol=3, frameon=False)
    fig.tight_layout(rect=(0.035, 0.025, 0.985, 0.85), h_pad=2.0, w_pad=1.4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


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
    plot_path = args.plot or os.path.splitext(json_path)[0] + "-grouped.png"
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
        "python_executable": sys.executable,
        "rows": rows,
        "plot": os.path.basename(plot_path),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    write_tflops_chart(plot_path, rows, gpu_name)

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
