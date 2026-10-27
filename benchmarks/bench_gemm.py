#!/usr/bin/env python3
"""Compare the CUDA, Triton and Marlin W4A16 GEMM kernels."""
import argparse
import json
import math
import os
import statistics
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version

import torch
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
    ("CUDA W4A16", "cuda", "#ea580c"),
    ("Triton W4A16", "triton", "#16a34a"),
    ("Marlin W4A16", "marlin", "#9333ea"),
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


def load_weights(N, K, G, target_mb, device, marlin_module=None):
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

    marlin_weight_copies = []
    if marlin_module is not None:
        linear = torch.nn.Linear(K, N, bias=False, device=device, dtype=torch.float16)
        linear.weight.data.copy_(triton_kernel.dequantize_w4(triton_weights))
        packed = marlin_module.Layer(K, N, groupsize=G).to(device)
        packed.pack(linear, triton_weights.scales.t().contiguous())
        del linear

        marlin_count = n_copies(
            packed.B.numel() * packed.B.element_size()
            + packed.s.numel() * packed.s.element_size(),
            target_mb,
        )
        marlin_weight_copies.append((packed.B, packed.s, packed.workspace))
        for _ in range(marlin_count - 1):
            marlin_weight_copies.append(
                (
                    packed.B.clone(),
                    packed.s.clone(),
                    torch.zeros_like(packed.workspace),
                )
            )

    return cuda_weights, cuda_scales, triton_weight_copies, marlin_weight_copies


def write_tflops_chart(path, rows, gpu_name):
    layers = list(dict.fromkeys(row["layer"] for row in rows))
    phases = list(dict.fromkeys(row["mode"] for row in rows))
    backends = [backend for backend in BACKENDS if any(f"{backend[1]}_tflops" in row for row in rows)]
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/w4a16-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = 2
    panels = [(mode, layer) for mode in phases for layer in layers]
    panel_rows = math.ceil(len(panels) / columns)
    fig, axes = plt.subplots(panel_rows, columns, figsize=(13, 4.4 * panel_rows), squeeze=False)

    for panel_index, (mode, layer) in enumerate(panels):
        ax = axes[panel_index // columns][panel_index % columns]
        panel_data = [row for row in rows if row["layer"] == layer and row["mode"] == mode]
        batch_sizes = sorted({row["M"] for row in panel_data})[-3:]
        layer_rows = {row["M"]: row for row in panel_data}
        shape = next(row for row in panel_data)
        positions = list(range(len(batch_sizes)))
        bar_width = 0.78 / len(backends)
        offsets = [(index - (len(backends) - 1) / 2) * bar_width for index in range(len(backends))]

        for backend_index, (label, key, color) in enumerate(backends):
            values = [layer_rows[m][f"{key}_tflops"] for m in batch_sizes]
            ax.bar(
                [position + offsets[backend_index] for position in positions],
                values,
                width=bar_width,
                color=color,
                label=label,
            )

        ax.set_title(f'{mode.title()} · {layer}  (N × K = {shape["N"]:,} × {shape["K"]:,})')
        ax.set_ylabel("TFLOP/s")
        ax.set_xlabel("Prefill sequence length (tokens)" if mode == "prefill" else "Decode batch size")
        ax.set_xticks(positions, [f"{m:,}" for m in batch_sizes])
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for panel_index in range(len(panels), panel_rows * columns):
        axes[panel_index // columns][panel_index % columns].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle("W4A16 GEMM performance by phase", fontsize=17, y=0.995)
    fig.text(0.5, 0.965, gpu_name, ha="center")
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(backends), frameon=False)
    fig.tight_layout(rect=(0.035, 0.025, 0.985, 0.89), h_pad=2.0, w_pad=1.4)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    global cuda_w4a16_gemm, triton_kernel

    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", action="append", help="N,K (repeatable); defaults to Llama 3 8B projection shapes")
    parser.add_argument("--phase", choices=("both", "prefill", "decode"), default="both")
    parser.add_argument("--prefill-M", default="128,512,1024,2048,4096,8192",
                        help="comma-separated prefill sequence lengths")
    parser.add_argument("--decode-M", default="1,2,4,8,16,32",
                        help="comma-separated decode batch sizes")
    parser.add_argument("--M", default=None,
                        help="legacy override: use these M values for every selected phase")
    parser.add_argument("--G", type=int, default=128, help="group size (64 or 128)")
    parser.add_argument("--target-mb", type=float, default=128, help="weight working set for each backend")
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--no-marlin", action="store_true", help="benchmark only CUDA and Triton")
    parser.add_argument("--json", default=None, help="result JSON path")
    parser.add_argument("--plot", default=None, help="TFLOP/s chart path")
    parser.add_argument("--md", action="store_true", help="print a Markdown table")
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    json_path = args.json or os.path.join(results_dir, f"gemm-benchmark-{stamp}.json")
    plot_path = args.plot or os.path.splitext(json_path)[0] + "-grouped.png"
    shapes = (
        [(f"{int(n)}x{int(k)}", (int(n), int(k))) for n, k in (value.split(",") for value in args.shape)]
        if args.shape
        else list(LLAMA3_8B.items())
    )
    phases = []
    if args.phase in ("both", "prefill"):
        phases.append(("prefill", args.prefill_M))
    if args.phase in ("both", "decode"):
        phases.append(("decode", args.decode_M))
    if args.M is not None:
        phases = [(mode, args.M) for mode, _ in phases]
    phase_Ms = {
        mode: list(dict.fromkeys(int(value) for value in values.split(",")))
        for mode, values in phases
    }
    if any(not values or any(value < 1 for value in values) for values in phase_Ms.values()):
        raise ValueError("M values must be positive integers")

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch environment with an accessible GPU is required.")
    if args.G not in (64, 128):
        raise ValueError("--G must be 64 or 128")

    from w4a16_opt import w4a16_gemm as cuda_w4a16_gemm
    import w4a16_triton as triton_kernel

    marlin_module = None
    marlin_version = None
    if not args.no_marlin:
        try:
            import marlin as marlin_module
        except ImportError as exc:
            raise RuntimeError(
                "Marlin is enabled by default but is not installed in this Python environment. "
                "Install it with `pip install .` from the Marlin checkout or pass --no-marlin."
            ) from exc
        try:
            marlin_version = version("marlin")
        except PackageNotFoundError:
            marlin_version = "unknown"
        if args.G != 128:
            raise ValueError("Marlin supports group size 128 in this benchmark; use --G 128 or --no-marlin")
        invalid_shapes = [
            f"{name} ({N}x{K})"
            for name, (N, K) in shapes
            if N % 256 != 0 or K % 128 != 0
        ]
        if invalid_shapes:
            raise ValueError(
                "Marlin requires N divisible by 256 and K divisible by 128; unsupported: "
                + ", ".join(invalid_shapes)
            )

    device = "cuda"
    props = torch.cuda.get_device_properties(0)
    gpu_name = props.name
    print(
        f"# {gpu_name}  SMs={props.multi_processor_count}  "
        f"torch={torch.__version__} cuda={torch.version.cuda} triton={triton.__version__}  "
        f"weight working set={args.target_mb:.0f} MiB  "
        f"Marlin={'off' if marlin_module is None else marlin_version}"
    )

    rows = []
    for layer, (N, K) in shapes:
        cuda_weights, cuda_scales, triton_weights, marlin_weights = load_weights(
            N, K, args.G, args.target_mb, device, marlin_module
        )
        for mode, Ms in phase_Ms.items():
            for M in Ms:
                x = torch.randn((M, K), device=device, dtype=torch.float16)
                cuda_calls = [
                    lambda w=w, s=s: cuda_w4a16_gemm(w, x, s, variant=-1)
                    for w, s in zip(cuda_weights, cuda_scales)
                ]
                triton_calls = [
                    lambda pw=pw: triton_kernel.w4a16_gemm(x, pw)
                    for pw in triton_weights
                ]

                timings = {
                    "cuda": graph_time_us(cuda_calls, args.reps),
                    "triton": graph_time_us(triton_calls, args.reps),
                }
                if marlin_weights:
                    output = torch.empty((M, N), device=device, dtype=torch.float16)
                    marlin_calls = [
                        lambda B=B, s=s, workspace=workspace: marlin_module.mul(
                            x, B, output, s, workspace
                        )
                        for B, s, workspace in marlin_weights
                    ]
                    timings["marlin"] = graph_time_us(marlin_calls, args.reps)

                row = {"mode": mode, "layer": layer, "N": N, "K": K, "M": M, "G": args.G}
                for key, (median_us, min_us) in timings.items():
                    row[f"{key}_us"] = median_us
                    row[f"{key}_min_us"] = min_us
                    row[f"{key}_tflops"] = tflops(M, N, K, median_us)
                rows.append(row)
                print(f"finished {mode} {layer} M={M}")
        del cuda_weights, cuda_scales, triton_weights, marlin_weights
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
            "phase_M": phase_Ms,
            "marlin": marlin_module is not None,
            "marlin_version": marlin_version,
            "backends": [
                label for label, key, _ in BACKENDS
                if any(f"{key}_us" in row for row in rows)
            ],
            "timing": "CUDA graph median per operation; W4A16 backends use distinct packed weight copies",
        },
        "python_executable": sys.executable,
        "rows": rows,
        "plot": os.path.basename(plot_path),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
        file.write("\n")
    write_tflops_chart(plot_path, rows, gpu_name)

    keys = [key for _, key, _ in BACKENDS if any(f"{key}_us" in row for row in rows)]
    headers = ["Phase", "Layer", "M"] + [f"{key} us" for key in keys]
    if args.md:
        print("| " + " | ".join(headers) + " |")
        print("|" + "---|" * len(headers))
        for row in rows:
            values = [row["mode"], row["layer"], str(row["M"])]
            values.extend(f'{row[f"{key}_us"]:.2f}' for key in keys)
            print("| " + " | ".join(values) + " |")
    else:
        print("  ".join(f"{header:>20}" for header in headers))
        for row in rows:
            values = [row["mode"], row["layer"], str(row["M"])]
            values.extend(f'{row[f"{key}_us"]:.2f}' for key in keys)
            print("  ".join(f"{value:>20}" for value in values))
    print(f"\nSaved {len(rows)} rows to {json_path}")
    print(f"Saved TFLOP/s graph to {plot_path}")


if __name__ == "__main__":
    main()
