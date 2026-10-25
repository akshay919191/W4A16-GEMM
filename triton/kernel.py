from dataclasses import dataclass
from functools import lru_cache

import torch
import triton
import triton.language as tl

@dataclass
class W4Weights:
    wp: torch.Tensor
    scales: torch.Tensor
    N: int
    K: int
    G: int
    pack_bk: int

def _check(K, G, pack_bk):
    plane = pack_bk // 8
    assert pack_bk in (128, 256, 512), "pack_bk must be 128, 256 or 512"
    assert K % pack_bk == 0, f"K ({K}) must be a multiple of pack_bk ({pack_bk})"
    assert K % G == 0 and G % plane == 0, f"G ({G}) must divide K and be a multiple of PLANE ({plane})"

def _pack_nibbles(q: torch.Tensor, pack_bk: int) -> torch.Tensor:

    N, K = q.shape
    plane = pack_bk // 8
    qq = q.to(torch.int64).reshape(N, K // pack_bk, 8, plane)
    shifts = (4 * torch.arange(8, device=q.device, dtype=torch.int64)).view(1, 1, 8, 1)
    word = (qq << shifts).sum(dim=2)
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32).reshape(N, K // 8).contiguous()

def _unpack_nibbles(wp: torch.Tensor, K: int, pack_bk: int) -> torch.Tensor:

    N = wp.shape[0]
    plane = pack_bk // 8
    w = wp.to(torch.int64).reshape(N, K // pack_bk, 1, plane)
    shifts = (4 * torch.arange(8, device=wp.device, dtype=torch.int64)).view(1, 1, 8, 1)
    return ((w >> shifts) & 0xF).reshape(N, K)

def quantize_w4(W: torch.Tensor, G: int = 128, pack_bk: int = 128) -> W4Weights:

    N, K = W.shape
    _check(K, G, pack_bk)
    Wg = W.float().reshape(N, K // G, G)
    scale = (Wg.abs().amax(dim=-1) / 7.0).to(torch.float16)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = torch.round(Wg / scale.float().unsqueeze(-1)).clamp(-8, 7) + 8
    q = q.reshape(N, K).to(torch.int32)
    return W4Weights(_pack_nibbles(q, pack_bk), scale.t().contiguous(), N, K, G, pack_bk)

def dequantize_w4(pw: W4Weights) -> torch.Tensor:

    nib = _unpack_nibbles(pw.wp, pw.K, pw.pack_bk)
    s = pw.scales.repeat_interleave(pw.G, dim=0).t()
    return (nib - 8).to(torch.float16) * s

def unpack_w4_codes(pw: W4Weights) -> torch.Tensor:
    return _unpack_nibbles(pw.wp, pw.K, pw.pack_bk).to(torch.uint8)

@triton.jit
def _dequant_plane(w, j: tl.constexpr, s):

    nib = (w >> (4 * j)) & 0xF
    return (nib - 8).to(tl.float16) * s[None, :]

def _prune_configs(configs, named_args, **kwargs):

    bucket = kwargs.get("M_BUCKET", named_args.get("M_BUCKET", 256))
    cap = max(16, bucket)
    keep = [c for c in configs if c.kwargs["BLOCK_M"] <= cap]
    return keep or configs[:1]

_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=nw, num_stages=ns)
    for bm, bn, nw, ns in [

        (16, 32, 4, 4), (16, 64, 4, 3), (16, 64, 4, 4), (16, 128, 4, 3), (16, 128, 8, 4),

        (32, 64, 4, 3), (32, 128, 4, 3), (32, 128, 8, 3),

        (64, 64, 4, 3), (64, 128, 4, 2), (64, 128, 8, 3),
        (128, 64, 4, 2), (128, 64, 8, 3), (128, 128, 8, 2),
    ]
]

@triton.autotune(
    configs=_CONFIGS,
    key=["M_BUCKET", "N", "K", "SPLIT_K"],
    prune_configs_by={"early_config_prune": _prune_configs},
)
@triton.jit
def _w4a16_kernel(
    x_ptr, wp_ptr, s_ptr, out_ptr,
    M, N, K,
    G: tl.constexpr, PACK_BK: tl.constexpr, SPLIT_K: tl.constexpr, M_BUCKET: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    PLANE: tl.constexpr = PACK_BK // 8

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_k = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_p = tl.arange(0, PLANE)
    m_mask = offs_m < M
    n_mask = offs_n < N

    KB = K // PACK_BK
    kb_per = KB // SPLIT_K
    kb_start = pid_k * kb_per

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for kb in range(kb_start, kb_start + kb_per):

        w_ptrs = wp_ptr + offs_n[None, :] * (K // 8) + (kb * PLANE + offs_p)[:, None]
        w = tl.load(w_ptrs, mask=n_mask[None, :], other=0)

        for j in tl.static_range(8):
            k0 = kb * PACK_BK + j * PLANE
            s = tl.load(s_ptr + (k0 // G) * N + offs_n, mask=n_mask, other=0.0)
            wd = _dequant_plane(w, j, s)
            xk = tl.load(x_ptr + offs_m[:, None] * K + (k0 + offs_p)[None, :],
                         mask=m_mask[:, None], other=0.0)
            acc = tl.dot(xk, wd, acc)

    out_mask = m_mask[:, None] & n_mask[None, :]
    if SPLIT_K == 1:
        tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float16), mask=out_mask)
    else:
        tl.store(out_ptr + pid_k * M * N + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)

@triton.jit
def _splitk_reduce(ws_ptr, y_ptr, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(ws_ptr + s * MN + offs, mask=mask, other=0.0)
    tl.store(y_ptr + offs, acc.to(tl.float16), mask=mask)

@lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count

def _pick_split_k(M: int, N: int, KB: int, device_index: int) -> int:

    target = 4 * _num_sms(device_index)
    tiles = triton.cdiv(N, 64) * triton.cdiv(M, 64)
    cands = [d for d in range(1, 17) if KB % d == 0]
    for d in cands:
        if tiles * d >= target:
            return d
    return cands[-1]

def w4a16_gemm(x: torch.Tensor, pw: W4Weights, split_k: int = None) -> torch.Tensor:
    assert x.is_cuda and x.dtype == torch.float16 and x.is_contiguous() and x.dim() == 2
    M, K = x.shape
    assert K == pw.K, f"x has K={K}, weights have K={pw.K}"
    N = pw.N
    KB = K // pw.pack_bk
    split = split_k or _pick_split_k(M, N, KB, x.device.index or 0)
    assert KB % split == 0, f"split_k={split} must divide K // pack_bk = {KB}"

    y = torch.empty((M, N), dtype=torch.float16, device=x.device)
    out = y if split == 1 else torch.empty((split, M, N), dtype=torch.float32, device=x.device)
    m_bucket = min(256, triton.next_power_of_2(M))

    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(M, meta["BLOCK_M"]), split)
    _w4a16_kernel[grid](
        x, pw.wp, pw.scales, out, M, N, K,
        pw.G, pw.pack_bk, split, m_bucket,
    )
    if split > 1:
        BLOCK = 1024
        _splitk_reduce[(triton.cdiv(M * N, BLOCK),)](out, y, M * N, SPLIT_K=split, BLOCK=BLOCK)
    return y
