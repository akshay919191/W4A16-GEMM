"""Quantize / pack / reference helpers that define the kernel's weight format.

Format (see README, "Weight format"): unsigned 4-bit codes q in [0,15], zero-point 8,
    W[n, k] ~= (q[n, k] - 8) * scales[k // G, n]
packed so that each lane's 32-bit word is directly an mma.m16n8k16 A-fragment (8 nibbles).
"""
import torch


def fragment_map():
    """For every lane (0..31) and nibble i (0..7): (n, k) inside one 16x16 (n x k) A tile.

    Nibble i < 4  -> low  half of fragment register a[i]
    Nibble i >= 4 -> high half of fragment register a[i-4]
    a[p] holds row  g + 8*(p&1),  cols 2t + 8*(p>>1) (+1 for the high half),  g=lane/4, t=lane%4.
    """
    n_idx = [[0] * 8 for _ in range(32)]
    k_idx = [[0] * 8 for _ in range(32)]
    for lane in range(32):
        g, t = lane >> 2, lane & 3
        for i in range(8):
            p, h = i & 3, i >> 2
            n_idx[lane][i] = g + 8 * (p & 1)
            k_idx[lane][i] = 2 * t + 8 * (p >> 1) + h
    return n_idx, k_idx


def _tables(device):
    n_idx, k_idx = fragment_map()
    return (torch.tensor(n_idx, device=device, dtype=torch.long),
            torch.tensor(k_idx, device=device, dtype=torch.long))


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """q: uint8 [N, K] with values 0..15  ->  int32 [N*K/8] in kernel order.

    Word index = (((n/16) * (K/64) + k/64) * 32 + lane) * 4 + s,  s = (k % 64) / 16.
    """
    N, K = q.shape
    assert N % 16 == 0 and K % 64 == 0, "N must be a multiple of 16 and K of 64"
    NB, KC = N // 16, K // 64
    n_idx, k_idx = _tables(q.device)
    t = q.contiguous().view(NB, 16, KC, 4, 16).permute(0, 2, 3, 1, 4)    # [NB,KC,4,16n,16k]
    v = t[..., n_idx, k_idx].to(torch.int64)                              # [NB,KC,4,32,8]
    shifts = torch.arange(8, device=q.device, dtype=torch.int64) * 4
    w = (v << shifts).sum(-1)                                             # disjoint bits
    w = torch.where(w >= 2 ** 31, w - 2 ** 32, w).to(torch.int32)         # [NB,KC,4,32]
    return w.transpose(2, 3).contiguous().view(-1)                        # [NB,KC,32,4]


def unpack_int4(Wp: torch.Tensor, N: int, K: int) -> torch.Tensor:
    NB, KC = N // 16, K // 64
    n_idx, k_idx = _tables(Wp.device)
    w = Wp.view(NB, KC, 32, 4).transpose(2, 3).to(torch.int64) & 0xFFFFFFFF   # [NB,KC,4,32]
    shifts = torch.arange(8, device=Wp.device, dtype=torch.int64) * 4
    v = ((w.unsqueeze(-1) >> shifts) & 0xF).to(torch.uint8)                   # [NB,KC,4,32,8]
    t = torch.zeros(NB, KC, 4, 16, 16, dtype=torch.uint8, device=Wp.device)
    t[..., n_idx, k_idx] = v
    return t.permute(0, 3, 1, 2, 4).reshape(N, K)


def quantize_w4(W: torch.Tensor, group_size: int):
    """Symmetric round-to-nearest, per (n, k-group) absmax/7 scale, zero-point 8.

    Returns q uint8 [N,K] (0..15) and scales fp16 [K/G, N] (kernel layout).
    """
    N, K = W.shape
    G = group_size
    assert K % G == 0
    Wg = W.float().view(N, K // G, G)
    s = (Wg.abs().amax(-1, keepdim=True) / 7).clamp(min=1e-4).half().float()   # fp16-representable
    q = (torch.round(Wg / s).clamp(-8, 7) + 8).to(torch.uint8).view(N, K)
    scales = s.squeeze(-1).t().contiguous().half()
    return q, scales


def dequantize_w4(q: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """fp16 [N,K]. Bit-identical to the kernel's dequant: (q-8) is exact in fp16, one fp16 multiply."""
    s = scales.t().repeat_interleave(group_size, dim=1)                        # [N,K] fp16
    return (q.to(torch.float16) - 8) * s


def ref_gemm(x: torch.Tensor, q: torch.Tensor, scales: torch.Tensor, group_size: int) -> torch.Tensor:
    """fp32 reference of y = x @ W^T using the kernel's exact dequantized weights."""
    return x.float() @ dequantize_w4(q, scales, group_size).float().t()