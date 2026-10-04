import torch
import w4a16_opt   # import torch first, always

N, K = 4096, 4096
dev = "cuda"


dev = "cuda"

def pack_for_mma(q):
    """q: [N, K] ints in [-8, 7] -> [N/16, K/64, 32, 4] int32 in the kernel's fragment order."""
    N, K = q.shape
    assert N % 16 == 0 and K % 64 == 0
    u = (q.to(torch.int64) + 8).reshape(N // 16, 16, K // 64, 4, 16)   # [nb, row, kc, s, kk]
    u = u.permute(0, 2, 3, 1, 4)                                        # [nb, kc, s, row, kk]

    lane = torch.arange(32, device=q.device)
    n = torch.arange(8, device=q.device)
    g, t = (lane >> 2)[:, None], (lane & 3)[:, None]
    p, hi = (n & 3)[None, :], (n >> 2)[None, :]
    row = g + 8 * (p & 1)                      # [32, 8]
    col = 2 * t + 8 * (p >> 1) + hi            # nibble p = low half of reg p, p+4 = high half

    v = u[..., row, col]                       # [nb, kc, s, 32, 8]
    word = (v << (4 * n)).sum(-1)              # [nb, kc, s, 32]
    word = word.permute(0, 1, 3, 2).contiguous()                        # [nb, kc, 32, s]
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32)

def run(M, N=8192, K=8192):
    q = torch.randint(-8, 8, (N, K), device=dev)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    scale = (torch.rand(N, device=dev) * 0.02 + 0.005).half()

    y = w4a16_opt.w4a16_gemm(pack_for_mma(q), x, scale)
    torch.cuda.synchronize()
    ref = (x.float() @ q.float().T) * scale.float()

    err = (y.float() - ref).abs().max().item()
    ok = torch.allclose(y.float(), ref, rtol=1e-2, atol=1e-2)
    print(f"M={M:<3} N={N} K={K}  max_abs_err={err:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    torch.manual_seed(0)
    res = [run(M) for M in (1, 3, 8, 9, 16)]
    res.append(run(16, N=256, K=512))
    print("ALL PASSED" if all(res) else "SOME FAILED")

    