import torch
import w4a16_opt

dev = "cuda"

def pack_for_mma(q):
    N, K = q.shape
    u = (q.to(torch.int64) + 8).reshape(N // 16, 16, K // 64, 4, 16)
    u = u.permute(0, 2, 3, 1, 4)
    lane = torch.arange(32, device=q.device)
    n = torch.arange(8, device=q.device)
    g, t = (lane >> 2)[:, None], (lane & 3)[:, None]
    p, hi = (n & 3)[None, :], (n >> 2)[None, :]
    row = g + 8 * (p & 1)
    col = 2 * t + 8 * (p >> 1) + hi
    v = u[..., row, col]
    word = (v << (4 * n)).sum(-1)
    word = word.permute(0, 1, 3, 2).contiguous()
    word = torch.where(word >= 2**31, word - 2**32, word)
    return word.to(torch.int32)

def run(M, G, N=4096, K=4096):
    q = torch.randint(-8, 8, (N, K), device=dev)
    x = torch.randn(M, K, device=dev, dtype=torch.float16)
    scales = (torch.rand(K // G, N, device=dev) * 0.02 + 0.005).half()

    y = w4a16_opt.w4a16_gemm(pack_for_mma(q), x, scales)
    torch.cuda.synchronize()

    s_full = scales.float().repeat_interleave(G, dim=0)
    w_deq = q.float().T * s_full
    ref = x.float() @ w_deq

    err = (y.float() - ref).abs().max().item()
    ok = torch.allclose(y.float(), ref, rtol=2e-2, atol=5e-2)
    print(f"G={G:<4} M={M:<3} N={N} K={K}  max_abs_err={err:.4f}  {'PASS' if ok else 'FAIL'}")
    return ok

if __name__ == "__main__":
    torch.manual_seed(0)
    res = []
    for G in (64, 128):
        for M in (1, 3, 8, 9, 16):
            res.append(run(M, G))
        res.append(run(16, G, N=256, K=1024))
    print("ALL PASSED" if all(res) else "SOME FAILED")