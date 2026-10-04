import torch, w4a16_opt
from tests.edge import pack_for_mma   # or paste the packer

N = K = 8192
q = torch.randint(-8, 8, (N, K), device="cuda")
Wp = pack_for_mma(q)
scale = torch.rand(N, device="cuda").half()

for M in (1, 4, 8, 16):
    x = torch.randn(M, K, device="cuda", dtype=torch.float16)
    for _ in range(10): w4a16_opt.w4a16_gemm(Wp, x, scale)   # warmup
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100): w4a16_opt.w4a16_gemm(Wp, x, scale)
    end.record(); torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 100
    gbs = (N * K / 2) / (ms * 1e-3) / 1e9
    print(f"M={M:<3} {ms:.3f} ms   {gbs:.0f} GB/s weight bandwidth")