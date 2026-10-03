import torch
import w4a16_ext   # import torch first, always

N, K = 4096, 4096
dev = "cuda"

W  = torch.randint(-2**31, 2**31 - 1, (N, K // 8), dtype=torch.int32, device=dev)
A  = torch.randn(K, dtype=torch.float16, device=dev)
sW = torch.rand(N, dtype=torch.float16, device=dev) * 0.1
zW = torch.randn(N, dtype=torch.float16, device=dev)

out = w4a16_ext.w4a16_gemv(W, A, sW, zW)

Wi  = W.to(torch.int64) & 0xFFFFFFFF
nib = torch.stack([(Wi >> (4 * j)) & 0xF for j in range(8)], dim=-1).reshape(N, K)
nib = torch.where(nib >= 8, nib - 16, nib).float()
ref = ((nib @ A.float()) - zW.float()) * sW.float()

print(W.shape , (N , K // 8) , Wi.shape , out.shape)

print((out.float() - ref).abs().max())
print(torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-1))