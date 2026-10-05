"""Correctness tests for the W4A16 decode GEMM.

    pytest -q tests/                      # everything (GPU tests skip without CUDA)
    pytest -q tests/ -k "not llama"       # skip the big shapes
    compute-sanitizer --tool memcheck python -m pytest -q tests/ -k "shapes or onehot"
"""
import pytest
import torch

from .quant import (dequantize_w4, fragment_map, pack_int4, quantize_w4,
                         ref_gemm, unpack_int4)

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
GS = [64, 128]


def gemm(*a):
    from w4a16_opt import w4a16_gemm
    return w4a16_gemm(*a)


def make_problem(M, N, K, G, seed=0, w_std=0.02):
    gen = torch.Generator().manual_seed(seed)
    W = torch.randn(N, K, generator=gen) * w_std
    x = (torch.randn(M, K, generator=gen)).half().cuda()
    q, scales = quantize_w4(W.cuda(), G)
    return x, q, scales, pack_int4(q)


def assert_close(y, ref, tol=2e-3):
    assert torch.isfinite(y.float()).all(), "NaN/Inf in output"
    rel = (y.float() - ref).abs().max().item() / ref.abs().max().item()
    assert rel < tol, f"max-abs error / max|ref| = {rel:.3e} (tol {tol:.0e})"



def test_fragment_map_is_bijection():
    n_idx, k_idx = fragment_map()
    cells = {(n, k) for rn, rk in zip(n_idx, k_idx) for n, k in zip(rn, rk)}
    assert len(cells) == 256 and all(0 <= n < 16 and 0 <= k < 16 for n, k in cells)


@pytest.mark.parametrize("N,K", [(16, 64), (64, 512), (128, 1024)])
def test_pack_roundtrip(N, K):
    q = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    assert torch.equal(unpack_int4(pack_int4(q), N, K), q)


def test_pack_matches_kernel_decode():
    """Decode words exactly like dequant8 + the mma.m16n8k16 A-fragment layout, from the kernel's side."""
    N, K = 32, 128
    q = torch.randint(0, 16, (N, K), dtype=torch.uint8)
    Wp = pack_int4(q).view(N // 16, K // 64, 32, 4)
    out = torch.full((N, K), -1, dtype=torch.int64)
    for nb in range(N // 16):
        for kc in range(K // 64):
            for lane in range(32):
                g, t = lane >> 2, lane & 3
                for s in range(4):
                    word = int(Wp[nb, kc, lane, s]) & 0xFFFFFFFF
                    for p in range(4):
                        bits = (word >> (4 * p)) & 0x000F000F
                        row = nb * 16 + g + 8 * (p & 1)
                        col = kc * 64 + s * 16 + 2 * t + 8 * (p >> 1)
                        out[row, col] = bits & 0xFFFF
                        out[row, col + 1] = bits >> 16
    assert torch.equal(out, q.long())


@pytest.mark.parametrize("G", GS)
def test_quantization_sanity(G):
    """Not a kernel test: checks the RTN quantizer itself is reasonable."""
    W = torch.randn(256, 1024) * 0.02
    q, s = quantize_w4(W, G)
    Wd = dequantize_w4(q, s, G).float()
    cos = torch.nn.functional.cosine_similarity(W.flatten(), Wd.flatten(), dim=0)
    assert cos > 0.98


# ----------------------------------------------------------------------------- GPU: kernel

@cuda
@pytest.mark.parametrize("G", GS)
@pytest.mark.parametrize("M", [1, 2, 3, 7, 8, 9, 15, 16])
@pytest.mark.parametrize("N,K", [(64, 512), (128, 1024), (192, 1536), (4096, 4096)])
def test_shapes(M, N, K, G):
    x, q, scales, Wp = make_problem(M, N, K, G)
    assert_close(gemm(Wp, x, scales), ref_gemm(x, q, scales, G))


@cuda
@pytest.mark.parametrize("G", [128])
@pytest.mark.parametrize("M", [1, 4, 16])
@pytest.mark.parametrize("N,K", [(6144, 4096), (4096, 4096), (28672, 4096), (4096, 14336)])
def test_llama3_8b_shapes(M, N, K, G):
    x, q, scales, Wp = make_problem(M, N, K, G)
    assert_close(gemm(Wp, x, scales), ref_gemm(x, q, scales, G))


@cuda
@pytest.mark.parametrize("G", GS)
def test_onehot_k_mapping(G):
    """x = e_k  ->  y must equal column k of the dequantized W *exactly*. Hits every warp's K slice."""
    N, K = 128, 1024
    _, q, scales, Wp = make_problem(1, N, K, G)
    Wd = dequantize_w4(q, scales, G)
    ks = [0, 1, 2, 7, 8, 15, 16, 17, 63, 64, 65, 127, 128, 255, 256, 511, 512, 513, 1022, 1023]
    for i in range(0, len(ks), 16):
        chunk = ks[i:i + 16]
        x = torch.zeros(len(chunk), K, dtype=torch.half, device="cuda")
        for m, k in enumerate(chunk):
            x[m, k] = 1
        y = gemm(Wp, x, scales)
        assert torch.equal(y, Wd[:, chunk].t().contiguous()), f"k-mapping wrong for k in {chunk}"


@cuda
@pytest.mark.parametrize("fill", ["zeros", "fifteens", "random"])
def test_nibble_extremes(fill):
    M, N, K, G = 4, 128, 1024, 64
    gen = torch.Generator().manual_seed(1)
    q = {"zeros": torch.zeros(N, K, dtype=torch.uint8),
         "fifteens": torch.full((N, K), 15, dtype=torch.uint8),
         "random": torch.randint(0, 16, (N, K), dtype=torch.uint8, generator=gen)}[fill].cuda()
    scales = (torch.rand(K // G, N, generator=gen) * 0.05 + 0.001).half().cuda()
    x = torch.randn(M, K, generator=gen).half().cuda()
    assert_close(gemm(pack_int4(q), x, scales), ref_gemm(x, q, scales, G))


@cuda
def test_zero_input():
    x, q, scales, Wp = make_problem(5, 128, 1024, 64)
    assert not gemm(Wp, torch.zeros_like(x), scales).any()


@cuda
def test_deterministic():
    x, q, scales, Wp = make_problem(8, 1024, 2048, 128)
    assert torch.equal(gemm(Wp, x, scales), gemm(Wp, x, scales))


@cuda
@pytest.mark.parametrize("M", [1, 12])
def test_cuda_graph_capture(M):
    """The inference engine will call this inside CUDA graphs."""
    x, q, scales, Wp = make_problem(M, 512, 1024, 128)
    gemm(Wp, x, scales)                                   # warm up (loads ext, sets smem attr)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        y = gemm(Wp, x, scales)
    x2 = torch.randn_like(x)
    x.copy_(x2)
    g.replay()
    torch.cuda.synchronize()
    assert_close(y, ref_gemm(x2, q, scales, 128))


@cuda
def test_rejects_bad_inputs():
    def args(**kw):
        p = dict(M=2, N=64, K=512, G=64)
        p.update(kw)
        x, q, s, Wp = make_problem(p["M"], p["N"], p["K"], p["G"])
        return Wp, x, s

    Wp, x, s = args()
    with pytest.raises(RuntimeError, match="fp16"):
        gemm(Wp, x.float(), s)
    with pytest.raises(RuntimeError, match="int32"):
        gemm(Wp.to(torch.int64), x, s)
    with pytest.raises(RuntimeError, match="CUDA"):
        gemm(Wp.cpu(), x.cpu(), s.cpu())
    with pytest.raises(RuntimeError, match="contiguous"):
        gemm(Wp, torch.empty(512, 2, dtype=torch.half, device="cuda").t(), s)
    with pytest.raises(RuntimeError, match="M must be"):
        gemm(*args(M=17))
    with pytest.raises(RuntimeError, match="multiple of 512"):
        gemm(*args(K=256))
    with pytest.raises(RuntimeError, match="multiple of 64"):
        gemm(*args(N=32))
    with pytest.raises(RuntimeError, match="unsupported group size"):
        gemm(*args(G=32))