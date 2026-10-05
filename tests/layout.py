# numpy-only emulation of the kernel's decode + mma + C-fragment mapping, vs a plain matmul.
import numpy as np
rng = np.random.default_rng(0)

def fragment_map():
    n=[[0]*8 for _ in range(32)]; k=[[0]*8 for _ in range(32)]
    for lane in range(32):
        g,t=lane>>2,lane&3
        for i in range(8):
            p,h=i&3,i>>2
            n[lane][i]=g+8*(p&1); k[lane][i]=2*t+8*(p>>1)+h
    return np.array(n),np.array(k)

def pack(q):
    N,K=q.shape; NB,KC=N//16,K//64
    ni,ki=fragment_map()
    t=q.reshape(NB,16,KC,4,16).transpose(0,2,3,1,4)
    v=t[...,ni,ki].astype(np.uint64)
    w=(v<<(np.arange(8,dtype=np.uint64)*np.uint64(4))).sum(-1).astype(np.uint32)
    return np.ascontiguousarray(w.transpose(0,1,3,2)).reshape(-1)   # [NB,KC,32,4]

def kernel_emulation(Wp,x,scales,N,K,G):
    M=x.shape[0]; KC=K//64; WARPS=8; ITERS=K//(64*WARPS)
    y=np.zeros((M,N),np.float32)
    Wp=Wp.reshape(N//16,KC,32,4)
    for nbl in range(N//64):
      red=np.zeros((WARPS,M,64),np.float32)
      for warp in range(WARPS):
        c=np.zeros((4,M,16),np.float32)   # c[r][token][n16]
        for it in range(ITERS):
          kc=warp*ITERS+it
          for s in range(4):
            for r in range(4):
              nb=nbl*4+r
              A=np.zeros((16,16),np.float32)
              for lane in range(32):
                g,t=lane>>2,lane&3
                q=int(Wp[nb,kc,lane,s])
                for p in range(4):
                  bits=(q>>(4*p))&0x000F000F
                  lo,hi=bits&0xFFFF,bits>>16
                  row=g+8*(p&1); col=2*t+8*(p>>1)
                  sc=scales[(kc//(G//64)), nbl*64+r*16+row]       # s_lo/s_hi pick by p&1
                  A[row,col]=(lo-8)*sc; A[row,col+1]=(hi-8)*sc
              B=x[:, kc*64+s*16: kc*64+s*16+16].T                   # [16k, M]
              c[r]+= (A@B).T
        for r in range(4): red[warp,:,r*16:(r+1)*16]=c[r]
      y[:,nbl*64:(nbl+1)*64]=red.sum(0)
    return y

for (N,K,G,M) in [(64,512,64,3),(128,1024,128,9),(64,1024,64,16)]:
    q=rng.integers(0,16,(N,K),dtype=np.uint8)
    scales=rng.random((K//G,N)).astype(np.float32)*0.05+0.001
    x=rng.standard_normal((M,K)).astype(np.float32)
    Wp=pack(q)
    Wdeq=(q.astype(np.float32)-8)*np.repeat(scales.T,G,axis=1)
    ref=x@Wdeq.T
    got=kernel_emulation(Wp,x,scales,N,K,G)
    print(N,K,G,M,"max rel err",np.abs(got-ref).max()/np.abs(ref).max())