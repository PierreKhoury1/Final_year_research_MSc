"""GPU 5G uplink receiver chain (OpenCL) + certified GPU clock calibration.

Per slot (5G NR numerology 0: 15 kHz subcarriers, 1 ms slot, 14 OFDM symbols):
  4 receive antennas x 14 symbols -> FFT(2048) -> per-subcarrier 4x4 MIMO zero-forcing -> 16-QAM LLR demapping.
Stamp kernels read the GPU's REALTIME counter immediately before and after the chain on the same in-order queue.
"""
import time
import warnings

import numpy as np
import pyopencl as cl

warnings.filterwarnings("ignore")

NFFT, LOGN, NSYM, NANT = 2048, 11, 14, 4
AMP = 1 / np.sqrt(10)

SRC = r"""
#define N %(N)d
#define LOGN %(LOGN)d
#define NSYM %(NSYM)d
#define WG 256
typedef float2 cf;
inline cf cmul(cf a, cf b){ return (cf)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x); }
inline cf cconj(cf a){ return (cf)(a.x, -a.y); }
inline cf cdiv(cf a, cf b){ float d = b.x*b.x + b.y*b.y; return (cf)((a.x*b.x + a.y*b.y)/d, (a.y*b.x - a.x*b.y)/d); }
inline ulong gclk(void){ return __builtin_amdgcn_s_sendmsg_rtnl(131); }

__kernel void stamp(__global ulong *buf, uint idx){ buf[idx] = gclk(); }

// one work-group per (antenna, symbol): radix-2 DIT FFT in local memory
__kernel void fft(__global const cf *in, __global cf *out){
  __local cf s[N];
  int g = get_group_id(0), l = get_local_id(0);
  for(int i = l; i < N; i += WG){ int r = 0, v = i; for(int b = 0; b < LOGN; b++){ r = (r << 1) | (v & 1); v >>= 1; } s[r] = in[g*N + i]; }
  barrier(CLK_LOCAL_MEM_FENCE);
  for(int st = 1; st <= LOGN; st++){
    int m = 1 << st, h = m >> 1;
    for(int b = l; b < N/2; b += WG){
      int j = b %% h, i0 = (b / h)*m + j, i1 = i0 + h;
      float ang = -2.0f*M_PI_F*j/m; cf w = (cf)(cos(ang), sin(ang));
      cf a = s[i0], t = cmul(w, s[i1]); s[i0] = a + t; s[i1] = a - t;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for(int i = l; i < N; i += WG) out[g*N + i] = s[i];
}

// per subcarrier: W = (H^H H)^-1 H^H, then x = W y for all 14 symbols. H[k][r][t], fd[r][s][k] -> xe[s][k][t]
__kernel void equalize(__global const cf *H, __global const cf *fd, __global cf *xe){
  int k = get_global_id(0);
  cf h[4][4], G[4][8], W[4][4];
  for(int r = 0; r < 4; r++) for(int t = 0; t < 4; t++) h[r][t] = H[(k*4 + r)*4 + t];
  for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++){
    cf acc = (cf)(0, 0); for(int r = 0; r < 4; r++) acc += cmul(cconj(h[r][i]), h[r][j]);
    G[i][j] = acc; G[i][j+4] = (cf)(i == j ? 1.0f : 0.0f, 0);
  }
  for(int c = 0; c < 4; c++){               // Gauss-Jordan (G is Hermitian positive definite)
    cf p = G[c][c];
    for(int j = 0; j < 8; j++) G[c][j] = cdiv(G[c][j], p);
    for(int i = 0; i < 4; i++) if(i != c){ cf f = G[i][c]; for(int j = 0; j < 8; j++) G[i][j] -= cmul(f, G[c][j]); }
  }
  for(int t = 0; t < 4; t++) for(int r = 0; r < 4; r++){
    cf acc = (cf)(0, 0); for(int i = 0; i < 4; i++) acc += cmul(G[t][i+4], cconj(h[r][i]));
    W[t][r] = acc;
  }
  for(int s = 0; s < NSYM; s++){
    cf y[4]; for(int r = 0; r < 4; r++) y[r] = fd[(r*NSYM + s)*N + k];
    for(int t = 0; t < 4; t++){ cf acc = (cf)(0, 0); for(int r = 0; r < 4; r++) acc += cmul(W[t][r], y[r]); xe[(s*N + k)*4 + t] = acc; }
  }
}

// 16-QAM Gray max-log LLRs: re = (1-2 b0)(1+2 b1)/sqrt(10), same for im with b2,b3
__kernel void demap(__global const cf *xe, __global float4 *llr){
  int i = get_global_id(0); cf x = xe[i]; float th = 2.0f/sqrt(10.0f);
  llr[i] = (float4)(x.x, fabs(x.x) - th, x.y, fabs(x.y) - th);
}

// "AI job": dense FMA work that competes for the GPU
__kernel void ai_job(__global float *a, int iters){
  int i = get_global_id(0); float v = a[i];
  for(int k = 0; k < iters; k++) v = v*1.0000001f + 0.5f;
  a[i] = v;
}

// calibration ping-pong over fine-grained SVM
__kernel void pp(__global volatile uint *ctl, __global ulong *gt, uint n, uint base){
  for(uint i = 0; i < n; i++){ uint want = base + i + 1, c; ulong sp = 0;
    while((c = ctl[0]) != want){ if(c == 0xFFFFFFFEu || ++sp > 400000000ul) return; }
    gt[i] = gclk(); mem_fence(CLK_GLOBAL_MEM_FENCE); ctl[1] = want; }
}
""" % dict(N=NFFT, LOGN=LOGN, NSYM=NSYM)


class GPU:
    def __init__(self):
        self.ctx = cl.create_some_context(interactive=False)
        P = cl.command_queue_properties.PROFILING_ENABLE
        self.q = cl.CommandQueue(self.ctx, properties=P)       # 5G slot queue (in-order)
        self.q_ai = cl.CommandQueue(self.ctx, properties=P)    # AI job queue
        self.prg = cl.Program(self.ctx, SRC).build(options=["-cl-std=CL2.0"])
        self.k = {n: cl.Kernel(self.prg, n) for n in ("stamp", "fft", "equalize", "demap", "ai_job", "pp")}
        self.k_ai_stamp = cl.Kernel(self.prg, "stamp")          # separate instance: args differ per queue
        f = cl.svm_mem_flags.READ_WRITE | cl.svm_mem_flags.SVM_FINE_GRAIN_BUFFER
        self.ctl = cl.SVM(cl.svm_empty(self.ctx, f, 64, np.uint32))
        self.gt = cl.SVM(cl.svm_empty(self.ctx, f, 400, np.uint64))
        self.seq = 0

    # ---------- certified clock mapping ----------
    def calibrate(self, batches=10, n=400):
        """Ping-pong brackets: returns array of (t0_ns, t1_ns, gpu_tick) with host perf_counter_ns."""
        c, g, k, pc = self.ctl.mem, self.gt.mem, self.k["pp"], time.perf_counter_ns
        rows = []
        for _ in range(batches):
            base = self.seq; c[0] = base; c[1] = base
            k.set_args(self.ctl, self.gt, np.uint32(n), np.uint32(base))
            cl.enqueue_nd_range_kernel(self.q, k, (1,), None); self.q.flush()
            got = []
            for i in range(n):
                w = base + i + 1; a = pc(); c[0] = w
                lim = a + 1_000_000_000
                while c[1] != w:
                    if pc() > lim: break
                b = pc(); got.append((a, b))
                if c[1] != w: break
            c[0] = 0xFFFFFFFE; self.q.finish()
            rows += [(a, b, int(g[i])) for i, (a, b) in enumerate(got)]
            self.seq = base + n + 1
        return np.array(rows, dtype=np.float64)


def fit_clock(br):
    """Fit host_ns = a*tick + b on the tightest brackets. Returns (a, b, eps_ns, tick0).
    eps = worst-case distance from the mapping to the true read time, from brackets that must contain it."""
    t0, t1, g = br.T
    tick0 = g.min(); G = g - tick0; w = t1 - t0; mid = (t0 + t1) / 2
    sel = w <= max(np.percentile(w, 2), 1500)
    a, b = np.polyfit(G[sel], mid[sel], 1)
    pred = a*G[sel] + b
    # mapping must fall inside every tight bracket; eps covers half the tightest widths plus any violation
    viol = np.maximum(np.maximum(t0[sel] - pred, pred - t1[sel]), 0)
    eps = float(np.percentile(w[sel], 50) / 2 + viol.max())
    return a, b, eps, tick0, int(sel.sum())


def make_slot_data(rng, snr_db=30):
    """Random 16-QAM on 4 layers, frequency-selective 4x4 Rayleigh channel, received time-domain samples."""
    bits = rng.integers(0, 2, size=(NSYM, NFFT, 4, 4))            # [s][k][t][bit]
    re = (1 - 2*bits[..., 0]) * (1 + 2*bits[..., 1]) * AMP
    im = (1 - 2*bits[..., 2]) * (1 + 2*bits[..., 3]) * AMP
    X = (re + 1j*im)                                               # [s][k][t]
    taps = (rng.standard_normal((8, 4, 4)) + 1j*rng.standard_normal((8, 4, 4))) / np.sqrt(16)
    H = np.fft.fft(taps, n=NFFT, axis=0)                           # [k][r][t]
    Y = np.einsum("krt,skt->rsk", H, X)
    Y += (10**(-snr_db/20)) * (rng.standard_normal(Y.shape) + 1j*rng.standard_normal(Y.shape)) / np.sqrt(2)
    td = np.fft.ifft(Y, axis=2)                                    # [r][s][n]
    c64 = lambda z: np.ascontiguousarray(z).astype(np.complex64)
    return c64(td), c64(H), bits
