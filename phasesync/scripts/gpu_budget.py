# When is the GPU necessary vs incidental for distributed phase tracking + CJT/MU precoding?
# Real flops (1 complex MAC = 8 real flops). Order-of-magnitude only.
CM = 8
def budget(label, trps, ant_per_trp, prb, layers, srs_ues, slot_s, fs, sym=14):
    ports = trps*ant_per_trp; sc = prb*12
    srs_re = sc//4                                   # comb-4 SRS, 1 symbol
    # (1) SRS LS channel estimate + ~10 MAC/RE denoise/interp per port per UE
    srs = srs_ues*ports*srs_re*11*CM
    # (2) per-TRP oscillator phase/freq loop: pool phase of all UEs' estimates (1 MAC per RE) + PI update
    loop = srs_ues*ports*srs_re*1*CM + trps*20
    # (3a) apply loop output to existing BFW: diag(e^{-j theta}) * W per PRB
    apply_diag = ports*layers*prb*CM
    # (3b) recompute RZF BFW per PRB: H^H H (L*L*P), LxL inverse (~L^3), W=H^H inv (P*L*L)
    rzf = prb*(2*ports*layers*layers + layers**3)*CM
    # (4) intra-slot CFO de-rotation in time domain (needs time-domain samples: RU/FPGA in O-RAN 7.2x)
    td = ports*fs*slot_s*1*CM
    # (5) PDSCH precoding itself (for scale): ports x layers per data RE
    pdsch = sc*sym*ports*layers*CM
    print(f"\n{label}: {ports} ports, {prb} PRB, {layers} layers, {srs_ues} SRS UEs/slot, slot {slot_s*1e3} ms")
    for n,v in [("SRS chan est",srs),("phase loop (pooled)",loop),("apply diag phase fix",apply_diag),
                ("RZF recompute/slot",rzf),("time-domain NCO derotate",td),("PDSCH precoding",pdsch)]:
        print(f"   {n:26s} {v/1e6:10.2f} MFLOP/slot = {v/slot_s/1e9:9.2f} GFLOP/s")
budget("MSc testbed (4 B210 TRPs x1 ant, 20 MHz)", 4, 1, 51, 2, 2, 0.5e-3, 23.04e6)
budget("Small CJT cluster (4 TRP x 4 ant, 100 MHz)", 4, 4, 273, 4, 8, 0.5e-3, 122.88e6)
budget("Cell-free/6G-scale (16 TRP x 4 ant = 64 ports, 100 MHz, 16 layers)", 16, 4, 273, 16, 48, 0.5e-3, 122.88e6)
budget("6G FR3 (16 TRP x 16 ant = 256 ports, 400 MHz@120k? use 273PRB@60k, 32 layers)", 16, 16, 273, 32, 64, 0.25e-3, 245.76e6)
print("\nRef points: one modern CPU core (AVX-512 FP32) ~ 50-100 GFLOP/s peak; A100 ~ 19.5 TFLOP/s FP32; RTX 4090 ~ 82 TFLOP/s FP32.")
