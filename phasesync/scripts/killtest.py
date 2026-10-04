"""
Kill test: N distributed TRPs with independent oscillators, reciprocity CJT (TDD, SRS-based).
Question 1: how much coherent gain is lost if you just use the latest SRS (stale)?
Question 2: how much does a 2nd-order tracking loop (alpha-beta == type-II digital PLL) recover?

Pure numpy, runs on a laptop CPU in ~1 minute. No Sionna needed for this first test.

Model per TRP n (UE oscillator is common to all TRPs and cancels in relative phase):
  y_n(t) = y0_n + y_RWFM(t) + y_WFM(t)              fractional frequency
  theta_n(t) = 2*pi*fc * integral(y_n) + white PM (PLL jitter, iid per evaluation)
SRS at TRP n measures  psi_n = theta_n - theta_ref + N(0, sigma_est)   (ref = TRP 0)
DL precoder for TRP n applied at slots  t_srs + D ... t_srs + D + P - 1 uses predicted psi_n.
Metrics: coherent-gain efficiency |sum e^{j eps}|^2 / N^2 (dB) and 2-UE ZF rate loss.
"""
import numpy as np, sys

def osc_paths(rng, trials, N, steps, dt, fc, y0max, a_wfm, b_rwfm):
    y0 = rng.uniform(-y0max, y0max, (trials, N, 1))
    wfm = rng.normal(0, a_wfm / np.sqrt(dt), (trials, N, steps))           # sigma_y(tau)=a/sqrt(tau)
    rw = np.cumsum(rng.normal(0, np.sqrt(3 * b_rwfm**2 * dt), (trials, N, steps)), axis=2)  # sigma_y(tau)=b*sqrt(tau)
    y = y0 + wfm + rw
    th = 2 * np.pi * fc * np.cumsum(y, axis=2) * dt
    return th - th[:, :1, :]                                              # relative to TRP 0

def run(fc, y0max, slot, P, D, trials=200, N=4, T=1.0, a_wfm=7.5e-13, b_rwfm=1e-10,
        pm_deg_at_3g5=1.0, est_deg=1.0, K=2, snr_db=20, seed=0, finit_avg=1):
    rng = np.random.default_rng(seed)
    sub = 4                                   # evaluation points per slot
    dt = slot / sub
    steps = int(T / dt)
    psi = osc_paths(rng, trials, N, steps, dt, fc, y0max, a_wfm, b_rwfm)
    pm = np.deg2rad(pm_deg_at_3g5 * fc / 3.5e9)
    nslots = steps // sub
    srs_slots = np.arange(0, nslots - D - P, P)
    # measurements
    meas = psi[:, :, srs_slots * sub] + rng.normal(0, np.deg2rad(est_deg), (trials, N, len(srs_slots))) \
        + rng.normal(0, pm, (trials, N, len(srs_slots)))
    meas[:, 0, :] = 0
    Tm = P * slot
    # alpha-beta tracker on unwrapped innovation (2nd-order loop). gains from a coarse grid, chosen once.
    def track(alpha, beta, f_init):
        ph = meas[:, :, 0].copy(); fr = f_init.copy()   # fr in rad/s
        out_ph = np.zeros_like(meas); out_fr = np.zeros_like(meas)
        out_ph[:, :, 0] = ph; out_fr[:, :, 0] = fr
        for k in range(1, meas.shape[2]):
            pred = ph + fr * Tm
            e = np.angle(np.exp(1j * (meas[:, :, k] - pred)))
            ph = pred + alpha * e
            fr = fr + beta * e / Tm
            out_ph[:, :, k] = ph; out_fr[:, :, k] = fr
        return out_ph, out_fr
    # coarse frequency acquisition (models a 2-symbol DMRS/TRS-style intra-slot CFO estimate, +-1/(2*0.25ms)=+-2 kHz
    # unambiguous, accuracy ~ est noise / (2pi*0.25ms) ). Without it, P*slot aliasing kills the loop when CFO is large.
    f_true = (psi[:, :, sub] - psi[:, :, 0]) / slot
    f_init = f_true + rng.normal(0, np.deg2rad(est_deg) * np.sqrt(2) / 0.25e-3 / np.sqrt(finit_avg), f_true.shape)
    f_init[:, 0] = 0
    results = {}
    eval_idx = []   # (srs index k, eval sample index)
    for k, s in enumerate(srs_slots):
        for d in range(D, D + P):
            for q in range(sub):
                eval_idx.append((k, (s + d) * sub + q))
    kk = np.array([e[0] for e in eval_idx]); tt = np.array([e[1] for e in eval_idx])
    truth = psi[:, :, tt] + rng.normal(0, pm, (trials, N, len(tt)))
    truth[:, 0, :] = 0
    horizon = (tt - srs_slots[kk] * sub) * dt     # seconds since SRS
    est = {}
    est['stale'] = meas[:, :, kk]
    best = None
    for alpha in [0.2, 0.4, 0.6, 0.8]:
        for beta in [0.01, 0.03, 0.1, 0.2, 0.4]:
            ph, fr = track(alpha, beta, f_init)
            e_ = ph[:, :, kk] + fr[:, :, kk] * horizon
            loss = np.mean(np.abs(np.exp(1j * (truth - e_)).sum(1))**2) / N**2
            if best is None or loss > best[0]:
                best = (loss, alpha, beta, e_)
    est['loop'] = best[3]
    est['ideal'] = truth
    for name, e_ in est.items():
        err = np.angle(np.exp(1j * (truth - e_)))
        g = np.mean(np.abs(np.exp(1j * err).sum(1))**2) / N**2
        rms = np.rad2deg(np.sqrt(np.mean(err[:, 1:]**2)))
        # 2-UE ZF with N single-antenna TRPs, iid Rayleigh, static channel; phase error enters as diag(e^{j err})
        sub_idx = rng.choice(err.shape[2], 64, replace=False)
        rates = []; rates0 = []
        snr = 10**(snr_db / 10)
        for i in range(min(trials, 50)):
            H = (rng.normal(size=(K, N)) + 1j * rng.normal(size=(K, N))) / np.sqrt(2)
            W = np.linalg.pinv(H); W /= np.linalg.norm(W, axis=0, keepdims=True)
            for j in sub_idx:
                Heff = H * np.exp(1j * err[i, :, j])[None, :]
                G = np.abs(Heff @ W)**2 * snr / K
                sig = np.diag(G); intf = G.sum(1) - sig
                rates.append(np.sum(np.log2(1 + sig / (intf + 1))))
                G0 = np.abs(H @ W)**2 * snr / K
                rates0.append(np.sum(np.log2(1 + np.diag(G0) / 1)))
        results[name] = (10 * np.log10(g), rms, np.mean(rates) / np.mean(rates0))
    return results, best[1:3]

if __name__ == '__main__':
    cases = [
        # label, fc, y0max (relative freq offset bound), slot, SRS period P (slots), latency D (slots)
        ("FR1 3.5G, RUs locked +-50ppb (3GPP bound), SRS 5ms", 3.5e9, 50e-9, 0.5e-3, 10, 4),
        ("FR1 3.5G, RUs locked +-50ppb, SRS 20ms",              3.5e9, 50e-9, 0.5e-3, 40, 4),
        ("FR1 3.5G, free TCXO +-2ppm (USRP B210), SRS 5ms",    3.5e9, 2e-6, 0.5e-3, 10, 4),
        ("FR3 7G, locked +-50ppb, SRS 5ms",                    7e9, 50e-9, 0.5e-3, 10, 4),
        ("FR3 13G, locked +-50ppb, SRS 5ms",                   13e9, 50e-9, 0.5e-3, 10, 4),
        ("FR3 13G, locked +-50ppb, SRS 20ms",                  13e9, 50e-9, 0.5e-3, 40, 4),
        ("FR2 28G, locked +-50ppb, SCS120 SRS 2.5ms",          28e9, 50e-9, 0.125e-3, 20, 8),
    ]
    for lab, fc, y0, slot, P, D in cases:
        r, ab = run(fc, y0, slot, P, D)
        print(f"\n{lab}   (loop alpha,beta={ab})")
        for k, (g, rms, rr) in r.items():
            print(f"   {k:6s} coherent-gain eff {g:7.2f} dB | rms rel phase err {rms:7.1f} deg | 2-UE ZF rate / ideal-ZF {rr:5.2f}")
    print("\nnon-coherent reference for N=4: coherent-gain eff = -6.02 dB")
