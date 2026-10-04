# Distributed-radio phase coherence: examiner verdict and plan (2026-10-04)

Produced by a verification workflow: four research tracks (prior art, 3GPP/6G status, oscillator/PLL numbers, feasibility and kill test) and a strict examiner. Verdict: **pursue with changes**.

## Plain answer

Short version: keep the topic, but change the question. As you pitched it ("a digital PLL across radios, running on the GPU, so cheap crystals replace GPSDOs"), it has been done since 2012. AirSync, MegaMIMO, Chorus (LTE, 2018) and Merlo & Nanzer (2025, 2-4 USRPs, no cables) all do it. Jiang et al. (2026) already track inter-radio phase from uplink SRS every TDD period on NR-style hardware. Doing it on a GPU is a choice of platform, so it does not make the work new. Any examiner will cite these papers within a minute.

Phase-locking a GPU to a CPU would do nothing for 5G. The carrier phase that ruins coherent transmission lives in each radio's crystal and RF PLL, not in any compute clock. Drop that idea.

There is a real, live problem where crystals and PLLs hurt 5G and 6G. 3GPP added coherent joint transmission (CJT) across several sites in Rel-18 assuming perfect sync. Rel-19 then had to add UE reports of phase and frequency offset because sync is not perfect. Cao et al. (IEEE JSAC 2023) measured commercial O-RAN radios locked with PTP+SyncE. Each radio's own PLL still drifted more than 20 degrees in 7.5 ms, and multi-user spectral efficiency fell 46% with 5 ms calibration delay and 64% with 10 ms. Their own remark says the drift comes from each radio's PLL and that LO phase tracking is future work.

The question nobody has answered on hardware is this: is that PLL drift predictable over the 1.5 to 4 ms you have to look ahead in an O-RAN/GPU baseband, so that software tracking can fix it? Or is it fast PLL noise that only better PLL hardware can fix? Where does the answer flip as carrier frequency rises from 3.5 GHz to the 7-15 GHz bands 6G is targeting?

You can answer this in weeks, with 3 USRPs or with no hardware at all. Either outcome is a result, and it uses exactly your crystal/PLL/timing interest. The GPU is still part of it, because it is where the loop would sit in a real DU and where you run the large simulations. It is not the novelty, though, and you should not claim it is. This will not single-handedly transform 5G. It is a narrow, honest, testable contribution on a problem that 3GPP and operators name as a blocker. That is what a strong MSc thesis, or a workshop or conference paper, looks like.

## Novelty, honestly

These parts are NOT new:
- Estimating per-radio phase and frequency offsets from pilots, tracking them with a PLL or Kalman filter, and pre-rotating each radio's precoder. AirSync 2012 (Kalman, FPGA), MegaMIMO 2012, MegaMIMO 2.0 2016, Chorus 2018 (LTE), Rogalin 2014, Quitin/Madhow 2013.
- Cable-free coherent USRP arrays. Merlo/Nanzer 2025: 60-70 ps, >99% median coherent gain.
- Kalman LO tracking inside the NR-style TDD frame. Ngo & Larsson 2025/26, in simulation.
- Per-TDD-period UL-SRS phase tracking with COTS RRUs. Jiang et al. 2026 at 26 GHz; Cao et al. 2023 measured the drift.
- UE-reported inter-TRP phase, frequency and delay offsets. Rel-19 TS 38.214 5.2.1.4.9-11, cjtc-Dd/F/P.
- Running the loop on a GPU. The arithmetic is tiny (under 2 GFLOP/s for 4 TRPs), so a CPU can do it.

What appears open (from the four reports plus my own spot checks of Cao, Jiang and Merlo, and two web searches; still check this with your supervisor):
(1) An experimental answer on whether inter-radio LO drift under a shared frequency reference can be predicted at the O-RAN weight-computation horizon (about 1.5-4 ms). All existing work either assumes a model (Wiener) or reports one operating point. Cao's Remark 2 attributes the drift to the radio PLL and leaves tracking as future work.
(2) A head-to-head of network-side SRS prediction against Rel-19 CJTC UE reports (24 or 11.6 degree phase steps, 0.39-13.3 ppb frequency steps, plus CSI latency). No public comparison was found.
(3) The carrier frequency at which the residual after an ideal slot-rate loop is set by the synthesizer's PLL phase noise rather than by tracking. That gives a PLL specification for 6G FR3 (7-24 GHz) CJT.

This is an incremental, measurement-driven contribution, not a new paradigm. The evidence does not support calling it revolutionary.

## The specific research question

For multi-user coherent joint transmission, radios share a frequency reference (PTP/SyncE-like) but each has its own LO PLL. The DU must fix precoders about 1.5-4 ms before air time (O-RAN 7.2x / Aerial FAPI 3-slot advance plus the TDD/SRS period). Two questions follow.

(a) What fraction of the measured inter-radio carrier-phase drift can a second-order predictor (type-II DPLL or 2-state Kalman fitted to the oscillator's Allan deviation) remove, compared with simply holding the last SRS estimate?

(b) As carrier frequency rises (3.5 -> 7 -> 13 GHz), at what point does the residual stop being limited by tracking and start being limited by the radios' own PLL synthesizer phase noise? In other words, what PLL/oscillator specification would 6G FR3 CJT radios need to keep residual phase error at or below about 5 degrees (the multi-user ZF requirement)?

Use the Rel-19 CJTC UE-report quantisation and latency as the standards baseline.

## Why it matters for 5G

Rel-18 CJT (up to 4 TRPs, FR1) assumed ideal sync. Rel-19 NR_MIMO_Ph5 added "CJT with non-ideal synchronization" via UE reports, which shows the problem is real and only partly addressed. PTP/SyncE aligns time to tens of ns. That is about 1260 degrees of carrier per ns at 3.5 GHz, so PTP cannot align carrier phase.

On commercial PTP+SyncE O-RAN radios, Cao et al. measured +/-30 degree LO wander, more than 20 degrees within 7.5 ms, and a 46%/64% multi-user spectral-efficiency loss at 5/10 ms calibration delay. Systems work such as RANBooster (SIGCOMM 2025) assumes PTP-synced RUs are coherent enough.

Ericsson/Nokia authors (Shafi et al. 2025) and XGMF 2026 name phase-synchronous downlink as a main reason D-MIMO adoption is slow. A measured answer to "can DU-side prediction fix this, or must the RU PLL be better?" bears directly on whether TDD CJT can rely on network-side tracking, which needs no 3GPP change, instead of UE reports that industry calls undesirable.

The size of the effect is in multi-user ZF: the SINR ceiling is about 15 dB at 10 degrees and about 21 dB at 5 degrees. In single-user beamforming it is small: 30 degrees costs about 0.5-1 dB.

## Why it matters for 6G

The 6G Radio study (RP-251881) requires reusing the 3.5 GHz site grid at about 7 GHz with comparable coverage, which pushes toward more antennas and cooperative or distributed MIMO. No inter-TRP sync mechanism has been agreed in 6GR, and the RAN1 CSI discussions mention "uncalibrated radios and phase errors" while still using the Rel-18 CJT codebook as baseline. Normative specs come in Rel-21 (about 2027-28), so evidence produced now can still matter.

Phase drift per ppb scales with carrier frequency: 1.26 deg/ms at 3.5 GHz, about 2.5 at 7 GHz, about 4.7 at 13 GHz, 10 at 28 GHz. Synthesizer jitter also scales, roughly with 20log(fc). The kill-test model already puts the ideal-loop floor near 6.5 degrees at 13 GHz and about 10 degrees at 28 GHz, where ZF rate falls to about 0.89 of ideal.

For 6G FR3 distributed MIMO, the question therefore shifts from "better software tracking" to "how good must each radio's PLL and reference be". That is a quantitative hardware-specification result linking crystals and PLLs directly to a 6G design decision.

## First experiment

Measure whether the inter-radio carrier-phase drift between two independently synthesised radios sharing one 10 MHz reference can be predicted 1.5-4 ms ahead. Hold-last-SRS is compared with a type-II DPLL and with a 2-state Kalman filter, replayed offline on the real phase trace, and the residual is turned into a multi-user ZF loss. There is a $0 no-hardware fallback on digitised Cao et al. traces.

1. Day 1: ask your supervisor or lab manager for 3 USRPs (B210/B200/X310) and a way to share 10 MHz (OctoClock or a signal generator plus a splitter). Also get 2x 30 dB attenuators, a 2:1 combiner and SMA cables. Check that each B210 is one TRP, because its two channels share an LO.
2. Wire it cabled, with no over-the-air link and no licence needed: TX-A and TX-B go through the attenuators into the combiner and then into RX. Case S: TX-A and TX-B take the shared 10 MHz (and PPS). Case F: both run on internal TCXOs, as a control. The RX may use its own clock, because its LO cancels when you take the phase difference.
3. Open each device as a separate UHD multi_usrp in Python and tune once to 3.5 GHz (repeat later at 5.8 GHz if possible). Never retune during a run, because AD9361 LO phase re-randomises on retune. TX-A sends a CW tone at +100 kHz baseband and TX-B at -100 kHz. Use narrowband (1 MSps or less) so that sample-clock offset does not matter.
4. Record 3 x 10 minutes per case. Note room temperature, and do one run with a deliberate disturbance (a fan on the boards).
5. Offline (numpy, laptop): for each 0.5 ms block, take the phase of each tone bin and form psi(t) = phi_A - phi_B. Unwrap it, remove the mean frequency, and compute the Allan deviation and phase PSD from 1 ms to 10 s.
6. Replay: sample psi at SRS periods {2.5, 5, 10, 20} ms with measurement noise added at 20 dB SNR. Predict ahead by h in {1.5, 2.5, 4} ms with (a) hold-last, (b) an alpha-beta / type-II DPLL with a bandwidth sweep, and (c) a 2-state Kalman with Q fitted to the measured ADEV. Record the residual rms for each method.
7. Extend the existing killtest.py (4 TRPs, 2-4 UEs, RZF). Feed it the measured residuals and report the multi-user ZF SINR ceiling and the rate ratio against ideal. Sanity anchor: feed it a Cao-like trace (+/-30 deg wander, >20 deg per 7.5 ms) and check that it reproduces a 40-65% loss at 5-10 ms delay.
8. Fallback if no USRPs: digitise Cao et al. Fig. 8a/8b and Jiang et al.'s tracking plots (WebPlotDigitizer, from arXiv 2208.14048 and 2601.14648) and run steps 6-7 on those traces. Say clearly that this is coarse.

- **Metric:** Residual rms inter-radio phase error (degrees) at prediction horizon h = 1.5 / 2.5 / 4 ms for hold-last vs DPLL vs Kalman, and its conversion to the multi-user ZF SINR ceiling (about -20log10(sigma_rad) dB) and RZF rate ratio against ideal. Secondary: relative Allan deviation and phase PSD of psi in case S (a small public dataset in itself), and the measured CFO in case F.
- **Success looks like:** In case S, hold-last at h = 2.5 ms gives 10 degrees or more, and the DPLL/Kalman predictor cuts it at least 2x, to 5 degrees or less. That lifts the ZF ceiling from about 15 dB or less to about 21 dB or more, and the rate ratio from 0.9 or less to 0.97 or more. The predictor's advantage grows with h. The drift is then predictable, DU-side tracking is the fix, and you go on to the Sionna study and the Rel-19 CJTC comparison.
- **Failure looks like:** (i) Hold-last is already 3 degrees or less at 4 ms. The B210 PLLs under a shared reference are much quieter than Cao's commercial RRUs, so the problem does not reproduce on this kit. Rely on the Cao-calibrated model, try X310/UBX or different LO settings, and state the hardware dependence. (ii) The predictor is no better than hold-last (within about 10%) and the PSD is flat over the 100 Hz-1 kHz range. The drift is fast PLL-loop noise that no slot-rate loop can remove, so the fix is the RU's PLL or clock-cleaner design, not software. That is a publishable negative result, and you pivot to question (b): a PLL specification for FR3 CJT. (iii) Case F shows kHz-level CFO and fast slips. That only confirms frequency discipline is mandatory and is expected; do not treat it as the result.
- **Cost and time:** About 3 weeks. 2-3 days to set up and get the UHD script working, 1 day to record, 1-2 weeks for analysis and the replay/ZF model. Cost is $0 if the lab has 3 USRPs and a 10 MHz source, plus about $100-300 for attenuators, a combiner and cables. Buying 3 B210s would cost about $7k, which is out of budget, so use the fallback instead. No GPU is needed for this step. The fallback (digitised Cao/Jiang traces) costs $0 and takes about 1 week.

## Later experiments

- Sionna (PyTorch, Sionna 2.2 has no oscillator block, so write a ~50-line per-TRP exp(j*theta_n(t)) block): 38.901 UMi/InH, N = 2/4/8 TRPs x 4 antennas, K = 2-8 UEs, RZF downlink built from generic OFDM blocks. The oscillator model is fitted to your measured ADEV plus a PLL phase-noise term (not a pure Wiener process, which overstates PLL-locked drift by more than 10x). Report sum SE against prediction horizon and SRS period. About 2-10 GPU-hours on vast.ai, $1-15.
- Standards baseline: implement Rel-19 CJTC quantisation (cjtc-P M = 16/32, cjtc-F 0.1/0.2 ppm with M = 16/32/256, cjtc-Dd) with realistic CSI-report latency (5-20 ms). Compare it with network-side SRS prediction on the same traces. Check the formulas against the official TS 38.214 v19.x, including the sign convention for FO.
- FR3 crossover (6G): repeat at 7, 13 and 28 GHz with numerology 30/60/120 kHz. Scale the reference noise by 20log(fc/fref) and use PLL datasheet models (e.g. TI LMX2594 FOM, ADI ADF4372). Find the minimum synthesizer integrated jitter and reference ADEV that keep the residual at or below 5 degrees at a 2.5 ms horizon. The output is a 'required PLL spec vs carrier' curve.
- Close the loop on hardware: RX feeds psi-hat and f-hat back over Ethernet, TX-B pre-rotates (digitally or via the UHD DSP NCO), and both transmit together. Measure coherent gain against the ideal 3 dB, and ZF null depth at a second RX. Null depth is the sensitive metric: about -15 dB at 10 degrees.
- Multi-UE pooling: the TRP oscillator term is common to all UEs' SRS, while Doppler is per UE. Test whether a pooled per-TRP estimator separates TRP drift from UE motion and allows longer SRS periods. Avoid cycle slips with a 2-symbol DMRS/TRS frequency-acquisition step.
- GPU integration, presented honestly: fuse the per-TRP tracker into a batched SRS->RZF PyTorch/CUDA module and measure the added latency at 64-256 ports. If an Aerial cuPHY testbed with 2+ RUs becomes available, insert it into the SRS-BFW pipeline. Claim zero added latency, not that the GPU is needed for the loop.

## What it can and cannot prove

It CAN show, for the specific radios measured, whether inter-radio LO drift under a shared frequency reference can be predicted at O-RAN-relevant horizons, and how much multi-user ZF performance a second-order predictor recovers compared with hold-last and with Rel-19 UE-report quantisation. With the FR3 extension it can give a quantitative, model-based PLL/oscillator specification for CJT at 7-28 GHz.

It CANNOT prove:
- that the result transfers to commercial O-RAN RUs. USRP/AD9361 PLLs differ from RRU clock trees, and Cao's drift was 10-20x the idealised model, so state the hardware dependence.
- end-to-end 5G-stack gains. OAI and srsRAN cannot do multi-TRP CJT with independent radios, and there is no Aerial multi-RU testbed.
- mobility, multipath robustness of OTA/SRS estimation, or scaling past about 4 radios.
- that cheap free-running crystals can replace PTP/SyncE. At 3.5 GHz a 0.5-2 ppm TCXO gives 1.75-7 kHz CFO (6-23% of a 30 kHz subcarrier). That causes ICI a DU-side frequency-domain rotation cannot remove, and TS 38.104 requires +/-50 ppb anyway.
- anything about GPU necessity. The loop is cheap enough for a CPU.

## Biggest risks

- Hardware is unconfirmed. Without 3 USRPs and a shared 10 MHz source, the distinctive measured part becomes a digitised-trace or simulation study, which reviewers may read as a rerun of Ngo & Larsson or BeamSync.
- Unrepresentative hardware. B210 PLLs under a shared reference may drift far less (or differently) than commercial RRUs, which would leave nothing to fix on your kit. Mitigation: report it, and anchor the simulation to Cao et al.'s measured +/-30 deg and >20 deg per 7.5 ms.
- Prior-art crowding. Any wording like 'novel digital PLL across radios' or 'GPU makes coherence possible' will be rejected; keep the claim to the predictability measurement, the Rel-19 comparison and the FR3 PLL-spec curve.
- Model mistakes. A pure Wiener phase model overstates ms-scale drift for PLL-locked carriers by more than 10x. 3GPP TR 38.803 pole-zero models have no close-in drift and understate it. Either mistake makes the simulation meaningless.
- Measurement artefacts. AD9361 LO phase re-randomises on retune. TX and RX use separate synthesizers, so reciprocity is not guaranteed. Sample-clock offset matters in free-running runs. Cycle slips occur at long SRS periods (80 ms) and with XO-class oscillators at FR3.
- Weak metric choice. Single-user coherent gain hides the effect (30 degrees costs about 0.5-1 dB); always report multi-user ZF SINR, rate or null depth.
- Standards details come partly from secondary mirrors (iTecSpec, ShareTechnote, summaries). Verify the CJTC formulas, RP numbers and the TS 38.104 TAE and frequency-error values against the official 3GPP/ETSI documents before quoting them in the thesis.

## If not this

If no radios are available and the student wants to stay in simulation, do question (b) on its own: "What PLL and reference-oscillator specification do 6G FR3 (7-24 GHz) distributed-MIMO radios need for multi-user CJT under O-RAN latency, and when is a shared LO or reference unavoidable?" Combine datasheet PLL models (FOM, loop bandwidth, reference ADEV) with Cao-calibrated drift, run GPU Monte-Carlo sweeps in Sionna ($1-15), and compare with Rel-19 CJTC quantisation. It keeps crystals, PLLs, timing and GPUs, needs no hardware, and gives a concrete spec curve, though it is weaker than a measured result.

A second fallback: publish a small open dataset of inter-USRP carrier-phase drift (free-running, shared 10 MHz, shared 10 MHz with disturbances, over 1 ms to 10 min). Published measured traces are scarce (only isolated plots in Cao 2023).

Do not pursue: GPU-to-CPU phase locking (no effect on RF coherence), or "cheap crystals with no shared reference in NR" (physically limited by CFO and ICI under split 7.2).

## Research-track numbers

### prior-art

- MegaMIMO 2012: 10 USRP2 APs, independent oscillators, 95th-percentile phase misalignment 0.05 rad (2.86 deg), about 10x throughput with 10 APs, 2.4 GHz.
- AirSync 2012: phase coherence 'within a few degrees', timing within the OFDM CP, WARP FPGA, 2-4 APs, Kalman-predicted phase drift.
- MegaMIMO 2.0 2016: 4 APs x 2 antennas (8x8 virtual MIMO), Zynq Z7020 + FMCOMMS2, 2.457 GHz / 20 MHz, up to 3.3x throughput. AGC gain steps can introduce up to pi rad phase change, and uncorrected phase errors cap SNR at about 12-14 dB.
- Chorus 2018: LTE-compatible, leaderless, 2.7x over leader-based D-MIMO.
- RFClock 2021: frequency offset <0.107 Hz mean, time/phase within 5 ns, 5-node beamforming.
- Merlo/Nanzer 2025: 2-4 COTS X310s, no cables/GNSS, about 60-70 ps time/phase std at 27 dB SNR, median coherent gain >99%, 3.73 ppb frequency RMSE under motion, carriers 1.0 and 2.1 GHz.
- Cao et al. JSAC 2023: 4 COTS O-RAN RRUs, 4.9 GHz / 100 MHz, PTP+SyncE-locked. LO phase wander about +/-30 deg, >20 deg drift in 7.5 ms. 5 ms calibration delay puts 5% of channels above 10 deg error; 10 ms puts 10% above 15 deg. Sum-SE loss 46% (5 ms) and 64% (10 ms) for 16-stream ZF.
- Jiang et al. 2026: 4 TRPs, 26 GHz / 200 MHz. Without phase tracking the error is >90 deg; SRS-based tracking brings it below about 20 deg (at about 30 dBm calibration power); calibration overhead cut to about 10%.
- Repeater-aided OTA sync (2026, simulation): about 1e-3 rad RMSE at 30 m with a 10 mW repeater. At 3 GHz, 0.01 rad corresponds to 0.5 ps.
- White Rabbit D-MIMO (Bigler 2018): sub-ns accuracy, ps precision. About 0.8 ps is needed for phase-accurate arrays at 5 GHz, and about 33 ps for 0.01 m/s velocity accuracy.
- ARoF CJT (EuCNC 2023): +5 dB combined diversity and power gain with 2 TRxPs.
- Techtile WPT: 14 dB measured vs 14.9 dB theoretical for 31 elements. Phase error <20 deg costs <1 dB; >40 deg costs >3 dB.
- CJT tolerance (inter-cell calibration study, arXiv 1905.01046): pi/8 (22.5 deg) gives trivial loss, pi/4 is acceptable, pi/2 gives clear degradation.
- Twente cooperative ISAC targets: 3.3 ppb frequency stability and 4 ps time error, reachable with crystal oscillators at millisecond-scale OTA resync intervals.
- My own budget (3.5 GHz, 30 kHz SCS, 0.5 ms slot; script at /tmp/claude-0/-home-user/c22b0607-07f1-5108-a24d-4caf6e85f472/scratchpad/phasesync/budget.py): XO +/-10 ppm gives 35 kHz CFO (1.17x the subcarrier spacing; OFDM breaks, so digital phase correction alone cannot fix it). XO 2 ppm gives 7 kHz (0.23 SCS, ICI-limited SIR about 7.5 dB). TCXO 0.5 ppm gives 1.75 kHz (315 deg per slot, ICI SIR about 19.5 dB, not enough for 256-QAM). OCXO 10 ppb gives 35 Hz (6.3 deg per slot). Free-running crystals therefore need frequency locking (OTA, SyncE or two-tone), or time-domain pre-compensation at the radio, before any per-slot phase loop helps.
- Coherent-gain loss for i.i.d. Gaussian residual phase error sigma (large N approximation e^-sigma^2): 10 deg costs 0.13 dB, 20 deg 0.53 dB, 30 deg 1.19 dB, 45 deg 2.68 dB. Monte Carlo with N=4: 0.10, 0.39 and 0.86 dB at 10, 20 and 30 deg. Single-user coherent combining is forgiving. Multi-user ZF is not: a crude leakage ceiling is about 1/sigma^2, roughly 15 dB SINR at 10 deg and 9 dB at 20 deg. That is why the Cao et al. losses are tens of percent.

Open:

- A REAL-TIME, NR-compliant inter-RU phase-tracking loop inside a GPU DU (Aerial cuPHY or OAI on GPU) for COTS split-7.2 O-RAN RUs. Aerial has no inter-RU calibration or CJT. RANBooster (SIGCOMM 2025) builds dMIMO from PTP-synced commodity RUs and assumes PTP is enough, while Cao et al. measured >20 deg drift in 7.5 ms on PTP+SyncE RRUs. A 'PTP-only dMIMO vs PTP + SRS phase loop' comparison on real RUs is a concrete, publishable systems gap. The contribution is the measurement plus closing the loop at line rate, not the algorithm.
- Calibration aging across SRS-to-PDSCH latency: Cao et al. show 46-64% SE loss at 5-10 ms delay. Predicting the phase forward with a fitted oscillator model (Kalman with Wiener+CFO states, or a PLL-type loop filter), validated on measured RU phase traces, is only analysed in simulation (Ngo & Larsson; Sun et al. 2608.10637). An experimental 'SE vs prediction horizon' curve on real hardware is still missing.
- Public measurement datasets of inter-RU / inter-USRP carrier-phase drift under PTP-only, PTP+SyncE and free-running TCXO setups, over ms to minute timescales and temperature. Only isolated plots exist (Cao 2023). RENEW publishes CFO datasets for distributed uplink (Zafari et al., arXiv 2508.08506), but not for downlink CJT phase.
- Free-running cheap crystals (no SyncE) in NR numerology: at 3.5 GHz even a 0.5 ppm TCXO gives about 5.8% of a 30 kHz subcarrier as CFO, which causes ICI. With split 7.2 the DU only sends frequency-domain IQ, so a GPU loop can apply per-symbol phase but cannot remove intra-symbol CFO. Making 'cheap crystals' work therefore needs a frequency-transfer or OTA disciplining stage at the RU. That interface (what the DU estimates vs what the RU corrects) is not well specified.
- Mobility and multipath robustness of OTA/SRS-based tracking (Collmann et al. 2026 find OTA calibration multipath-sensitive; Jiang et al. needed sensing assistance in dynamic scenes). Scaling past about 4-8 radios on hardware (Larsson's unbounded-error topologies). Interaction with UE-side Rel-19 reports (fusing network-side SRS tracking with UE-reported offsets).
- Distributed ISAC long-coherence on hardware: most D-ISAC phase sync work (MovISAC, OTA D-ISAC) is simulation only.

Red flags:

- The brief's 'MegaMIMO (SIGCOMM 2013)' is wrong. It is SIGCOMM 2012, and JMB is the same paper. MegaMIMO 2.0 is SIGCOMM 2016, not MobiCom.
- Claiming novelty for 'Kalman phase tracking across distributed APs/TRPs' would be false. It is in AirSync (2012) and Ngo & Larsson (2025/26).
- Claiming novelty for 'per-slot SRS-based inter-TRP phase tracking in NR' would be false. Jiang et al. (arXiv 2601.14648, 2026) and Cao et al. (JSAC 2023) do it with COTS O-RAN RRUs, although with PTP+SyncE-locked references.
- 'Cheap crystals with no shared reference' is physically limited in NR. Free-running XO/TCXO CFOs at 3.5 GHz (1.75-35 kHz) are a sizeable fraction of the 30 kHz subcarrier spacing, so frequency must be disciplined (SyncE, OTA two-tone or two-way transfer) before a slot-rate phase loop can work. With O-RAN split 7.2, the GPU DU cannot apply intra-symbol frequency correction.
- Simulation-only results with a Wiener phase-noise model will look like Ngo & Larsson and BeamSync. Without measured oscillator traces or hardware, the contribution will read as a re-derivation.
- Sionna 2.2.0 has no phase-noise/CFO impairment model. Any 'realistic oscillator model' must be implemented and justified by the student (for example the 3GPP TR 38.803 multi-pole/zero PN model or measured Allan deviation).
- USRP availability is unconfirmed. Without at least 2 radios with independent references, the only distinctive (experimental) part of the plan collapses.
- Some numbers above are from secondary summaries rather than the full papers: the AirSync 'few degrees', Vidyut 8.2x and Chorus 2.7x figures, and the exact Rel-19 report quantities. Check them against the primary PDFs or RAN1 chair notes before citing in a thesis.

### standards

- Rel-19 cjtc-P phase report: M_Phi in {16,32}, so 24 deg or 11.6 deg steps over 2*pi (TS 38.214 section 5.2.1.4.11)
- Rel-19 cjtc-F frequency report: A_FO in {0.1, 0.2} ppm, M_FO in {16,32,256}, so 0.39-13.3 ppb steps (TS 38.214 section 5.2.1.4.10)
- Rel-19 cjtc-Dd delay report: A_D in {CP/2, CP}, M_D in {32..256}. At 30 kHz SCS that is 4.6-75.6 ns steps; at 15 kHz, 9.2-151 ns (TS 38.214 section 5.2.1.4.9)
- Rel-18 CJT: up to 4 TRPs, FR1, ideal sync/backhaul assumed; CSI-RS ports extended to 128 in Rel-19
- TS 38.104 BS frequency error: +/-50 ppb wide area, +/-100 ppb medium range/local area. Inter-site difference up to 100 ppb = 350 Hz at 3.5 GHz
- TS 38.104 TAE: 65 ns for MIMO on a carrier. TDD cell phase sync: 3 us. Ericsson: the 65-260 ns TAE applies to co-located deployments only
- O-RAN timing categories: A relative TAE <=130 ns, B <=260 ns. Regular O-RU |TE| <=80 ns, enhanced <=35 ns. Fronthaul FFO <=21/27/30 ppb; about 13 ppb (Class B T-TSC) or about 3 ppb (Class C)
- Carrier phase drift at 3.5 GHz: 1.26 deg/ms per ppb. 3 ppb = 3.8 deg/ms; 13 ppb = 16.4 deg/ms; 50 ppb = 63 deg/ms; 100 ppb = 126 deg/ms; free-running TCXO at 0.5-2 ppm = 630-2520 deg/ms
- Uncorrected CFO as a fraction of 30 kHz SCS: 100 ppb = 0.012 (ICI SIR about 35 dB); 500 ppb = 0.058 (about 19 dB); 2 ppm = 0.23 (unusable). A frequency-domain DU-side rotation cannot remove ICI
- Cao et al. (JSAC 2023), COTS RRUs with PTP+SyncE: LO phase wander +/-30 deg; 5 ms delay gives >10 deg error for 5% of channels; JP-RZF spectral-efficiency loss 46% at 5 ms and 64% at 10 ms
- Phase-error tolerance: 15 deg is the cell-free threshold cited by XGMF 2026 and Cao et al. 2-TRP single-user loss: 0.07 dB at 15 deg, 0.3 dB at 30 deg. Multi-user ZF SINR ceiling of about 1/sigma^2: 21 dB at 5 deg, 15 dB at 10 deg
- Timing residual: 1 ns gives 18 deg phase error at the edge of a 100 MHz band (+/-50 MHz)
- Wireless-only prior art (Merlo/Nanzer, arXiv 2506.07267): about 60-70 ps time/phase precision, more than 99% median coherent gain, 3.73 ppb RMSE frequency, COTS SDRs, no cables or GNSS
- Aerial cuPHY: FAPI SLOT.indication 3 slots ahead, 500 us L2 budget, 64T64R SRS-based RZF beamforming, PTRS not supported
- 6G timeline: SID RP-251881 (RAN#108, June 2025); TR 38.914 v1.0.0 at TSG#112 (June 2026); WG study to about June 2027; Rel-21 specs about 2027-2028

Open:

- Network-side, standard-transparent, PREDICTIVE per-RU LO phase/frequency tracking for TDD multi-user CJT under the O-RAN 7.2x weight-computation latency (about 1.5-4 ms horizon). Cao et al. identify exactly this ('LO phase tracking') as future work after measuring 46-64% SE loss at 5-10 ms calibration delay
- How a network-side SRS loop compares with the Rel-19 UE-assisted CJTC reports (quantisation of 24/11.6 deg, 4.6-75 ns, 0.4-13 ppb, plus CSI-report latency). No public head-to-head evaluation was found
- Realistic stochastic models of RU LO phase wander under PTP/SyncE (PLL phase noise plus random-walk wander of about +/-30 deg) for simulators like Sionna. Most D-MIMO papers assume either perfect sync or a constant CFO
- 6GR has not decided how inter-TRP sync is handled. RAN1 6GR CSI work (2026) uses the Rel-18 CJT codebook as baseline and mentions 'uncalibrated radios and phase errors', but no inter-TRP sync agreement was found, and RP-251881 has no D-MIMO objective. Contributions on network-side calibration RS and timelines could still matter
- Whether Aerial cuPHY can form one cell from several O-RUs with joint precoding (undocumented); PTRS is not supported
- The per-symbol versus per-slot compensation trade-off, and the residual-delay (sub-ns) estimation needed for wideband (100-200 MHz) CJT at 3.5-7 GHz

Red flags:

- Heavy prior art: AirSync, MegaMIMO, Rogalin 2014, BeamSync, Cao et al. JSAC 2023 (COTS O-RAN RRUs), Merlo/Nanzer 2025 (60-70 ps, 3.73 ppb, wireless-only), plus Rel-19 CJTC. 'Digital PLL across radios' is not new in itself. Novelty must be the predictive GPU-DU loop under O-RAN timing for multi-user CJT, and the evaluation against Rel-19 reporting
- 'Cheap free-running XO/TCXO without shared references' is NOT viable in an O-RAN 7.2x split. The iFFT runs in the RU, so a DU-side frequency-domain correction cannot remove ICI. 0.5-2 ppm gives CFO/SCS of 0.06-0.23 at 30 kHz (ICI SIR about 19 dB or worse), and TS 38.104 requires +/-50 ppb anyway. RUs still need SyncE/PTP or OTA frequency discipline; the GPU loop can only handle phase and residual ppb-level frequency
- If only single-user CJT is evaluated the gains look trivial (15 deg costs 0.07 dB for 2 TRPs). The benefit appears only with multi-user / cell-free spatial multiplexing (SINR ceiling of about 1/sigma^2)
- Industry scepticism: Rel-18 multi-TRP trials showed 'only marginal gains' and industry is 'slow, if even reluctant' on D-MIMO (Shafi/Larsson/Parkvall/Toskala). Fronthaul/backhaul cost may dominate sync as the adoption barrier
- Aerial cuPHY does not document CJT or multi-RU joint precoding, and real Aerial needs supported RU hardware. Rented vast.ai GPUs can only run the simulation (Sionna), not an over-the-air Aerial test
- Source caveats: the MIMO Ph5 WID number (RP-234007, revision RP-242394) and the Rel-20 MIMO Ph6 scope come from secondary summaries, not from the 3GPP portal or RAN1 chair notes; curl to 3GPP/ETSI PDFs was bot-blocked. The cjtc-F mapping as rendered appears non-negative, so check the sign convention in the official TS 38.214 v19.x. The 6GR CSI status comes from a ShareTechnote tdoc digest, and my per-meeting attribution of RAN1#124bis-#126 has not been verified against chair notes
- The 1/sigma^2 SINR ceiling and the ICI formula (pi*eps)^2/3 are rules of thumb, not cited results. Verify them in simulation before quoting

### crystals-phase-noise

- Frequency tolerance: XO +-10..50 ppm (+-20 ppm typical); TCXO +-0.5..2.5 ppm (USRP X310 internal reference 2.5 ppm); OCXO +-10..50 ppb after aging; X310 GPSDO 20 ppb not locked (spec sheet, quoted elsewhere as +-25 ppb), +-0.01 ppb locked; CSAC SA65 +-5e-11 at shipment, +-5e-10 retrace, aging <9e-10/month; 3GPP TS 38.104 base station +-0.05 ppm (wide area), +-0.1 ppm (medium/local area); ITU-T G.8262 SyncE clock free-run 4.6 ppm
- ADEV at 1 s: TCXO 2e-11..1e-9 (good 10 MHz parts 2e-10..2e-11); OCXO 2e-12 (Abracon O-CDF), 5e-12 (MtronPTI XO5123); CSAC SA65 3e-10 (1 s), 1.5e-10 (10 s), 3e-11 (100 s); my model at 1 ms / 10 ms / 100 ms / 1 s (fh = 10 kHz): XO 5.9e-10 / 3.0e-10 / 4.1e-10 / 5.3e-10; TCXO 2.6e-10 / 7.5e-11 / 7.3e-11 / 9.3e-11; OCXO 1.0e-10 / 1.4e-11 / 5.4e-12 / 5.3e-12; CSAC 1.4e-9 / 1.7e-9 / 1.9e-9 / 1.1e-9 (the model's 1 s CSAC value is about 4x above the 3e-10 datasheet spec, so it is conservative)
- 10 MHz reference phase noise (dBc/Hz at 1 Hz / 10 Hz / 100 Hz / 1 kHz / 10 kHz / 100 kHz): CSAC SA65 <-44 / <-64 / <-110 / <-128 / <-135 / <-140 (datasheet); good TCXO about -65 / -100 / -125 / -142 / -150 / -155; low-noise OCXO about -90 / -120 / -140 / -150 / -155 / -160 (KYOCERA AVX KLNE gives -120 at 10 Hz)
- PLL synthesizer: LMX2594 -110 dBc/Hz at 100 kHz offset on a 15 GHz carrier, 45 fs RMS jitter at 7.5 GHz (100 Hz-100 MHz), figure of merit -236 dBc/Hz, normalised 1/f -129 dBc/Hz; in-band noise = FOM + 10log(f_PD) + 20log(N) = about -125 dBc/Hz at 3.5 GHz with a 100 MHz phase-detector frequency; reference noise rises by 20log(fc/fref): +50.9 dB (3.5 GHz), +69 dB (28 GHz), +83 dB (140 GHz) from 10 MHz
- Carrier phase noise in my model (dBc/Hz at 1k / 10k / 100k / 1M Hz): FR1 TCXO -91 / -99 / -104 / -131; FR1 OCXO -99 / -104 / -109 / -134; FR2 TCXO -73 / -81 / -86 / -89; FR2 OCXO -81 / -86 / -91 / -94; 140 GHz OCXO -67 / -72 / -77 / -80 (pessimistic; D-band literature: -121.6 dBc/Hz at 1 MHz at 150 GHz, -108 dBc/Hz at 10 MHz at 140 GHz). 3GPP TR 38.803 29.55 GHz model: -67 / -87 / -97 / -99 (and -109 at 10 MHz)
- Integrated jitter per radio, 1 kHz-10 MHz: FR1 0.29 deg (TCXO) / 0.16 deg (OCXO); FR2 4.5 / 2.5 deg; 140 GHz 22 / 13 deg; TR 38.803 29.55 GHz model 2.2 deg; USRP X310 (SBX) 1.0 deg at 3.5 GHz, 1.5 deg at 6 GHz
- Drift caused by frequency offset, as deg/ms at 3.5 / 28 / 140 GHz: two XOs 40 ppm 5.0e4 / 4.0e5 / 2.0e6; two X310 TCXOs 5 ppm 6.3e3 / 5.0e4 / 2.5e5; 3GPP bound 0.1 ppm 126 / 1008 / 5040; 20 ppb (two unlocked GPSDOs) 50 / 403 / 2016; 10 ppb 12.6 / 101 / 504; 1 ppb 1.26 / 10.1 / 50.4; 0.39 ppb (Rel-19 CJTC-F finest step) 0.49 / 3.9 / 19.7; 0.2 ppb 0.25 / 2.0 / 10.1; 0.02 ppb (two locked GPSDOs) 0.025 / 0.20 / 1.0
- Time for an uncorrected frequency offset to reach 10 / 30 deg: 100 ppb takes 79 / 238 us at 3.5 GHz and 9.9 / 30 us at 28 GHz; 1 ppb takes 7.9 / 23.8 ms at 3.5 GHz, 0.99 / 2.98 ms at 28 GHz, 0.20 / 0.60 ms at 140 GHz. Residual frequency error after estimation must therefore be below about 1.6 ppb (FR1) or 0.2 ppb (FR2) to stay within 10 deg over a 5 ms update period
- Measured: COTS RRUs at 4.9 GHz with PTP plus SyncE drifted >20 deg in 7.5 ms and stayed within about +-30 deg over 250 ms (Cao 2022); USRP B210s on internal clocks had 150-350 Hz frequency offset, more than 180 deg in 10 ms, falling to 0-0.5 Hz (about 0.2 ppb) with a shared OctoClock (RFClock); two GPSDOs: +-500 ns pulse-per-second error and 0-100 ns relative clock wander (RFClock); wireless distributed-array calibration reached 60-70 ps time/phase precision and 3.73 ppb frequency RMS error, with median coherent gain above 99% (Merlo)
- Rms phase error between two free-running radios (frequency offset tracked, zero latency) at update interval T = 0.125 / 1 / 5 / 10 / 20 / 100 ms. FR1 TCXO hold: 0.6 / 0.7 / 1.5 / 2.6 / 5.0 / 23.9 deg; linear: 0.8 / 0.9 / 1.1 / 1.4 / 2.1 / 10.8. FR1 XO hold: 0.8 / 1.7 / 7.2 / 14.1 / 27.8 / 134. FR2 TCXO hold: 8.9 / 9.4 / 14.0 / 22.4 / 41.0 / 191. FR2 OCXO hold: 5.1 / 5.1 / 5.2 / 5.3 / 5.8 / 12.3. 140 GHz OCXO hold: 25.4 / 25.6 / 26.0 / 26.7 / 28.8 / 61.5. 140 GHz TCXO hold: 44.5 / 46.8 / 70 / 112 / 205 / 956
- Shared frequency (SyncE-like, relative noise kept only above a 10 Hz bandwidth), hold, T = 1 / 10 / 100 ms: FR1 TCXO 0.6 / 1.0 / 1.9 deg (levels off at 2.0); FR1 XO 1.1 / 3.7 / 9.5 (levels off at 9.9); FR2 TCXO 9.2 / 11.3 / 17.2; FR2 XO 13.9 / 31.3 / 77.0; FR2 OCXO about 5.1-5.4
- Longest update interval that keeps the RMS error between two radios below 10 / 30 deg (model): FR1 XO 6.1-19.8 / 19.8-52.5 ms; FR1 TCXO 35.6-77.6 / 115-206 ms; FR1 OCXO 807-1190 ms / >2.6 s; FR1 CSAC 1.9-5.1 / 6.1-11 ms; FR2 TCXO 1.57 ms (hold only) / 13.4-35.6 ms; FR2 OCXO 64-94 / 250-449 ms; FR2 XO never / 1.9-6.1 ms; 140 GHz never for any 10 MHz-reference chain in the model
- Coherent combining loss (dB) for independent Gaussian per-transmitter error sigma, N = 2 / 4 / 8: 5 deg 0.02 / 0.02 / 0.03; 10 deg 0.07 / 0.10 / 0.12; 20 deg 0.26 / 0.39 / 0.46; 30 deg 0.55 / 0.86 / 1.02; 45 deg 1.14 / 1.84 / 2.24; 60 deg 1.76 / 3.01 / 3.80; fully random 3.0 / 6.0 / 9.0. Uniform error within +-30 deg: 0.20 / 0.30 / 0.35; within +-90 deg: 1.53 / 2.57 / 3.19
- Rough interference ceiling for zero-forcing / multi-user joint transmission: 21.2 dB (5 deg), 15.1 dB (10 deg), 8.9 dB (20 deg), 5.0 dB (30 deg)
- Wiener linewidth from the phase noise level: beta = 2*pi*f^2*10^(L/10); L(100 kHz) = -80 / -90 / -100 / -110 dBc/Hz gives beta = 628 / 62.8 / 6.28 / 0.63 Hz and an RMS walk after 1 ms of 114 / 36 / 11.4 / 3.6 deg per radio. This applies only to free-running VCOs, not to PLL-locked carriers
- 3GPP timing figures: Rel-19 CJTC-Dd delay-offset range CP/2 or CP; MIMO time alignment error (TAE) 65 ns in TS 38.104 (from memory; my search did not confirm it). A timing error dt causes a phase slope across the band: keeping it under 10 deg at the band edge needs relative timing stability of about 0.56 ns over 100 MHz (FR1) and 0.14 ns over 400 MHz (FR2) between updates. G.8262 SyncE jitter generation 0.5 UIpp at 10G is about 48 ps peak-to-peak, roughly 60 deg at 3.5 GHz or 480 deg at 28 GHz if the radio's PLL does not clean it up

Open:

- Realistic simulation models for phase drift between TRPs from one slot to the next. The 3GPP models cover only phase noise within a symbol and are flat close in. Published distributed-MIMO studies mostly use a single Wiener process. There is no open model that combines reference random-walk and flicker FM, the PLL transfer, and SyncE/PTP disciplining, calibrated against measured COTS drift (e.g. Cao's >20 deg per 7.5 ms). Neither Sionna nor 3GPP provides one.
- FR2 multi-user CJT with TCXO-grade radios. My numbers put it right at the edge: 10 deg needs updates every 1.6 ms or less with near-zero latency, and multi-user precoding needs about 5 deg. Prediction that accounts for latency (a Kalman filter whose prior comes from oscillator physics, plus a frequency state) and choosing the update rate to fit TDD/SRS opportunities is a narrow open question with numbers attached.
- How network-side continuous estimation (from SRS/DMRS, on the GPU) compares with the Rel-19 UE-reported CJTC-P/F quantisation (24 or 11.6 deg phase steps, 0.39-13.3 ppb frequency steps) and its reporting delay. I found no published head-to-head comparison.
- Whether reciprocity holds when TX and RX use separate synthesizers (e.g. USRP SBX/UBX daughterboards, the AD9361). Phases measured on the uplink may not apply to the downlink without a calibration after every retune. Cao found the phase between antennas inside one RRU stable, but the general case for cheap radios is less clear.
- At sub-THz frequencies, whether any slot-rate loop can work, or whether a shared reference or LO (RF-over-fiber, a wireless reference tone) is unavoidable. In my model fast jitter alone gives 25-45 deg between two radios. This is model-dependent, so I would treat it as an open hardware question rather than a settled one.

Red flags:

- The direction is crowded with prior art: Ngo and Larsson 2025/26 (Kalman over-the-air phase tracking in the TDD flow), Cao 2022 and Jiang 2026 (real COTS RRU testbeds at 4.9 and 26 GHz), Merlo (wireless calibration of time, frequency and phase), RFClock, BeamSync, AirSync, Chorus, MegaMIMO, and 3GPP Rel-19 CJTC UE reporting. Calling it 'revolutionary' would be an overclaim. It is an incremental systems or measurement contribution at best.
- My model numbers depend on assumptions: a 10 MHz reference, LMX2594-class FOM, PLL bandwidth of 200 kHz (FR1) or 1 MHz (FR2+), and typical rather than datasheet-guaranteed reference phase noise for XO/TCXO/OCXO. Measured COTS drift (Cao) is about 10-20x worse at FR1. The 140 GHz results are pessimistic compared with state-of-the-art D-band PLLs. Use the tables for orders of magnitude, not as final numbers.
- The 3GPP TR 38.803 parameters I evaluated (Fp1 = 1 Hz) come from secondary sources (arXiv 2412.05841 and MathWorks), not the TR itself. The '65 ns MIMO TAE' and the G.8262 MTIE wander values are from memory and unverified. Check them against the original documents before quoting.
- A Wiener-only phase model, as in Ngo-Larsson style analyses, applied to PLL-locked radios would overstate millisecond-scale drift by more than 10x. 3GPP pole-zero models alone would understate it, because they have no close-in drift. Either mistake would make the simulation results meaningless.
- Combining-loss results assume independent Gaussian errors and ideal channel knowledge. Channel aging from UE mobility, delay and timing offsets (which need delay estimation per subband, cf. CJTC-Dd) and reciprocity mismatch between TX and RX chains are not included and may dominate in practice.
- The CSAC and GPSDO 'upgrades' do not solve carrier-phase coherence. CSAC short-term stability is worse than a TCXO, and two GPSDOs wander tens of ns relative to each other. Any proposal arguing 'cheap XO plus tracking replaces GPSDO/CSAC' has to make the comparison on short-term phase, where GPSDO and CSAC were never the right baseline.

### feasibility-test

- Kill-test results (measured with /tmp/.../phasesync/killtest.py; N=4, D=4 slots). FR1 3.5 GHz, ±50 ppb, 5 ms SRS: no loop -5.66 dB / 103 deg / ZF ratio 0.28; loop -0.01 dB / 4.0 deg / 0.99.
- FR1, tight PTP cluster ±2 ppb: 5 ms SRS, no loop -0.05 dB / 10.3 deg / 0.89, loop 2.9 deg / 0.99. 20 ms SRS, no loop -0.39 dB / 28.9 deg / 0.77. 80 ms SRS, no loop -2.9 dB / 83 deg.
- FR3 13 GHz, ±2 ppb: 5 ms SRS, no loop -0.68 dB / 38 deg / ZF 0.58, loop 6.0 deg / 0.96. 20 ms SRS, no loop -3.2 dB / 85 deg / 0.37-0.40, loop with averaged starting CFO 6.9 deg / 0.94.
- FR2 28 GHz, 2.5 ms SRS: loop floor 9.9-10.1 deg, ZF ratio 0.89-0.90, coherent gain -0.07 dB. This floor is set by the modelled synthesizer jitter, not by the loop.
- Free-running TCXO ±2 ppm at 3.5 GHz is 7 kHz of CFO (0.23 of the 30 kHz SCS, ICI SIR about 12 dB). ±50 ppb is 175 Hz, ICI SIR about 40 dB. Phase accumulates at 126 deg/ms for a 2x0.05 ppm relative offset at FR1 (from a sibling worker's oscillator budget, /tmp/.../phasesync/osc/out.txt).
- SRS periodicities allowed by 3GPP: 1, 2, 4, 5, 8, 10, 16, 20, 32, 40, 64, 80, 160 ... 2560 slots. The loop is safe up to about 20 ms with good starting-CFO acquisition and fails (cycle slips) at 80 ms in this model.
- GPU budget (/tmp/.../phasesync/gpu_budget.py). MSc testbed (4 TRPs, 20 MHz): under 1.1 GFLOP/s in total, so a CPU is enough. 4 TRP x 4 antennas, 100 MHz: SRS estimation 18 GFLOP/s, loop 1.7, RZF 2.5, PDSCH precoding 47. 64 ports, 16 layers, 48 SRS UEs: 443 / 40 / 161 / 751 GFLOP/s. 256 ports, 0.25 ms slot: 4.7 / 0.43 / 4.9 / 12 TFLOP/s. For reference: one CPU core 50-100 GFLOP/s, A100 19.5 TFLOP/s FP32.
- Hardware cost: B210 $2,387 (Digilent). X310 $11,462 plus UBX-160 $2,645 each. Minimum rig 3x B210 is about $7.2k (zero if the lab has them), plus SMA cables, 20-30 dB attenuators and a 4:1 combiner, about $100-300.
- Cloud cost: vast.ai A100 80GB about $0.43-1.50/h, RTX 4090 about $0.29-0.59/h. A full Sionna sweep is roughly 2-10 GPU-hours, about $1-15. The numpy kill test needs no GPU.
- Coherent-gain loss for random phase error of rms sigma (large N): 10log10(exp(-sigma^2)), i.e. 10 deg about -0.13 dB, 30 deg about -1.2 dB. ZF leakage gives an SINR ceiling of about -20log10(sigma_rad): 10 deg about 15 dB, 5 deg about 21 dB.

Open:

- What real hardware does over the time horizons that matter. Relative-phase traces between independent cheap oscillators (B210 TCXOs) at 1-100 ms, fed through the loop, give residual error vs SRS period and latency. Most papers report a single operating point, not the trade-off curve.
- Where the bottleneck moves from software to hardware: the carrier frequency (FR1 -> FR3 7-24 GHz -> FR2) at which the error after an ideal slot-rate loop is set by synthesizer integrated jitter. This tells you whether 6G FR3 CJT needs better PLLs or better tracking, and it ties straight to the PLL interest.
- Pooling across UEs: the TRP oscillator term is common to every UE's SRS, while UE Doppler and channel change are per-UE. A GPU estimator that pools all UEs, ports and REs each slot could separate TRP drift from UE motion and allow longer SRS periods. I have not tested this; mobility was not modelled.
- Cycle-slip-free frequency acquisition and tracking at long SRS periods (>=40-80 ms) and with XO-class oscillators at FR3. The simple loop failed here.
- Interaction with TX/RX reciprocity calibration drift (AD9361 TX and RX LOs are separate synthesizers) and with SFO when the sample clock and LO share one free-running crystal.

Red flags:

- Heavy prior art: Rel-19 CJTC, Ngo & Larsson 2025-26, Merlo/Nanzer 2025, RFClock, MegaMIMO/AirShare, Quitin/Madhow 2013, Cao 2022, Jiang 2026. A plain "digital PLL across radios" is not novel. The novelty has to be the measured trade-off, FR3/PLL-floor analysis, or multi-UE pooled GPU tracking.
- The dramatic -6 dB no-loop result is a straw man. Commercial TDD CJT already tracks frequency, and a tight PTP/SyncE cluster (±2 ppb) at FR1 with 5 ms SRS loses only 0.05 dB with no tracking at all. The real problem appears only at FR3/FR2, long SRS periods or loose frequency sync.
- My simulation is simplified: single antenna per TRP, static UE, no SFO or timing error, perfect reciprocity calibration, white PM scaled linearly with fc, a crude alpha-beta grid (not an optimal Kalman), and oscillator noise coefficients (a_wfm=7.5e-13, b_rwfm=1e-10) that are assumed rather than measured. The cycle-slip failures partly reflect my crude coarse-CFO starting step.
- The GPU is not necessary for the loop; claiming that would be overselling. For a 4-radio testbed a CPU handles everything (under 1.1 GFLOP/s).
- USRP pitfalls: B210 channels share LOs (one TRP per B210). X310 daughterboards share one reference. AD9361 LO phase re-randomises on retune. Free-running sample clocks drift 2 us/s at 2 ppm, beyond the 2.3 us CP at 30 kHz within about 1 s unless resampled or burst-retimed. Without PPS, device times are offset by host latency. OTA work needs ISM bands or a licence; cabled is cleaner for the kill test.
- Lab USRPs are unconfirmed. Without them the hardware stages cost about $7k (3x B210), and only the numpy/Sionna stages are $0-15.
- OAI and srsRAN cannot run multi-TRP CJT with independent radios, so this will not be a full 5G-stack demo within an MSc timeline.

## Sources

- Cao et al., COTS RRU cell-free OTA calibration and phase sync (JSAC 2023), verified locally: +/-30 deg wander, >20 deg at 7.5 ms, 46%/64% loss at 5/10 ms, Remark 2 on PLL drift and LO phase tracking: https://arxiv.org/abs/2208.14048
- Jiang et al., mmWave cell-free bidirectional phase-coherent transmission with UL-SRS calibration phase tracking (2026), verified locally: >90 deg without tracking, <20 deg with: https://arxiv.org/abs/2601.14648
- Merlo, Wagner, Lancaster, Nanzer, real-time digital wireless time/frequency/phase calibration on COTS X310s (2025), verified locally: host misses real-time deadlines, FPGA as future work: https://arxiv.org/abs/2506.07267
- Ngo & Larsson, distributed MIMO with OTA phase calibration integrated into the TDD flow (Kalman): https://arxiv.org/abs/2509.03722
- AirSync (Balan et al. 2012): https://arxiv.org/abs/1205.6862
- MegaMIMO 2.0, real-time distributed MIMO (SIGCOMM 2016): https://people.csail.mit.edu/rahul/papers/rtmegamimo-sigcomm2016.pdf
- Chorus (SIGCOMM 2018): https://dl.acm.org/doi/10.1145/3230543.3230578
- BeamSync (Larsson group): https://arxiv.org/abs/2311.11070
- RFClock (MobiCom 2021): https://genesys-lab.org/papers/RFCLOCK_MOBICOM2021.pdf
- Rel-19 overview, inter-TRP time/frequency/phase offset reporting for CJT: https://arxiv.org/pdf/2312.15174
- TS 38.214 Rel-19 CJTC-P phase report (mirror): https://itecspec.com/3gpp/38.214/s/5.2.1.4.11
- TS 38.214 Rel-19 CJTC-F frequency report (mirror): https://itecspec.com/3gpp/38.214/s/5.2.1.4.10
- UE-assisted inter-TRP calibration for CJT patent application: https://patents.justia.com/patent/20250119258
- Qualcomm Rel-18 deck (CJT assumes ideal sync): https://www.qualcomm.com/content/dam/qcomm-martech/dm-assets/documents/a-closer-look-at-5g-advanced-release-18-web.pdf
- 3GPP TR 38.914 / 6G study news: https://www.3gpp.org/news-events/3gpp-news/6g-38914
- Shafi, Larsson, Lin, Parkvall, Toskala et al., industrial viewpoints on 6G RAN (D-MIMO phase sync as a main challenge): https://arxiv.org/abs/2508.08225
- XGMF Advanced MIMO white paper v2.0 (15 deg threshold, adoption limited by sync): https://xgmf.jp/pdf/2026/Beyond-5G-White-Paper-A-MIMO_v2.0.pdf
- RANBooster (SIGCOMM 2025, assumes PTP-synced RUs are coherent): https://www.microsoft.com/en-us/research/wp-content/uploads/2025/07/RANBooster.pdf
- Ericsson, 5G synchronization requirements: https://www.ericsson.com/en/reports-and-papers/ericsson-technology-review/articles/5g-synchronization-requirements-and-solutions
- O-RAN synchronization considerations (ATIS WSTS 2024): https://cdn.atis.org/wsts.atis.org/2024/05/03190654/05_ORAN-Synchronization-Considerations_Armstrong_FINAL.pdf
- NVIDIA Aerial cuPHY features (SRS-based RZF BFW, no CJT documented): https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/features/cuphy_features.html
- NVIDIA Sionna (2.2.0 has no phase-noise/CFO block): https://github.com/NVlabs/sionna
- Bondada, Jakubisin, Buehrer, sync offsets and CSI delay in D-MIMO: https://arxiv.org/abs/2503.24314
- MATLAB nrPhaseNoise (TR 38.803 models, for cross-checking): https://www.mathworks.com/help/5g/ref/nrphasenoise-system-object.html
- USRP B210 (price, +/-2 ppm): https://digilent.com/shop/ni-ettus-usrp-b210-2x2-70mhz-6ghz-sdr-cognitive-radio/
- TI LMX2594 PLL datasheet: https://www.ti.com/lit/ds/symlink/lmx2594.pdf
- Local kill-test scripts and outputs: /tmp/claude-0/-home-user/c22b0607-07f1-5108-a24d-4caf6e85f472/scratchpad/phasesync/killtest.py, killtest3_out.txt, gpu_budget.py, osc/osc_budget.py

Scripts from the feasibility track (numpy kill test of slot-rate phase loops vs no loop, GPU compute budget, oscillator budget) are in `scripts/`. They are models, not measurements.
