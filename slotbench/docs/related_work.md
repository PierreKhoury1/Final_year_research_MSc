# Related work and gap (checked 2026-10-01)

Summaries from a literature sweep. Only the YinYangRAN preprint was read in full; spot-check the other
numbers against the papers before citing them.

## Closest work

| Work | Setup | Mechanisms | Tail reporting | Key result |
|---|---|---|---|---|
| **Weaver**, Xue et al., arXiv 2609.35276 (Sep 2026) | LDPC decode on DGX Spark; OPT training on L40S; 8 sites (128-512 simulated) | green contexts vs MPS vs MIG | deadline-met %, p95/p99 network delay | 40-85% SMs idle; 99.8% HARQ deadlines met under 1 ms; 2.1-3.7x training throughput; green contexts reconfigure in us vs MPS ~300 ms, MIG ~7 s |
| **YinYangRAN**, Lo Schiavo et al., IEEE INFOCOM 2024 ([preprint](https://neclab.eu/fileadmin/user_upload/YinYangRAN_Resource_Multiplexing_in_GPU-Accelerated_Virtualized_RANs_pre-print.pdf)) | one A100, commercial GPU FEC/LDPC, 100 MHz DU; TES-RNN forecaster as ML neighbour | MPS SM split; MIG discussed | reliability quantiles | naive sharing ~50% meets 1 ms; MPS split trade-off curve; MIG reconfig 6.9 s vs MPS 0.27 s; learned controller +50% reliability |
| **Aerial ISAC**, Villa et al., arXiv 2512.06493 | full Aerial cuPHY L1 + AI dApp on GH200 and DGX Spark | MPS vs MIG | AI app latency only | MPS: memory contention with cuPHY; MIG removes it (~1.8x lower AI latency); no L1 miss rates |
| **NVIDIA / SoftBank AITRAS** (industry) | GH200, 20 x 100 MHz cells per server | whole-GPU or MIG, static splits | none published | "carrier-grade" RAN with AI alongside; utilisation up to 100% vs ~30% |
| CAORA (arXiv 2503.07420); Li et al. (arXiv 2605.07547) | simulation | MIG allocation by RL / LLM agent | n/a | no real GPU timing |
| Martin et al., arXiv 2601.07600 | A100, Jetson Orin, inference | MPS vs MIG vs green contexts | not RAN | MIG isolates best; green contexts cheap but no memory isolation |
| REEF, Han et al., OSDI 2022 | AMD GPU, DNN inference | reset-based us preemption | not RAN | real-time inference with <2% overhead |

Context, not GPU: Concordia (SIGCOMM 2021) and Nuberu (MobiCom 2021) set the 99.999% / worst-case framing
for CPU vRAN. Unverified: an MDPI Telecom 2026 tail-latency study of GPU LDPC decoding alone.

## Already shown elsewhere (do not claim as new)

- Naive GPU sharing breaks PHY deadlines; MPS SM partitioning protects the RAN at an ML throughput cost.
- MIG isolates well but reconfigures in seconds; green contexts are the fast, fine-grained option.
- MPS leaves memory-subsystem contention that MIG removes.

## What this project can claim

1. Consumer GPU (RTX 3060) vs A100/H100: all prior work uses datacenter or Grace-class parts.
2. Worst-case statistics over ~1e6 slots (p99.99, max, consecutive-miss bursts) for a full uplink
   pipeline (FFT, channel estimation, MMSE, demod, LDPC), not LDPC alone.
3. Isolation depends on the neighbour: MPS protects against LLM-decode and vision neighbours but not
   back-to-back GEMM (first RTX 3060 results).
4. Mechanisms prior RAN work did not measure: cross-process stream priority (ineffective), the ~2 ms
   time-slice quantum as the cause of misses, CUDA graphs vs streams.
5. CPU-GPU synchronisation view: clocks agree to ~1.2 us, but the GPU start time slips by ~2 ms under
   co-location; MPS restores it to ~4 us at the cost of slower execution.
6. One open harness covering seven mechanisms with the AI throughput cost of each.

## How to differentiate further

- Make neighbour-dependence the central result: add a memory-bandwidth hog and explain which shared
  resource (SMs, L2/DRAM bandwidth, launch queue, time slice) causes misses under each mechanism.
- Report tails and miss bursts as primary metrics, with a Pareto frontier per mechanism and GPU class.
- Test green contexts and MPS limits on the 3060 against MIG on A100; measure reconfiguration cost
  directly instead of citing Weaver/YinYangRAN.
