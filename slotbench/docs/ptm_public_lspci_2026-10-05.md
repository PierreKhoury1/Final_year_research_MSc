# Do NVIDIA GPUs advertise PCIe PTM? Public-dump survey (2026-10-05)

Source: github.com/linuxhw/LsPCI (public `lspci -vvnn` dumps from hw-probe), commit HEAD on 2026-10-05.
Scanned all 4830 files under `Server/` with `tools/lspci_ptm_scan.py`, plus two notebook dumps by hand.

| | count |
|---|---|
| NVIDIA GPU entries (VGA / 3D controller) | 1184 |
| ... with extended config space visible (root-run lspci) | 685 |
| ... advertising Precision Time Measurement (ext cap 0x001F) | **0** |

Includes A100 PCIe 40/80 GB (21, all ext-visible), A100 SXM4, A30, A40 (16), H100 SXM5 (4, HPE Cray XD665),
RTX 6000 Ada, RTX 4090, T4, V100, P100. Example H100 capability list: Secondary PCIe, LTR, Resizable BAR,
Data Link Feature, 16 GT/s PHY, AER, Lane Margining, ARI, SR-IOV, Power Budgeting, DOE, DSN. No PTM.

Positive control: the same parser finds PTM on 143 Broadcom BCM57416 NICs, 27 BCM57412 NICs and hundreds of
Intel root ports in the same dumps, so the absence on GPUs is not a parsing artifact.

Not covered: H100 PCIe/NVL, H200, L40/L40S, Blackwell (none with ext caps in this set). A hidden capability
cannot be ruled out, but nothing public shows one. Conclusion: GPU-side PTM is not available on these parts;
host-side PCIe timestamp references must come from a PTM-capable endpoint (e.g. a NIC) instead.
