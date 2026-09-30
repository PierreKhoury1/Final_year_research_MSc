// Builds the findings deck from ../clk/summary.json (all numbers come from the measured data).
const pptxgen = require("pptxgenjs");
const fs = require("fs");
const S = JSON.parse(fs.readFileSync("../clk/summary.json", "utf8"));
const D = S.default, T = S.tuned;

const INK = "0E1A2B", INK2 = "1B2B44", AMBER = "F2A93B", TEAL = "2A9D8F", ROSE = "D1495B";
const TEXT = "1F2937", MUTED = "6B7280", LIGHT = "F3F5F8", WHITE = "FFFFFF", ICE = "C9D6EA";
const HF = "Cambria", BF = "Calibri";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9"; // 10 x 5.625
pres.title = "Does the GPU know what time it is?";

const PH = ["tight", "sleepy", "cpu_load", "gpu_load", "tight2"];
const PHL = { tight: "Tight loop", sleepy: `Sleepy (~${Math.round(D.phases.sleepy.gap_ms)} ms gaps)`, cpu_load: "CPU load", gpu_load: "GPU load", tight2: "Tight loop (again)" };
const us = (ns) => ns / 1000;
const fmtUs = (ns) => { const u = ns / 1000; return u >= 1000 ? (u / 1000).toFixed(1) + " ms" : u >= 10 ? u.toFixed(0) + " µs" : u.toFixed(2) + " µs"; };
const best = Math.min(...PH.map((p) => Math.min(D.phases[p].p1, T.phases[p].p1)));

function title(s, text, dark) {
  s.addText(text, { x: 0.5, y: 0.3, w: 9, h: 0.75, fontFace: HF, fontSize: 30, bold: true, color: dark ? WHITE : INK, margin: 0, isTextBox: true, valign: "middle" });
}
function note(s, text, dark) {
  s.addText(text, { x: 0.5, y: 5.1, w: 9, h: 0.3, fontFace: BF, fontSize: 10, color: dark ? ICE : MUTED, margin: 0, isTextBox: true });
}
function badge(s, n, x, y, color) {
  s.addShape(pres.shapes.OVAL, { x, y, w: 0.46, h: 0.46, fill: { color: color || AMBER }, line: { color: color || AMBER } });
  s.addText(String(n), { x, y, w: 0.46, h: 0.46, fontFace: HF, fontSize: 16, bold: true, color: INK, align: "center", valign: "middle", margin: 0, isTextBox: true });
}
function stat(s, x, y, w, big, label, color, dark) {
  s.addText(big, { x, y, w, h: 0.8, fontFace: HF, fontSize: 40, bold: true, color: color || AMBER, margin: 0, isTextBox: true });
  s.addText(label, { x, y: y + 0.82, w, h: 0.7, fontFace: BF, fontSize: 13, color: dark ? ICE : TEXT, margin: 0, isTextBox: true, valign: "top" });
}
function card(s, x, y, w, h, fill) {
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill || LIGHT }, line: { color: fill || LIGHT } });
}
const axis = (extra) => Object.assign({
  catAxisLabelColor: MUTED, valAxisLabelColor: MUTED, catAxisLabelFontFace: BF, valAxisLabelFontFace: BF,
  catAxisLabelFontSize: 10, valAxisLabelFontSize: 10, valGridLine: { color: "E5E7EB", size: 0.5 }, catGridLine: { style: "none" },
  titleFontFace: BF, titleFontSize: 12, titleColor: TEXT, legendFontFace: BF, legendFontSize: 10,
}, extra);

// 1. Title
{
  const s = pres.addSlide(); s.background = { color: INK };
  s.addText("MSc Advanced Computing · project feasibility · 28 Sep 2026", { x: 0.6, y: 0.7, w: 8.8, h: 0.35, fontFace: BF, fontSize: 13, color: AMBER, margin: 0, isTextBox: true });
  s.addText("Does the GPU know what time it is?", { x: 0.6, y: 1.3, w: 8.8, h: 1.4, fontFace: HF, fontSize: 44, bold: true, color: WHITE, margin: 0, isTextBox: true });
  s.addText("Measuring CPU–GPU clock alignment on a laptop, and why the answer points to an OCXO time reference", { x: 0.6, y: 2.8, w: 8, h: 0.9, fontFace: BF, fontSize: 18, color: ICE, margin: 0, isTextBox: true });
  // tick-mark motif: a row of clock ticks
  for (let i = 0; i < 24; i++) {
    const tall = i % 6 === 0;
    s.addShape(pres.shapes.LINE, { x: 0.6 + i * 0.37, y: tall ? 4.35 : 4.5, w: 0, h: tall ? 0.45 : 0.3, line: { color: tall ? AMBER : "3B4F6E", width: tall ? 2.5 : 1.5 } });
  }
  s.addText("Pierre · King's College London", { x: 0.6, y: 4.95, w: 8, h: 0.3, fontFace: BF, fontSize: 12, color: ICE, margin: 0, isTextBox: true });
  s.addNotes("Feasibility experiment run on my own laptop before committing to hardware. Everything in the deck comes from measured data.");
}

// 2. Why it matters
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Why a GPU's sense of time matters");
  stat(s, 0.5, 1.4, 2.8, "±1.5 µs", "max time error between 5G TDD base stations (3GPP / ITU-T G.8271)", ROSE);
  stat(s, 3.6, 1.4, 2.8, "500 µs", "one 5G slot at 30 kHz subcarrier spacing: the deadline the processing must hit", TEAL);
  stat(s, 6.7, 1.4, 2.8, "GPU", "now runs the radio stack itself (e.g. NVIDIA Aerial), so the GPU's clock sits on the timing path", AMBER);
  card(s, 0.5, 3.45, 9, 1.35);
  s.addText([
    { text: "The question: ", options: { bold: true, color: INK } },
    { text: "when code on a GPU reads its clock, how close is that to true time, and how would we even check?", options: { color: TEXT } },
  ], { x: 0.8, y: 3.6, w: 8.4, h: 1.05, fontFace: BF, fontSize: 17, margin: 0, isTextBox: true, valign: "middle" });
  s.addNotes("Timing is the glue of 5G. If AI workloads share the GPU with the radio stack, the GPU's notion of time has to be trustworthy.");
}

// 3. Two meanings of synchronise
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "\"Synchronise\" means two different things");
  const col = (x, head, color, code, body) => {
    card(s, x, 1.35, 4.3, 2.9);
    s.addShape(pres.shapes.OVAL, { x: x + 0.3, y: 1.6, w: 0.36, h: 0.36, fill: { color }, line: { color } });
    s.addText(head, { x: x + 0.8, y: 1.55, w: 3.3, h: 0.45, fontFace: HF, fontSize: 19, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
    s.addText(code, { x: x + 0.3, y: 2.2, w: 3.7, h: 0.5, fontFace: "Courier New", fontSize: 13, color: INK, fill: { color: WHITE }, margin: 6, isTextBox: true, valign: "middle" });
    s.addText(body, { x: x + 0.3, y: 2.9, w: 3.7, h: 1.2, fontFace: BF, fontSize: 14, color: TEXT, margin: 0, isTextBox: true, valign: "top" });
  };
  col(0.5, "Execution sync", MUTED, "cudaDeviceSynchronize()", "\"Wait until the GPU has finished.\" Tells you work is done, not when it finished. The CPU notices late.");
  col(5.2, "Clock alignment", AMBER, "cpu_ns ≈ a · gpu_tick + b", "\"This GPU timestamp equals that CPU time.\" Needs offset b, rate a, and an honest error bound. This project is about this one.");
}

// 4. Setup + method diagram
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "The experiment: bracket every GPU clock read");
  // lanes
  const L = 0.5, R = 6.2;
  s.addText("CPU", { x: L, y: 1.55, w: 0.6, h: 0.3, fontFace: BF, fontSize: 13, bold: true, color: TEAL, margin: 0, isTextBox: true });
  s.addText("GPU", { x: L, y: 3.25, w: 0.6, h: 0.3, fontFace: BF, fontSize: 13, bold: true, color: AMBER, margin: 0, isTextBox: true });
  s.addShape(pres.shapes.LINE, { x: 1.1, y: 1.7, w: R - 1.1, h: 0, line: { color: TEAL, width: 2 } });
  s.addShape(pres.shapes.LINE, { x: 1.1, y: 3.4, w: R - 1.1, h: 0, line: { color: AMBER, width: 2 } });
  // t0 -> gpu read -> t1
  s.addShape(pres.shapes.LINE, { x: 1.7, y: 1.7, w: 1.6, h: 1.7, line: { color: INK, width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.LINE, { x: 3.3, y: 1.7, w: 1.6, h: 1.7, flipV: true, line: { color: INK, width: 1.5, endArrowType: "triangle" } });
  s.addShape(pres.shapes.OVAL, { x: 1.6, y: 1.6, w: 0.2, h: 0.2, fill: { color: TEAL }, line: { color: TEAL } });
  s.addShape(pres.shapes.OVAL, { x: 4.8, y: 1.6, w: 0.2, h: 0.2, fill: { color: TEAL }, line: { color: TEAL } });
  s.addShape(pres.shapes.OVAL, { x: 3.2, y: 3.3, w: 0.2, h: 0.2, fill: { color: AMBER }, line: { color: AMBER } });
  s.addText("t0: CPU reads TSC,\nwrites \"go\"", { x: 1.0, y: 1.0, w: 1.6, h: 0.55, fontFace: BF, fontSize: 11, color: TEXT, margin: 0, isTextBox: true, align: "center" });
  s.addText("t1: CPU sees \"done\",\nreads TSC", { x: 4.1, y: 1.0, w: 1.7, h: 0.55, fontFace: BF, fontSize: 11, color: TEXT, margin: 0, isTextBox: true, align: "center" });
  s.addText("GPU sees \"go\", reads its own clock, writes \"done\"", { x: 1.8, y: 3.6, w: 3.1, h: 0.5, fontFace: BF, fontSize: 11, color: TEXT, margin: 0, isTextBox: true, align: "center" });
  s.addShape(pres.shapes.RECTANGLE, { x: 1.7, y: 4.3, w: 3.2, h: 0.12, fill: { color: ICE }, line: { color: ICE } });
  s.addText("bracket = t1 − t0: the GPU read happened somewhere inside", { x: 1.0, y: 4.5, w: 4.6, h: 0.35, fontFace: BF, fontSize: 11, italic: true, color: MUTED, margin: 0, isTextBox: true, align: "center" });
  // spec card
  card(s, 6.5, 1.3, 3.0, 3.6);
  s.addText([
    { text: "Machine", options: { bold: true, color: INK, breakLine: true } },
    { text: "AMD Ryzen AI 7 350 laptop, on battery", options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "GPU", options: { bold: true, color: INK, breakLine: true } },
    { text: "Radeon 860M iGPU (gfx1152), OpenCL 2.0", options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Clocks compared", options: { bold: true, color: INK, breakLine: true } },
    { text: `CPU TSC ≈ ${D.tsc_mhz.toFixed(0)} MHz`, options: { breakLine: true } },
    { text: `GPU REALTIME ≈ ${D.gpu_mhz.toFixed(2)} MHz`, options: { breakLine: true } },
    { text: " ", options: { fontSize: 6, breakLine: true } },
    { text: "Samples", options: { bold: true, color: INK, breakLine: true } },
    { text: `${(D.n + T.n).toLocaleString("en-GB")} brackets` },
  ], { x: 6.75, y: 1.45, w: 2.6, h: 3.3, fontFace: BF, fontSize: 12, color: TEXT, margin: 0, isTextBox: true, valign: "top" });
  s.addNotes("Persistent GPU kernel and CPU thread ping-pong through fine-grained shared memory. No kernel launch inside the loop. GPU clock read via the RDNA 3.5 s_sendmsg_rtn GET_REALTIME message (the AMD equivalent of NVIDIA's %globaltimer).");
}

// 5. v1 vs v2 method
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Python launch vs native ping-pong");
  s.addChart(pres.charts.BAR, [
    { name: "Median bracket (µs)", labels: ["v1 Python, idle", "v1 Python, CPU load", "v2 native, tight", "v2 native, CPU load", "v2 native, best 1%"],
      values: [840.1, 232.7, +us(D.phases.tight.p50).toFixed(1), +us(D.phases.cpu_load.p50).toFixed(1), +us(best).toFixed(2)] },
  ], axis({ x: 0.5, y: 1.25, w: 5.6, h: 3.7, barDir: "bar", chartColors: [TEAL], showValue: true, dataLabelPosition: "outEnd",
    dataLabelFontSize: 10, dataLabelColor: TEXT, dataLabelFormatCode: "#,##0.##", valAxisLogScaleBase: 10, showLegend: false, showTitle: true, title: "Bracket width, µs (log scale, lower is better)" }));
  stat(s, 6.5, 1.4, 3.0, `${Math.round(840100 / best)}×`, "tighter than the first Python attempt (idle median vs native best case)", AMBER);
  stat(s, 6.5, 3.05, 3.0, fmtUs(best), "best-case bracket: close to the 5G ±1.5 µs budget, but only in the best 1%", TEAL);
  s.addNotes("v1: Python launches one kernel per sample and waits, so launch and wake-up overhead swamp everything. v2: a persistent GPU kernel spins on shared memory, compiled C host loop timed with the TSC.");
}

// 6. Finding 1: official API
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  badge(s, 1, 0.5, 0.45); s.addText("The official clock-pairing API is fake", { x: 1.1, y: 0.3, w: 8.4, h: 0.75, fontFace: HF, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
  s.addText("clGetDeviceAndHostTimer() should return one GPU reading and one CPU reading taken together. On this driver:", { x: 0.5, y: 1.3, w: 9, h: 0.6, fontFace: BF, fontSize: 15, color: TEXT, margin: 0, isTextBox: true });
  s.addText("device = 12136683975300\nhost   = 12136683975300", { x: 0.5, y: 2.0, w: 5.2, h: 0.9, fontFace: "Courier New", fontSize: 18, color: WHITE, fill: { color: INK }, margin: 12, isTextBox: true, valign: "middle" });
  s.addText("Identical. Both are the CPU's performance counter. The GPU clock is never consulted.", { x: 6.0, y: 2.0, w: 3.5, h: 0.9, fontFace: BF, fontSize: 14, bold: true, color: ROSE, margin: 0, isTextBox: true, valign: "middle" });
  card(s, 0.5, 3.25, 9, 1.6);
  s.addText([
    { text: "Also: ", options: { bold: true, color: INK } },
    { text: "kernel start/end timestamps from the driver are already converted into CPU time, with no way to see the conversion. " },
    { text: "The GPU's real clock (~99.8 MHz) is reachable only through an undocumented instruction (s_sendmsg_rtn GET_REALTIME).", options: { bold: true } },
  ], { x: 0.8, y: 3.4, w: 8.4, h: 1.3, fontFace: BF, fontSize: 15, color: TEXT, margin: 0, isTextBox: true, valign: "middle" });
  note(s, "AMD Adrenalin driver 32.0.13070.1002, OpenCL platform AMD-APP 3640.0");
}

// 7. Finding 2: rate
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  badge(s, 2, 0.5, 0.45); s.addText("On this chip, GPU and CPU clocks are locked", { x: 1.1, y: 0.3, w: 8.4, h: 0.75, fontFace: HF, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
  const lab = D.windows.map((w) => `${w.t.toFixed(0)}s`);
  s.addChart(pres.charts.LINE, [
    { name: "Default run", labels: lab, values: D.windows.map((w) => +(w.ppm * 1000).toFixed(1)) },
    { name: "Tuned run", labels: lab, values: T.windows.slice(0, lab.length).map((w) => +(w.ppm * 1000).toFixed(1)) },
  ], axis({ x: 0.5, y: 1.2, w: 5.8, h: 3.75, chartColors: [TEAL, AMBER], lineSize: 2, lineDataSymbol: "circle", lineDataSymbolSize: 5,
    valAxisMinVal: -100, valAxisMaxVal: 100, showLegend: true, legendPos: "b", showTitle: true, title: "GPU/CPU rate per 10 s window, parts per billion", catAxisLabelFrequency: 3 }));
  const ratio = D.tsc_mhz / D.gpu_mhz;
  stat(s, 6.7, 1.3, 2.8, "20 : 1", `CPU TSC (${D.tsc_mhz.toFixed(1)} MHz) to GPU clock (${D.gpu_mhz.toFixed(2)} MHz), exact to ${Math.abs((ratio / 20 - 1) * 1e9).toFixed(1)} ppb: one shared source`, TEAL);
  stat(s, 6.7, 3.0, 2.8, `±${(1000 * Math.max(D.wander_ppm_std, T.wander_ppm_std)).toFixed(0)} ppb`, "wander between them: the measurement noise floor, through load, sleep and GPU burn", AMBER);
  s.addNotes("The GPU clock runs at 99.81 MHz, not the 100 MHz you'd assume; assuming 100 MHz would be ~1.9 ms wrong per second. But its ratio to the CPU TSC is exactly 20 (within 0.0006 ppm). On an APU both derive from one reference, so relative drift is zero and only latency and offset matter. A discrete NVIDIA card has its own crystal, so there drift is real. That is where the OCXO reference matters most.");
}

// 8. Finding 3: bracket widths per phase
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  badge(s, 3, 0.5, 0.45); s.addText("Typical round trip is 10–150× the best case", { x: 1.1, y: 0.3, w: 8.4, h: 0.75, fontFace: HF, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
  const labs = PH.map((p) => PHL[p]);
  const mk = (key) => [
    { name: "Default (core 0, normal priority)", labels: labs, values: PH.map((p) => +us(D.phases[p][key]).toFixed(1)) },
    { name: "Tuned (core 5, time-critical)", labels: labs, values: PH.map((p) => +us(T.phases[p][key]).toFixed(1)) },
  ];
  s.addChart(pres.charts.BAR, mk("p50"), axis({ x: 0.4, y: 1.2, w: 4.6, h: 3.7, barDir: "col", barGrouping: "clustered", chartColors: [TEAL, AMBER],
    valAxisLogScaleBase: 10, showLegend: true, legendPos: "b", showTitle: true, title: "Median bracket, µs (log)", catAxisLabelFontSize: 9 }));
  s.addChart(pres.charts.BAR, mk("p99"), axis({ x: 5.1, y: 1.2, w: 4.6, h: 3.7, barDir: "col", barGrouping: "clustered", chartColors: [TEAL, AMBER],
    valAxisLogScaleBase: 10, showLegend: true, legendPos: "b", showTitle: true, title: "99th percentile bracket, µs (log)", catAxisLabelFontSize: 9 }));
  note(s, `Tuning (other core, top priority) cuts the median from ${fmtUs(D.phases.cpu_load.p50)} to ${fmtUs(T.phases.cpu_load.p50)} under CPU load, but the millisecond tails stay`);
  s.addNotes(PH.map((p) => `${PHL[p]}: default p50 ${fmtUs(D.phases[p].p50)}, p99 ${fmtUs(D.phases[p].p99)}; tuned p50 ${fmtUs(T.phases[p].p50)}, p99 ${fmtUs(T.phases[p].p99)}`).join("\n"));
}

// 9. Finding 4: where the time goes
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  badge(s, 4, 0.5, 0.45); s.addText("Where slow round trips lose their time", { x: 1.1, y: 0.3, w: 8.4, h: 0.75, fontFace: HF, fontSize: 28, bold: true, color: INK, margin: 0, isTextBox: true, valign: "middle" });
  const labs = PH.map((p) => PHL[p]);
  const tot = (C, p) => C.phases[p].stall_before_ms + C.phases[p].stall_after_ms || 1;
  s.addChart(pres.charts.BAR, [
    { name: "Before GPU read (GPU slow to notice)", labels: labs, values: PH.map((p) => +(100 * D.phases[p].stall_before_ms / tot(D, p)).toFixed(0)) },
    { name: "After GPU read (reply slow to reach CPU)", labels: labs, values: PH.map((p) => +(100 * D.phases[p].stall_after_ms / tot(D, p)).toFixed(0)) },
  ], axis({ x: 0.4, y: 1.2, w: 5.9, h: 3.75, barDir: "bar", barGrouping: "stacked", valAxisMaxVal: 100, chartColors: [AMBER, TEAL], showValue: true,
    dataLabelPosition: "ctr", dataLabelFontSize: 10, dataLabelColor: INK, showLegend: true, legendPos: "b", showTitle: true, title: "Share of excess time in brackets > 20 µs (default run), %" }));
  stat(s, 6.7, 1.3, 2.8, `${(100 * D.phases.tight.frac_over_20us).toFixed(0)}%`, "of round trips took > 20 µs even in the plain tight loop. The return path dominates, except when the GPU was asleep", ROSE);
  stat(s, 6.7, 3.0, 2.8, fmtUs(Math.max(...PH.map((p) => D.phases[p].max))), "longest single stall. Suspect: the iGPU also drives the display and gets time-sliced (not yet proven)", AMBER);
  s.addNotes("Using the fitted clock mapping, each slow bracket can be split into time before the GPU read and time after it. This separates 'GPU was not running our code' from 'CPU was not running our thread'.");
}

// 10. Finding 5: no ground truth
{
  const s = pres.addSlide(); s.background = { color: INK };
  badge(s, 5, 0.5, 0.45); s.addText("Nothing in the machine can referee", { x: 1.1, y: 0.3, w: 8.4, h: 0.75, fontFace: HF, fontSize: 28, bold: true, color: WHITE, margin: 0, isTextBox: true, valign: "middle" });
  s.addText(`CPU and GPU agree perfectly, yet both run ${Math.abs(Math.round(D.gpu_ppm_vs_100))} ppm slow against Windows' own timer (QPC). Which is right? Every clock here comes from a cheap crystal on one board, so nothing can referee.`, { x: 0.5, y: 1.3, w: 9, h: 0.9, fontFace: BF, fontSize: 16, color: ICE, margin: 0, isTextBox: true });
  const box = (x, head, body, c) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 2.45, w: 2.85, h: 2.3, rectRadius: 0.08, fill: { color: INK2 }, line: { color: INK2 } });
    s.addText(head, { x: x + 0.2, y: 2.6, w: 2.45, h: 0.5, fontFace: HF, fontSize: 17, bold: true, color: c, margin: 0, isTextBox: true });
    s.addText(body, { x: x + 0.2, y: 3.1, w: 2.45, h: 1.55, fontFace: BF, fontSize: 13, color: WHITE, margin: 0, isTextBox: true, valign: "top" });
  };
  box(0.5, "We can measure", "the ratio between two clocks, and how tightly a read can be bracketed", TEAL);
  box(3.575, "We cannot measure", "which clock is wrong, or drift that both clocks share (temperature, ageing)", ROSE);
  box(6.65, "The fix", "an external reference ≥10× more stable than either: an oven-controlled crystal (OCXO), disciplined by GPS", AMBER);
}

// 11. Proposed thesis setup
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Proposed MSc project: a hardware time referee");
  const blocks = [
    ["GPS antenna", "true time, 1 PPS", LIGHT],
    ["OCXO board", "my PCB (KiCad): 10 MHz + PPS", "FCE7C8"],
    ["FPGA PCIe card", "counter on OCXO clock, DMA timestamps", LIGHT],
    ["Host RAM", "\"true time = T\" updated constantly", LIGHT],
    ["CPU + GPU", "both read T and their own clock", "D5EFEA"],
  ];
  blocks.forEach(([h, b, f], i) => {
    const x = 0.4 + i * 1.9;
    card(s, x, 1.35, 1.65, 1.55, f);
    s.addText(h, { x: x + 0.12, y: 1.45, w: 1.41, h: 0.45, fontFace: HF, fontSize: 14, bold: true, color: INK, margin: 0, isTextBox: true, align: "center", valign: "middle" });
    s.addText(b, { x: x + 0.12, y: 1.9, w: 1.41, h: 0.9, fontFace: BF, fontSize: 11, color: TEXT, margin: 0, isTextBox: true, align: "center", valign: "top" });
    if (i < blocks.length - 1) s.addShape(pres.shapes.LINE, { x: x + 1.67, y: 2.12, w: 0.21, h: 0, line: { color: INK, width: 1.5, endArrowType: "triangle" } });
  });
  s.addText("Research question", { x: 0.5, y: 3.25, w: 9, h: 0.35, fontFace: BF, fontSize: 13, bold: true, color: AMBER, margin: 0, isTextBox: true });
  s.addText("How accurately can code on a GPU know true time, and does a PCIe hardware reference beat software CPU–GPU correlation under real load?", { x: 0.5, y: 3.6, w: 9, h: 0.85, fontFace: HF, fontSize: 18, color: INK, margin: 0, isTextBox: true });
  note(s, "Target platform: desktop with NVIDIA RTX (%globaltimer) · GeForce lacks official GPUDirect RDMA, so the card talks to the GPU via host memory");
}

// 12. Plan
{
  const s = pres.addSlide(); s.background = { color: WHITE };
  title(s, "Plan");
  const steps = [
    ["Done", "Software-only feasibility on laptop iGPU: method works, API gap found", TEAL],
    ["Next", "Port ping-pong to CUDA on a rented RTX; compare with CUPTI's own correlation", AMBER],
    ["Build", "OCXO + GPS PPS board in KiCad, fab; FPGA counter + DMA (LitePCIe)", AMBER],
    ["Test", "Idle / load / heat / reference loss; software vs hardware reference vs drift model", AMBER],
  ];
  steps.forEach(([k, v, c], i) => {
    const y = 1.3 + i * 0.9;
    s.addShape(pres.shapes.OVAL, { x: 0.5, y, w: 0.6, h: 0.6, fill: { color: c }, line: { color: c } });
    s.addText(String(i + 1), { x: 0.5, y, w: 0.6, h: 0.6, fontFace: HF, fontSize: 18, bold: true, color: i === 0 ? WHITE : INK, align: "center", valign: "middle", margin: 0, isTextBox: true });
    s.addText(k, { x: 1.3, y, w: 1.2, h: 0.6, fontFace: HF, fontSize: 18, bold: true, color: INK, valign: "middle", margin: 0, isTextBox: true });
    s.addText(v, { x: 2.5, y, w: 7, h: 0.6, fontFace: BF, fontSize: 15, color: TEXT, valign: "middle", margin: 0, isTextBox: true });
  });
}

// 13. Limits + takeaway
{
  const s = pres.addSlide(); s.background = { color: INK };
  title(s, "What this does and doesn't show", true);
  s.addText([
    { text: "One laptop, AMD iGPU sharing a clock source with the CPU, on battery. An NVIDIA card over PCIe has its own crystal and will drift.", options: { bullet: true, breakLine: true } },
    { text: "Bracket midpoints assume symmetric delays; any asymmetry is a hidden constant offset.", options: { bullet: true, breakLine: true } },
    { text: "Rate stability is relative: shared drift of both clocks is invisible without an external reference.", options: { bullet: true, breakLine: true } },
    { text: "About 10 minutes of data; thermal and ageing effects need hours.", options: { bullet: true } },
  ], { x: 0.5, y: 1.25, w: 9, h: 2.2, fontFace: BF, fontSize: 15, color: ICE, margin: 0, isTextBox: true, paraSpaceAfter: 8, valign: "top" });
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.5, y: 3.7, w: 9, h: 1.2, rectRadius: 0.08, fill: { color: INK2 }, line: { color: INK2 } });
  s.addText([
    { text: "Takeaway: ", options: { bold: true, color: AMBER } },
    { text: `software alone gets a GPU clock read to ~${fmtUs(best)} at best and tens of µs typically, and can't tell which clock is right. A hardware time reference is the missing piece, so it's the thesis.`, options: { color: WHITE } },
  ], { x: 0.8, y: 3.8, w: 8.4, h: 1.0, fontFace: BF, fontSize: 16, margin: 0, isTextBox: true, valign: "middle" });
}

pres.writeFile({ fileName: "GPU_Clock_Findings.pptx" }).then((f) => console.log("wrote", f));
