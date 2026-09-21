// Build the adaLN-single -> DPLM conditioning schematic as NATIVE pptx shapes
// (rounded rects, real connector arrows, auto-fit text). LAYOUT_WIDE = 13.3 x 7.5.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
const slide = pres.addSlide();

const COND = "2E86DE", COND_FILL = "EAF2FB";
const MOD = "E67E22", MOD_TXT = "A85B16", MOD_FILL = "FDF0E2", FFN_FILL = "F6EDE2";
const BLK = "34495E", BOX = "E3EAEF", CONT = "EEF2F4";

const arrow = (x, y, w, h, color = BLK, width = 1.75) =>
  slide.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width, endArrowType: "triangle" } });
const ml = (lines) => lines.map((t, i) => ({ text: t, options: { breakLine: i < lines.length - 1 } }));

// Title
slide.addText("How adaLN-single conditioning steers DPLM",
  { x: 0.3, y: 0.16, w: 12.73, h: 0.6, fontSize: 24, bold: true, align: "center", color: "222222" });

// ---------- LEFT: conditioning path ----------
const LX = 0.5, LW = 3.7, LCX = LX + LW / 2;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 1.05, w: LW, h: 0.78, rectRadius: 0.06, fill: { color: COND_FILL }, line: { color: COND, width: 1.5 } });
slide.addText(ml(["First-cyclization product class", "(1 of 22 classes)"]), { x: LX, y: 1.05, w: LW, h: 0.78, fontSize: 12, bold: true, color: COND, align: "center", valign: "middle", margin: 2 });
arrow(LCX, 1.85, 0, 0.26, COND, 1.5);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 2.13, w: LW, h: 0.58, rectRadius: 0.06, fill: { color: COND_FILL }, line: { color: COND, width: 1.5 } });
slide.addText("Class embedding  (640-dim)", { x: LX, y: 2.13, w: LW, h: 0.58, fontSize: 12.5, color: COND, align: "center", valign: "middle", margin: 2 });
arrow(LCX, 2.73, 0, 0.26, MOD, 1.5);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 3.01, w: LW, h: 1.0, rectRadius: 0.06, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["adaLN-single modulation MLP", "Linear(640 → 64) → SiLU → Linear(64 → 3840)", "+ per-layer offset table (zero-init)"]), { x: LX, y: 3.01, w: LW, h: 1.0, fontSize: 10.5, bold: true, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });
arrow(LCX, 4.03, 0, 0.24, MOD, 1.5);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: LX, y: 4.29, w: LW, h: 0.72, rectRadius: 0.06, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["for every layer ℓ  →", "shift, scale, gate  ×2  (attention + FFN)"]), { x: LX, y: 4.29, w: LW, h: 0.72, fontSize: 10.5, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });

// bridge arrow into the block
arrow(4.3, 4.62, 1.45, 0, MOD, 2.5);
slide.addText("6 modulation\nparams / layer", { x: 4.25, y: 3.92, w: 1.55, h: 0.55, fontSize: 9, italic: true, color: MOD_TXT, align: "center", valign: "middle" });

// ---------- RIGHT: one transformer block ----------
const BX = 5.8, BY = 0.95, BW = 7.2, BH = 5.72;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: BX, y: BY, w: BW, h: BH, rectRadius: 0.05, fill: { color: CONT }, line: { color: BLK, width: 1.75, dashType: "dash" } });
slide.addText("ESM-2 / DPLM transformer block   × 30   (backbone FROZEN)", { x: BX, y: BY + 0.06, w: BW, h: 0.4, fontSize: 13, bold: true, color: BLK, align: "center", valign: "middle" });

const CX = 8.7, PW = 2.9, PX = CX - PW / 2;      // pipeline box geometry
const lab = (y, t) => slide.addText(t, { x: 10.35, y, w: 2.5, h: 0.42, fontSize: 9.5, italic: true, color: MOD_TXT, align: "left", valign: "middle", margin: 0 });

slide.addText("hidden states  x", { x: PX, y: 1.5, w: PW, h: 0.3, fontSize: 11, align: "center", color: "222222" });
arrow(CX, 1.82, 0, 0.18);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: PX, y: 2.0, w: PW, h: 0.44, rectRadius: 0.04, fill: { color: "FFFFFF" }, line: { color: MOD, width: 2.25 } });
slide.addText("LayerNorm", { x: PX, y: 2.0, w: PW, h: 0.44, fontSize: 11, align: "center", valign: "middle" });
lab(2.0, "× (1+scale) + shift");
arrow(CX, 2.44, 0, 0.16);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: PX, y: 2.6, w: PW, h: 0.46, rectRadius: 0.04, fill: { color: BOX }, line: { color: BLK, width: 1 } });
slide.addText("Multi-Head Self-Attention", { x: PX, y: 2.6, w: PW, h: 0.46, fontSize: 10.5, align: "center", valign: "middle" });
arrow(CX, 3.06, 0, 0.16);

slide.addShape(pres.shapes.OVAL, { x: CX - 0.22, y: 3.22, w: 0.44, h: 0.36, fill: { color: "FFFFFF" }, line: { color: MOD, width: 2 } });
slide.addText("⊙", { x: CX - 0.22, y: 3.21, w: 0.44, h: 0.36, fontSize: 14, color: MOD, align: "center", valign: "middle" });
lab(3.2, "× gate");
arrow(CX, 3.58, 0, 0.16);

slide.addShape(pres.shapes.OVAL, { x: CX - 0.22, y: 3.74, w: 0.44, h: 0.36, fill: { color: "FFFFFF" }, line: { color: BLK, width: 1.5 } });
slide.addText("+", { x: CX - 0.22, y: 3.72, w: 0.44, h: 0.36, fontSize: 16, align: "center", valign: "middle" });
slide.addText("residual add", { x: 10.35, y: 3.74, w: 2.5, h: 0.36, fontSize: 9.5, italic: true, color: BLK, align: "left", valign: "middle", margin: 0 });
arrow(CX, 4.1, 0, 0.18);

// FFN sub-block collapsed (same pattern)
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.45, y: 4.28, w: 4.5, h: 0.86, rectRadius: 0.05, fill: { color: FFN_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["FFN sub-block — same pattern:", "LayerNorm → ×(1+scale)+shift → FFN → × gate → ⊕ residual"]), { x: 6.45, y: 4.28, w: 4.5, h: 0.86, fontSize: 9.5, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });
arrow(CX, 5.14, 0, 0.18);
slide.addText("→ next layer", { x: PX, y: 5.34, w: PW, h: 0.3, fontSize: 10.5, italic: true, color: BLK, align: "center" });

// ---------- footer ----------
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: 6.78, w: 12.53, h: 0.5, rectRadius: 0.03, fill: { color: "F7F9FA" }, line: { color: "BBBBBB", width: 1 } });
slide.addText("zero-init gates ⇒ no-op at init (stable finetune)    •    only adaLN MLP + class embedding train ≈ 0.42M params    •    no timestep input (DPLM is time-agnostic)    •    CFG: class dropped 15% → learned null ⇒ guidance dial w",
  { x: 0.5, y: 6.78, w: 12.33, h: 0.5, fontSize: 8.5, color: "333333", align: "center", valign: "middle" });

pres.writeFile({ fileName: "/Users/soldatmat/Documents/terpene_synthases/dplm/_adaln_arch_slide/arch_slide.pptx" })
  .then(f => console.log("wrote", f));
