// adaLN CONDITIONING-MECHANISM zoom-in slide (native shapes). LAYOUT_WIDE 13.333x7.5.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
const slide = pres.addSlide();

const COND = "2E86DE", COND_FILL = "EAF2FB";
const MOD = "E67E22", MOD_TXT = "A85B16", MOD_FILL = "FDF0E2";
const GRN = "27AE60", GRN_FILL = "E8F5E9";
const PUR = "8E44AD", PUR_FILL = "F3E9F7";
const EQ_FILL = "F4F6F8", BLK = "34495E";

const arrowH = (x, y, w, color = MOD) =>
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width: 2, endArrowType: "triangle" } });
const ml = (lines) => lines.map((t, i) => ({ text: t, options: { breakLine: i < lines.length - 1 } }));

slide.addText("The adaLN conditioning mechanism — class embedding → per-layer modulation",
  { x: 0.3, y: 0.16, w: 12.73, h: 0.5, fontSize: 21, bold: true, align: "center", color: "222222" });

// ---------- ROW 1: parameter generation (horizontal) ----------
const Y1 = 0.95, H1 = 0.92;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.35, y: Y1, w: 2.75, h: H1, rectRadius: 0.05, fill: { color: COND_FILL }, line: { color: COND, width: 1.5 } });
slide.addText(ml(["class embedding  c", "(640-d, 1 of 22 classes)", "— or learned CFG null —"]), { x: 0.35, y: Y1, w: 2.75, h: H1, fontSize: 9.5, bold: true, color: COND, align: "center", valign: "middle", margin: 2 });
arrowH(3.12, Y1 + H1 / 2, 0.26, COND);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.4, y: Y1, w: 3.0, h: H1, rectRadius: 0.05, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["shared modulation MLP", "Linear(640 → 64) → SiLU", "→ Linear(64 → 3840)"]), { x: 3.4, y: Y1, w: 3.0, h: H1, fontSize: 9.5, bold: true, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });
arrowH(6.42, Y1 + H1 / 2, 0.26, MOD);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.7, y: Y1, w: 2.8, h: H1, rectRadius: 0.05, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["+ per-layer offset  Δℓ", "table 30 × 3840", "(zero-init)"]), { x: 6.7, y: Y1, w: 2.8, h: H1, fontSize: 9.5, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });
arrowH(9.52, Y1 + H1 / 2, 0.26, MOD);

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 9.8, y: Y1, w: 3.15, h: H1, rectRadius: 0.05, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText(ml(["split → 6 vectors (each 640-d):", "γ, β, g   for attention", "γ, β, g   for FFN"]), { x: 9.8, y: Y1, w: 3.15, h: H1, fontSize: 9.5, bold: true, color: MOD_TXT, align: "center", valign: "middle", margin: 2 });

slide.addText("3840 = 6 × H   (H = 640):   per layer, two sub-blocks × { scale γ,  shift β,  gate g }",
  { x: 0.35, y: 1.92, w: 12.6, h: 0.3, fontSize: 9.5, italic: true, color: "666666", align: "center" });

// ---------- ROW 2: the modulation equations (focus) ----------
slide.addText([{ text: "Applied at every layer, to both sub-blocks  ", options: { bold: true } },
               { text: "(attention: γ₁,β₁,g₁ — FFN: γ₂,β₂,g₂)", options: { bold: true, color: MOD_TXT } }],
  { x: 0.35, y: 2.34, w: 12.6, h: 0.32, fontSize: 12, align: "center", color: "222222" });

const eqBox = (y, num, label, eq, sub) => {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.0, y, w: 11.33, h: 1.04, rectRadius: 0.04, fill: { color: EQ_FILL }, line: { color: BLK, width: 1.25 } });
  slide.addText(`${num}  ${label}`, { x: 1.25, y: y + 0.07, w: 10.8, h: 0.3, fontSize: 11, bold: true, color: BLK, align: "left", margin: 0 });
  slide.addText(eq, { x: 1.0, y: y + 0.34, w: 11.33, h: 0.45, fontSize: 22, bold: true, fontFace: "Arial", color: "111111", align: "center", valign: "middle" });
  slide.addText(sub, { x: 1.0, y: y + 0.78, w: 11.33, h: 0.24, fontSize: 9, italic: true, color: "666666", align: "center" });
};
eqBox(2.74, "①", "Modulated LayerNorm  (FiLM-style affine)", "y  =  LayerNorm(h) ⊙ (1 + γ)  +  β", "scale γ and shift β modulate the normalized activations of the sub-block");
eqBox(3.92, "②", "Gated residual", "h  ←  h  +  g ⊙ Sublayer(y)", "gate g scales how much the sub-block writes into the residual stream");

// ---------- ROW 3: the two defining properties ----------
const Y3 = 5.2, H3 = 1.2;
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.4, y: Y3, w: 6.05, h: H3, rectRadius: 0.05, fill: { color: GRN_FILL }, line: { color: GRN, width: 1.5 } });
slide.addText("Zero-init  (adaLN-zero)", { x: 0.6, y: Y3 + 0.08, w: 5.7, h: 0.3, fontSize: 11.5, bold: true, color: GRN, margin: 0 });
slide.addText(ml(["final Linear + offset table initialized to 0", "⇒ γ = β = 0,  g = 0   ⇒   y = LayerNorm(h),  h ← h", "⇒ EXACT identity at init → stable finetune of frozen backbone"]),
  { x: 0.6, y: Y3 + 0.38, w: 5.7, h: 0.78, fontSize: 9.5, color: "245a32", align: "left", valign: "middle", margin: 0 });

slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.6, y: Y3, w: 6.33, h: H3, rectRadius: 0.05, fill: { color: PUR_FILL }, line: { color: PUR, width: 1.5 } });
slide.addText("Classifier-free guidance  (CFG)", { x: 6.8, y: Y3 + 0.08, w: 5.95, h: 0.3, fontSize: 11.5, bold: true, color: PUR, margin: 0 });
slide.addText(ml(["training: replace c by a learned null vector with p = 0.15", "sampling: logits = (1 + w)·cond − w·null", "⇒ guidance dial w controls steering strength at generation"]),
  { x: 6.8, y: Y3 + 0.38, w: 5.95, h: 0.78, fontSize: 9.5, color: "5b2c6f", align: "left", valign: "middle", margin: 0 });

// ---------- footer ----------
slide.addText("Trainable: shared MLP + offset table + 22 class embeddings ≈ 0.42M params    •    ESM-2 backbone frozen    •    no timestep (DPLM is time-agnostic)",
  { x: 0.4, y: 6.62, w: 12.53, h: 0.35, fontSize: 8.5, color: "555555", align: "center" });

pres.writeFile({ fileName: "/Users/soldatmat/Documents/terpene_synthases/dplm/_adaln_arch_slide/cond_slide.pptx" })
  .then(f => console.log("wrote", f));
