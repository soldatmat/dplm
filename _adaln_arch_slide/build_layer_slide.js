// Per-layer adaLN application inside one ESM layer (ModifiedEsmLayer.forward).
// Native shapes. LAYOUT_WIDE 13.333 x 7.5.
const pptxgen = require("pptxgenjs");
const pres = new pptxgen();
pres.defineLayout({ name: "W", width: 13.333, height: 7.5 });
pres.layout = "W";
const slide = pres.addSlide();

const MOD = "E67E22", MOD_TXT = "A85B16", MOD_FILL = "FDF0E2";     // adaLN-injected ops
const ESM = "5D6D7E", ESM_FILL = "E8ECEF";                          // frozen ESM ops
const BLK = "2C3E50", GRY = "7F8C8D";

const arrowR = (x, y, w, color = BLK, width = 1.6) =>
  slide.addShape(pres.shapes.LINE, { x, y, w, h: 0, line: { color, width, endArrowType: "triangle" } });
const seg = (x, y, w, h, color = GRY, width = 1.5, arrow = false) =>
  slide.addShape(pres.shapes.LINE, { x, y, w, h, line: { color, width, endArrowType: arrow ? "triangle" : "none" } });
const ml = (lines) => lines.map((t, i) => ({ text: t, options: { breakLine: i < lines.length - 1 } }));

slide.addText("How adaLN is applied inside one ESM layer  (ModifiedEsmLayer.forward)",
  { x: 0.3, y: 0.16, w: 12.73, h: 0.5, fontSize: 21, bold: true, align: "center", color: "222222" });

// --- top: the per-layer modulation slice + unbind order ---
slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 0.82, w: 12.13, h: 0.52, rectRadius: 0.04, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText([{ text: "per-layer modulation slice  [B, 6, H]  →  unbind →  ", options: { bold: true, color: MOD_TXT } },
               { text: "( shift_attn, scale_attn, gate_attn,  shift_ffn, scale_ffn, gate_ffn )", options: { color: MOD_TXT } }],
  { x: 0.6, y: 0.82, w: 12.13, h: 0.52, fontSize: 11.5, align: "center", valign: "middle" });

// --- one sub-block lane ---
function lane(yTop, laneTitle, sub, sublayerLines) {
  const yc = yTop + 0.3, top = yTop, bracket = yTop - 0.34;
  slide.addText(laneTitle, { x: 0.35, y: yTop - 0.6, w: 4.0, h: 0.3, fontSize: 12, bold: true, color: BLK, align: "left" });

  // node h_in
  const hx = 0.85;
  slide.addShape(pres.shapes.OVAL, { x: hx - 0.21, y: yc - 0.21, w: 0.42, h: 0.42, fill: { color: "FFFFFF" }, line: { color: BLK, width: 1.5 } });
  slide.addText("h", { x: hx - 0.21, y: yc - 0.23, w: 0.42, h: 0.42, fontSize: 13, bold: true, align: "center", valign: "middle" });

  // LayerNorm (frozen ESM)
  arrowR(hx + 0.21, yc, 0.28);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 1.35, y: top, w: 1.45, h: 0.6, rectRadius: 0.04, fill: { color: ESM_FILL }, line: { color: ESM, width: 1.25 } });
  slide.addText("LayerNorm", { x: 1.35, y: top, w: 1.45, h: 0.6, fontSize: 10, align: "center", valign: "middle", color: BLK });

  // modulate (adaLN)
  arrowR(2.8, yc, 0.26, MOD);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 3.06, y: top, w: 2.25, h: 0.6, rectRadius: 0.04, fill: { color: MOD_FILL }, line: { color: MOD, width: 2 } });
  slide.addText(ml(["modulate", `· (1 + scale_${sub}) + shift_${sub}`]), { x: 3.06, y: top, w: 2.25, h: 0.6, fontSize: 9.5, bold: true, color: MOD_TXT, align: "center", valign: "middle", margin: 1 });

  // sub-layer (frozen ESM, dense+dropout WITHOUT residual)
  arrowR(5.31, yc, 0.26);
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 5.57, y: top, w: 3.25, h: 0.6, rectRadius: 0.04, fill: { color: ESM_FILL }, line: { color: ESM, width: 1.25 } });
  slide.addText(ml(sublayerLines), { x: 5.57, y: top, w: 3.25, h: 0.6, fontSize: 9, color: BLK, align: "center", valign: "middle", margin: 1 });

  // gate (adaLN)
  arrowR(8.82, yc, 0.22, MOD);
  slide.addShape(pres.shapes.OVAL, { x: 9.04, y: yc - 0.21, w: 0.42, h: 0.42, fill: { color: "FFFFFF" }, line: { color: MOD, width: 2 } });
  slide.addText("⊙", { x: 9.04, y: yc - 0.22, w: 0.42, h: 0.42, fontSize: 13, color: MOD, align: "center", valign: "middle" });
  slide.addText(`× gate_${sub}`, { x: 8.75, y: yc + 0.24, w: 1.0, h: 0.26, fontSize: 8.5, italic: true, color: MOD_TXT, align: "center" });

  // residual add
  arrowR(9.46, yc, 0.22);
  slide.addShape(pres.shapes.OVAL, { x: 9.68, y: yc - 0.22, w: 0.44, h: 0.44, fill: { color: "FFFFFF" }, line: { color: BLK, width: 1.5 } });
  slide.addText("+", { x: 9.68, y: yc - 0.24, w: 0.44, h: 0.44, fontSize: 16, align: "center", valign: "middle" });

  // out
  arrowR(10.12, yc, 0.3);
  slide.addShape(pres.shapes.OVAL, { x: 10.42, y: yc - 0.21, w: 0.42, h: 0.42, fill: { color: "FFFFFF" }, line: { color: BLK, width: 1.5 } });
  slide.addText("h", { x: 10.42, y: yc - 0.23, w: 0.42, h: 0.42, fontSize: 13, bold: true, align: "center", valign: "middle" });

  // residual bypass bracket: h_in -> up -> across -> down into (+)
  const plusX = 9.9;
  seg(hx, top, 0, -(top - bracket), GRY, 1.5);              // up-tick from h_in
  seg(hx, bracket, plusX - hx, 0, GRY, 1.5);                // top horizontal
  seg(plusX, bracket, 0, (yc - 0.22) - bracket, GRY, 1.5, true); // down into (+)
  slide.addText("residual  (h, unchanged)", { x: 5.0, y: bracket - 0.25, w: 3.2, h: 0.24, fontSize: 8.5, italic: true, color: GRY, align: "center" });
}

lane(2.05, "①  Attention sub-block", "attn", ["Self-Attention → dense + dropout", "( WITHOUT the built-in residual )"]);
lane(3.95, "②  FFN sub-block", "ffn", ["Intermediate → output dense + dropout", "( WITHOUT the built-in residual )"]);

// --- legend ---
slide.addShape(pres.shapes.RECTANGLE, { x: 11.05, y: 2.0, w: 0.22, h: 0.22, fill: { color: ESM_FILL }, line: { color: ESM, width: 1 } });
slide.addText("frozen ESM op", { x: 11.32, y: 1.95, w: 1.9, h: 0.3, fontSize: 8.5, color: BLK, valign: "middle" });
slide.addShape(pres.shapes.RECTANGLE, { x: 11.05, y: 2.34, w: 0.22, h: 0.22, fill: { color: MOD_FILL }, line: { color: MOD, width: 1.5 } });
slide.addText("adaLN-injected", { x: 11.32, y: 2.29, w: 1.9, h: 0.3, fontSize: 8.5, color: MOD_TXT, valign: "middle" });

// --- bottom notes ---
const note = (x, w, color, fill, title, body) => {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 5.05, w, h: 2.05, rectRadius: 0.04, fill: { color: fill }, line: { color, width: 1.25 } });
  slide.addText(title, { x: x + 0.18, y: 5.15, w: w - 0.36, h: 0.3, fontSize: 10.5, bold: true, color, align: "left" });
  slide.addText(body, { x: x + 0.18, y: 5.45, w: w - 0.36, h: 1.55, fontSize: 9.5, color: "333333", align: "left", valign: "top" });
};
note(0.4, 4.0, "5D6D7E", "EEF2F4", "Pre-LN architecture",
  ml(["ESM-2 normalizes BEFORE each sub-block and adds", "the residual without LN — so adaLN slots in exactly", "at the DiT/PixArt points (modulate after LN; gate on", "the residual branch)."]));
note(4.65, 4.0, "2C3E50", "F4F6F8", "modulate(x, shift, scale)",
  ml(["= x · (1 + scale) + shift", "", "scale, shift broadcast over the sequence length.", "scale = 0 ⇒ ×1, shift = 0 ⇒ identity (the zero-init no-op)."]));
note(8.9, 4.03, "E67E22", "FDF0E2", "Why dense+dropout WITHOUT the residual",
  ml(["ESM's EsmSelfOutput / EsmOutput normally fold the", "residual add into dense(). We call them without it, so", "the gate scales ONLY the sub-block's contribution —", "then we add the residual manually."]));

pres.writeFile({ fileName: "/Users/soldatmat/Documents/terpene_synthases/dplm/_adaln_arch_slide/layer_slide.pptx" })
  .then(f => console.log("wrote", f));
