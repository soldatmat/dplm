# A3 Conditioning Redesign — Literature Survey

Research-only survey for redesigning DPLM class-conditional generation of terpene synthases (conditioning on first-cyclization product class). DPLM = discrete/masked diffusion LM over amino-acid sequences, ESM-2 backbone. Current frozen per-class mean-embedding conditioning (cross-attn / prepend / LoRA) does NOT steer. Two candidate fixes surveyed: (1) classifier-free guidance / condition-dropout, (2) exemplar/neighbor-embedding conditioning.

Confidence flags used below: **[well-established]** = standard, many sources; **[protein-confirmed]** = demonstrated in a protein/bio-sequence model; **[thin]** = sparse or indirect evidence.

---

## 1. Classifier-Free Guidance (CFG) / Condition Dropout

### 1.1 The canonical method (Ho & Salimans 2022) [well-established]

Train ONE network to be both conditional and unconditional. During training, with probability `p_uncond` replace the conditioning `c` with a fixed **null token** `∅` (a single learned embedding, not zeros — see 1.4). At inference, extrapolate the two predictions away from the unconditional one:

- Continuous (epsilon-space): `ε̃ = ε(x,∅) + w·(ε(x,c) − ε(x,∅)) = (1+w)·ε(x,c) − w·ε(x,∅)`.
- The "guidance scale" knob is `w` (some papers call the same quantity `s` or `γ`, sometimes parameterized as `1+w`). `w=0` ⇒ pure conditional; larger `w` ⇒ stronger adherence to `c`, higher fidelity, **lower diversity**.

**Recommended values [well-established]:**
- Drop probability `p_uncond`: **10–20%** is the standard sweet spot. Original paper found ~10–20% works well and that performance is **not very sensitive** within this band; too-high dropout weakens the conditional model. (ProGen, a conditional protein LM, used a much higher tag-dropout of **0.4** — but that is autoregressive control-tag dropout, not diffusion CFG; see 1.5.)
- Guidance scale `w`: image diffusion typically **w ≈ 1–5** (continuous, classifier-free `s=1+w` reported as 1–4 in the paper's notation). For language/discrete settings the useful range is **smaller**, roughly **w ≈ 0.5–3** (see 1.2, 1.3); beyond ~3–5 quality degrades.

### 1.2 CFG for discrete / masked diffusion [well-established, formula confirmed]

The clean adaptation for masked/multinomial discrete diffusion (used by D3PM-style and masked-diffusion LMs, the DPLM family) operates in **logit / log-probability space**. The guided distribution is a tempered product of experts:

> `p_guided(x) ∝ p(x|c)^(1+w) · p(x)^(−w)`

Taking logs, this is exactly the same linear extrapolation, now on **per-token logits** (before softmax):

> `logit_guided = (1+w)·logit_cond − w·logit_uncond`  (equivalently `logit_uncond + w·(logit_cond − logit_uncond)`)

then softmax → sample/unmask. Two practical findings specific to discrete/masked diffusion:

- **Logit-space convex interpolation can beat extrapolation.** For categorical tokens, plain extrapolation can "overshoot" and put mass on degenerate tokens; some works report that interpolation-style or **normalized (softmax column-normalized) guidance** is more stable across `w`. ("Improving Classifier-Free Guidance in Masked Diffusion", 2025.)
- **Guidance schedule matters a lot in masked diffusion.** Strong guidance applied **early** (when the sequence is mostly `[MASK]`) HURTS — it causes premature, over-confident unmasking. **Keep `w` small early, ramp it up over the middle/late denoising steps.** Several analyses (incl. "What Exactly Does Guidance Do in Masked Discrete Diffusion Models", 2025) converge on this. This is the opposite of some image-diffusion schedules and is the single most important discrete-specific caveat.
- Empirically (ImageNet-class discrete diffusion, Kuleshov group): `w` baseline 0 (= no guidance), useful **w ≈ 2–3**, and `w ≳ 4–5` degrades. Their D-CFG trains by replacing the class with a dummy token = `num_classes+1` (a dedicated null index) with conditioning dropout.

### 1.3 CFG for autoregressive / MLM-style LMs [well-established]

"Stay on Topic with Classifier-Free Guidance" (Sanchez et al. 2023) applies CFG to autoregressive LLMs: `logp_guided = logp(x|c) + w·(logp(x|c) − logp(x|∅))` in **log-prob space**, with the unconditional branch being the prompt-dropped forward pass. The "drop the conditioning" branch is natural for LMs (just omit the prefix). Reported useful range is modest (`w` on the order of 1–3); large `w` causes degenerate / repetitive text. The takeaway transferring to DPLM: **logit-space CFG works for token models, but the useful `w` is smaller than for images and over-guidance produces degenerate sequences** (here: low-complexity / repetitive amino-acid runs).

### 1.4 Null-conditioning representation [well-established]

- Use a **single learned "null" embedding** (a trainable `∅` vector / dedicated null class index), NOT a zero vector and NOT mean-pooling. Zeros are a valid-looking point in embedding space and the model cannot cleanly distinguish "no condition" from "condition ≈ 0", which weakens guidance. A dedicated learned token is the standard and most robust choice (image diffusion, LLM-CFG, and discrete D-CFG all do this).
- For mean-embedding conditioning specifically (your current setup), the null should be an **extra learned vector in the same injection pathway** (extra cross-attn key/value, or an extra prepended token), trained via the dropout to represent "unconditional."

### 1.5 CFG / condition-dropout in protein & bio-sequence models [protein-confirmed]

- **DPLM itself** ("Diffusion LMs Are Versatile Protein Learners", 2024, App. D.5): explicitly develops **classifier-free guidance for adapter-tuned DPLM** as a "booster for cross-modal conditional generation," and notably claims it works **without intricate condition-dropout during training** (the adapter gives an unconditional path by zeroing/ablating the adapter). DPLM also has a separate *discrete classifier guidance* path with strength `η`. So CFG is native to your model family — but the public paper is light on exact `p`/`w` numbers (they live in the code/appendix).
- **ProGen / ProGen2** (conditional autoregressive protein LM): trains with **control-tag dropout = 0.4**, letting it generate conditionally or unconditionally. High because tags are categorical prefix tokens, not a CFG extrapolation; still evidence that aggressive condition-dropout is tolerated in protein LMs.
- **PRO-LDM** (Conditional Latent Diffusion for protein sequences, 2025) [protein-confirmed]: uses classifier-free guidance over function (GO) + organism class-label embeddings; reports that adjusting CFG lets them sample very different latent regions (controllability/outlier design) — i.e., CFG demonstrably steers a protein generator.
- **CFP-Gen** (Combinatorial Functional Protein Generation via diffusion LM, 2025): conditions an ESM-based discrete diffusion LM on GO/IPR/EC annotations via **feature-modulation (AGFM, FiLM-like) + ControlNet-style branch (RCFE)** with **composable, optionally-omitted conditions** — a design point relevant to A3 but it does *not* report a CFG `w`/`p`.
- General: classifier(-free) guidance is used in structure/sequence protein diffusion (Chroma, EvoDiff classifier-guidance, "Protein Design with Guided Discrete Diffusion" NeurIPS 2023). EvoDiff favors classifier guidance and full conditional fine-tuning over CFG; Chroma uses guidance heavily for structure.

**Protein-specific evidence quality:** CFG *exists and helps controllability* in protein generators **[protein-confirmed]**, but published, copy-pasteable `p_uncond` and `w` values for protein **sequence** diffusion are **[thin]** — you will likely need to sweep `w` yourself.

### Key citations — Topic 1
- Ho & Salimans, "Classifier-Free Diffusion Guidance", 2022 — https://arxiv.org/abs/2207.12598
- Sanchez et al., "Stay on Topic with Classifier-Free Guidance" (CFG for LLMs), 2023 — https://arxiv.org/abs/2306.17806
- Schiff/Kuleshov et al., "Simple Guidance Mechanisms for Discrete Diffusion Models" (D-CFG), 2024 — https://arxiv.org/abs/2412.10193 ; code https://github.com/kuleshov-group/discrete-diffusion-guidance
- "Improving Classifier-Free Guidance in Masked Diffusion", 2025 — https://arxiv.org/abs/2507.08965
- "What Exactly Does Guidance Do in Masked Discrete Diffusion Models", 2025 — https://arxiv.org/pdf/2506.10971
- Wang et al., "Diffusion LMs Are Versatile Protein Learners" (DPLM), 2024 — https://arxiv.org/abs/2402.18567 (CFG in App. D.5)
- "DPLM-2: A Multimodal Diffusion Protein LM", 2024 — https://arxiv.org/abs/2410.13782
- Madani et al., "ProGen", 2020 — https://arxiv.org/abs/2004.03497 (tag dropout 0.4)
- Zhang et al., "PRO-LDM", Advanced Science 2025 — https://advanced.onlinelibrary.wiley.com/doi/full/10.1002/advs.202502723
- "CFP-Gen", 2025 — https://arxiv.org/html/2505.22869

---

## 2. Exemplar / Retrieval / Neighbor-Embedding Conditioning

### 2.1 Conditioning on a reference example's embedding (general ML) [well-established]

- **Retrieval-Augmented Diffusion Models (RDM)** (Blattmann/Rombach 2022): trainable conditional diffusion model + fixed external DB + non-trainable retrieval fn. At **training**, for each target image retrieve its **k nearest neighbors** (CLIP embedding distance) and condition on the *encoded neighbors*. At **inference** the DB can be swapped → unconditional / class-conditional / text / style transfer just by changing what you retrieve. Directly analogous to "condition on neighbors of the same class." — https://arxiv.org/abs/2204.11824
- **kNN-Diffusion** (Sheynin et al. 2022): conditions on the kNN of a CLIP embedding; neighbors bridge the train/inference distribution gap. — https://arxiv.org/abs/2204.02849
- **Re-Imagen** (Chan et al. 2022): retrieval-augmented text-to-image; retrieved (image,text) pairs condition generation, esp. for rare entities. — https://arxiv.org/abs/2209.14491
- **ReMoDiffuse** (motion), **Prototype-Guided Diffusion** (2025, conditions on class prototypes without an external memory) — same family.

### 2.2 The trivial-copying / shortcut failure mode + mitigations [well-established]

When you condition on (a function of) the target itself, the model learns the identity shortcut (copy the condition). Standard mitigations, in rough order of importance for our setting:

1. **Condition on a DIFFERENT example, never the target.** Most robust structural fix. (RDM/kNN retrieve *other* images; the model can't copy because the conditioning sequence ≠ the target.) — directly supports the A3 plan.
2. **Information bottleneck on the condition.** Compress the reference so it carries *semantic class identity but not residue-level detail*. Canonical example — **Paint-by-Example** (Yang et al. 2022): to stop copy-paste of the exemplar, they (a) pass only the **single CLIP class token** (a 1-vector bottleneck) of the reference, (b) apply **strong augmentations** to the reference, (c) use a self-reference training scheme, (d) add CFG to dial similarity back up. — https://arxiv.org/abs/2211.13227. A per-class **mean embedding** (your current cond.) is itself a strong bottleneck — which is partly why it under-steers; an *individual* neighbor embedding carries more usable signal.
3. **Conditioning-input noise.** Add Gaussian noise to the conditioning embedding during training (and optionally inference) to destroy copyable high-frequency detail and force reliance on coarse/semantic content. Common in exemplar/audio-conditioning work.
4. **Stop-gradient** on the condition encoder (or freeze it) so the main model can't co-adapt the encoder into a copy channel. (You already use frozen ESM embeddings — keep that.)

### 2.3 Same-class-neighbor conditioning — precedent & best practice [protein-confirmed]

This is the closest published analog to the A3 plan, and it is **well-precedented in proteins**:

- **RAG-ESM** (Bitbol lab, 2025) [protein-confirmed, strongest analog]: conditions a pretrained **ESM2** LM on **homologous sequences** via a few **cross-attention** params, trained with a **discrete-diffusion objective**, and conditions on homologs at inference → SOTA among sequence-based models for **conditional protein generation + motif scaffolding**. This is essentially "DPLM-style discrete diffusion + same-family neighbor cross-attention conditioning" — i.e. a proof-of-concept for A3. Code: https://github.com/Bitbol-Lab/rag-esm ; paper https://www.biorxiv.org/content/10.1101/2025.04.02.646805v1.full
- **ProtMamba** (2024) [protein-confirmed]: prompt with several **same-cluster (homolog)** unaligned sequences as context → homolog-conditioned generation beats random-cluster context. Validates the **same-class neighbor vs random** distinction: they explicitly measure Hamming distance to context for "same cluster" vs "random cluster" sequences. — https://www.biorxiv.org/content/10.1101/2024.05.24.595730
- **Data-augmentation for label-specific homolog generation** (2025): generates within-family homologs while avoiding verbatim replication. — https://arxiv.org/abs/2507.15651
- **Bio-xLSTM**, **ProFam** — in-context / family-conditioned generation, same spirit.

**How to sample the neighbor (best practice):**
- **Random same-class neighbor** is the simplest, well-supported default (RDM uses kNN; ProtMamba samples within-cluster; for a *small* class, "random same-class" ≈ "kNN" anyway). Random same-class also acts as **data augmentation / regularization** because each target sees many different conditioning partners across epochs — this is desirable.
- **kNN in embedding space** (e.g., nearest same-class neighbor by ESM-embedding distance) gives a *tighter*, more informative condition, but risks the model leaning on a near-duplicate (closer to the copy regime) — pair with noise/bottleneck if used. RDM/kNN-diffusion use k≈4–20 neighbors and **sample one** per step/example.
- Practical hybrid: **random same-class neighbor at training** (for diversity/regularization) + **optionally add embedding noise**; reserve kNN for inference-time targeting.

**Add noise on top?** [well-established in general, thin in protein] Yes if (a) you use kNN/near-duplicate neighbors, or (b) you observe copying. A small Gaussian on the (continuous) neighbor embedding is the cheap lever. If neighbors are already "different sequence, same class," copying is structurally blocked and noise is optional.

### 2.4 Inference-time exemplar strategies [mixed: general well-established, protein thin]
- **Condition on a real held-out class member** — most faithful; matches training distribution. Default.
- **Centroid / medoid of the class** — closest to your current mean-embedding; smooth but least specific (and is exactly what under-steers now). A **medoid** (a real sequence nearest the centroid) is better than a synthetic mean because it's on-manifold.
- **Interpolate between two same-class members' embeddings** — explore the convex region "between" exemplars; standard latent-interpolation trick; produces intermediate variants.
- **Noised exemplar** — add Gaussian noise to a real member's embedding to "explore around" that region; controllable diversity knob at inference. Combine with CFG `w` to trade specificity vs. exploration.

### Key citations — Topic 2
- Blattmann/Rombach et al., "Retrieval-Augmented Diffusion Models" (RDM), 2022 — https://arxiv.org/abs/2204.11824
- Sheynin et al., "kNN-Diffusion", 2022 — https://arxiv.org/abs/2204.02849
- Chan et al., "Re-Imagen", 2022 — https://arxiv.org/abs/2209.14491
- Yang et al., "Paint by Example", 2022 (anti-copy: CLIP-token bottleneck + strong aug + self-ref + CFG) — https://arxiv.org/abs/2211.13227
- "RAG-ESM: Improving pretrained PLMs via sequence retrieval", 2025 — https://www.biorxiv.org/content/10.1101/2025.04.02.646805v1.full
- "ProtMamba: homology-aware alignment-free protein SSM", 2024 — https://www.biorxiv.org/content/10.1101/2024.05.24.595730
- "Data augmentation enables label-specific generation of homologous protein sequences", 2025 — https://arxiv.org/abs/2507.15651

---

## 3. Recommended options for DPLM A3

All assume: keep ESM-2 backbone, frozen ESM embeddings for the *condition*, inject via the existing cross-attention pathway (the RAG-ESM precedent shows a few cross-attn params suffice). The current per-class **mean** embedding is the prime suspect for non-steering: it's an over-aggressive bottleneck (one static vector/class) AND offers no unconditional contrast for guidance.

### Option A — Same-class-neighbor conditioning (RAG-ESM-style). *Recommended primary.*
- **Train:** for each target TPS, condition on the ESM embedding of a **randomly sampled DIFFERENT sequence of the same first-cyclization class**. Resample the neighbor every epoch (free data augmentation; structurally blocks copying since cond ≠ target).
- **Inference:** condition on a real held-out class member (or class **medoid**).
- **Knobs:** neighbor = random same-class (k = all class members, sample 1). No noise initially.
- **Rationale/tradeoff:** strongest protein precedent (RAG-ESM, ProtMamba). Replaces the static class mean with a *specific, informative, on-manifold* signal → should actually steer. Risk: if classes are tiny, "different" neighbor is still very close → add Option B's noise. No guidance at inference (simpler), so controllability is bounded by how well the model learned to use the condition.

### Option B — Neighbor conditioning + CFG. *Recommended if A steers but weakly.*
- **Train:** as Option A, PLUS condition-dropout: with **p_uncond = 15%** replace the neighbor embedding with a single **learned null embedding** (extra cross-attn key/value vector).
- **Inference:** logit-space CFG `logit = (1+w)·logit_cond − w·logit_null`, with a **ramped schedule** (small `w` while mostly masked, grow over denoising). Start sweep **w ∈ {1, 2, 3}**; expect best around **w ≈ 1.5–2.5**. Watch for low-complexity/repetitive sequences at high `w` (over-guidance).
- **Rationale/tradeoff:** adds an explicit fidelity↔diversity dial and an unconditional reference, the canonical fix for "conditioning doesn't steer." Native to DPLM (App. D.5). Cost: a `w` sweep and the masked-diffusion schedule caveat. Best controllability of the four.

### Option C — Neighbor + conditioning noise (anti-copy hardened). *Use if copying/low diversity appears, esp. with kNN neighbors or tiny classes.*
- **Train:** Option A, but the neighbor is the **kNN same-class neighbor** (most informative) AND add **Gaussian noise** to the (continuous) conditioning embedding (small σ, sweep e.g. 0.1–0.5 of per-dim std). Optionally also `p_uncond = 15%` for CFG.
- **Inference:** real member or **noised exemplar** to control exploration; CFG optional.
- **Rationale/tradeoff:** Paint-by-Example recipe (informative reference + bottleneck/aug to kill the copy shortcut). Max signal with copy protection; one extra hyperparameter (σ). If σ too high you wash out class identity.

### Option D — Keep class-label conditioning, add CFG only (cheapest A/B test). *Diagnostic baseline.*
- **Train:** current per-class conditioning (mean embedding OR a learned class-token embedding — prefer a **learned per-class token** over the frozen mean) + condition-dropout **p_uncond = 10–20%** with a learned null.
- **Inference:** logit-space CFG, ramped, **w ∈ {1,2,3,5}** sweep.
- **Rationale/tradeoff:** isolates "does the model just need a guidance dial?" from "does it need a richer condition?" Cheapest change. If D steers and A/B don't add much, the problem was guidance, not the conditioning representation. If D still doesn't steer at high `w`, the static class embedding is genuinely too weak → commit to A/B/C. Risk: per-class mean may remain too coarse to steer even with `w` — that's the informative negative result.

**Suggested order:** run **D** (diagnostic) and **A** (primary) in parallel; escalate to **B** (A + CFG) for controllability; reach for **C** only if diversity collapses / copying shows up.

### Evidence-quality flags for the recommendations
- Same-class-neighbor conditioning for proteins: **[protein-confirmed]** (RAG-ESM, ProtMamba).
- CFG existing/working in protein generators: **[protein-confirmed]** (DPLM, PRO-LDM); but specific protein-sequence `p`/`w` numbers: **[thin]** — sweep `w`, use `p=15%` as a safe general-ML default.
- Masked-diffusion guidance *schedule* (weak-early, strong-late) and logit-space product-of-experts formula: **[well-established]** in discrete-diffusion ML, **[thin]** specifically validated on protein sequences.
- Anti-copy via different-example + bottleneck + noise: **[well-established]** general ML (Paint-by-Example, RDM); for proteins the "different homolog" variant is **[protein-confirmed]**, the explicit noise-on-embedding lever is **[thin]** in protein work.
