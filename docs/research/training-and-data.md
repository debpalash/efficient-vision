# Research: training recipes & long-tail data strategy (2023–2026)

Sonnet 5 research agent findings, 2026-07-18. Ranked by expected impact.
Synthesis into the project plan lives in `docs/research-plan.md`.

## Long-tail (the project's hardest requirement)

### 1. Repeat Factor Sampling (RFS) — ADOPT
**LVIS, Gupta et al., CVPR 2019.** Oversample images containing rare-class
instances by a frequency-derived repeat factor. Dataloader-only change, still the
baseline every 2024-2025 paper compares against. Essentially free; the single
highest-leverage long-tail lever.

### 2. Logit Adjustment — ADOPT
**Menon et al., ICLR 2021 (arXiv 2007.07314); still the reference cheap baseline
in 2024-2026 surveys.** Fixed class-frequency-derived logit offset at train time
(or post-hoc). A few lines of code; near-SOTA baseline through 2025. Combine with
RFS as the two cheapest/highest-value moves.

### 3. Instance-aware RFS (IRFS) — ADOPT
**arXiv 2305.08069 (2023).** RFS counts images per class; IRFS also counts
instances (a rare class can appear many times in one image). +50% *relative* AP on
rare classes over vanilla RFS on LVIS. Trivial addition to the resampler.

### 4. Decoupled training / classifier re-training (cRT, SimLTD) — ADOPT
**Kang et al., ICLR 2020; SimLTD, CVPR 2025 (arXiv 2412.20047); LOS, ICLR 2025.**
Train normally, then freeze the backbone and re-train only the classifier head for
a short phase with class-balanced sampling. SimLTD's 3-stage recipe (head-class
pretrain → tail transfer → balanced joint fine-tune) set LVIS records without
extra labels. Maps directly to our workflow: a 5-10%-extra-epochs rebalancing pass.

### 5. FRACAL post-hoc calibration — CONSIDER
**arXiv 2410.11774, CVPR 2025.** Training-free logit calibration using spatial
fractal dimension. Up to +8.6% rare-class AP on LVIS. Inference-time only — cheap
to trial; LVIS-scale gains may not transfer to a small proprietary set.

### 6. Loss reweighting (Seesaw / EQLv2 / EFL) — CONSIDER (second pass)
**CVPR 2021/2022 lineage.** Real gains (EQLv2: +14-18 rare-class AP on LVIS) but
each needs per-dataset hyperparameter tuning. Invest only after RFS + logit
adjustment + IRFS + cRT are exhausted — worse gain per engineering hour.

### 7. Class-agnostic regression branch — CONSIDER (adapted)
**Rectify the Regression Bias, arXiv 2401.15885 (2024).** Long-tail work obsesses
over classification, but box-regression loss is also systematically worse for rare
classes; a shared class-agnostic regression head blended with the per-class head
gave +6 rare-class AP. Their mechanism assumes two-stage; for our single-stage OBB
head the transferable idea is keeping localization + angle fully class-agnostic —
which our design already does. Validates that choice; watch rare-class angle error
in the eval harness (task #15).

## Small-model training recipe

### 8. YOLO26-nano production recipe — ADOPT as starting point
**docs.ultralytics.com/guides/yolo26-training-recipe; arXiv 2606.03748.** For the
nano scale specifically: LR 0.0054 with steep cosine decay, ~1 epoch warmup, low
weight decay (6.4e-4), mosaic p=0.91 **disabled for the last 10 epochs**, light
mixup (0.012) and copy-paste (0.075), rotation/shear essentially zeroed, ~245
epochs (vs 40-80 for large models). Key insight: **small models underfit — train
them longer with lighter augmentation**; heavy augmentation is for big models with
spare capacity.

### 9. Augmentation annealing — CONSIDER (validates #8)
**RT-DETRv2, arXiv 2407.17140.** Strong augmentation early, weak late; higher LR
for smaller backbones. Independent confirmation of the same principles.

### 10. Label smoothing — SKIP (in flat form)
**MaxSup arXiv 2502.15798 + long-tail literature.** Flat label smoothing assumes
balanced labels; on long-tail data it suppresses confidence on rare classes.
Given hundreds of long-tail classes, do not enable flat ε smoothing (a silent
default in many detection configs). Only class-frequency-aware variants, if any.

## OBB augmentation

### 11. Rotation + mosaic/perspective corrupts OBB labels — ADOPT the avoidance
**MMRotate (arXiv 2204.13317); Ultralytics GitHub issues #10181, #9209, #6138,
yolov5 #12735 — open, acknowledged bugs.** Combining rotation-degree augmentation
with mosaic/perspective/scale produces distorted boxes, especially at tile edges,
due to scale+rotate+clip ordering in the xywhr representation. Reproduced across
multiple community reports. **This independently confirms our task #9 risk flag —
in production code, not just theory.**
**Rule: flip + scale + HSV as primary OBB-safe augmentations; mosaic kept but
visually verified at tile boundaries; degrees/shear/perspective zeroed or near-zero
— consistent with YOLO26-nano's own config.**

### 12. Select-Mosaic — CONSIDER
**arXiv 2406.05412 (2024).** Bias mosaic tile selection toward regions containing
rare/small instances instead of uniform random. Pairs naturally with RFS
oversampling; easy to reimplement.

## Rare-class rescue & data budgeting

### 13. Copy-paste — CONSIDER, with a rule
**Ecology & Evolution 2025 (PMC12588685); Ghiasi CVPR 2021.** At ~500 img/class:
+8%±2% mAP, but *hurt* 17% of classes (context/lighting mismatch). In the 1-8
img/class regime: helps at 1-2, **degrades at 4-8**. ~50 segments/class reaches
near-peak.
**Rule: copy-paste only for classes with <3 real examples; stop by ~4-8; watch
lighting mismatch.**

### 14. FixRes final fine-tune at deployment resolution — ADOPT
**arXiv 1906.06423; still the standard reference.** Train multi-scale for
representation quality, then short low-LR fine-tune at the exact deployment
resolution. Recovers most of the resolution-mismatch gap cheaply. Directly
relevant since we intend to deploy below 640px if EDA allows.

### 15. Multi-scale training — ADOPT
Standard YOLO practice; naive high-res-train/low-res-test costs real accuracy
(cited: 38.7→34.6 mAP at 800→400px without adaptation). Train across a band that
includes the deployment resolution, then FixRes fine-tune.

### 16. Images-per-class planning heuristic — ADOPT (as heuristic)
**Synthesis: DeFRCN/DE-ViT few-shot curves + camera-trap saturation data.** No
clean "N images = 95%" number exists. Accuracy climbs steeply from ~1→50
instances/class, plateaus by ~300-500. COCO few-shot SOTA reaches only 20-30 AP at
10-30 shots — 95% is plausible *only* because our classes are fixed and closed-set.
**Budget: ≥50-100 real instances per class minimum; treat <10-20 as few-shot tier
needing copy-paste + oversampling; diminishing returns past ~300-500.**

---

## Recommended stack (ordered by adoption cost)

1. **Free, do first:** RFS + IRFS + logit adjustment + cRT rebalancing pass.
2. **Training mechanics:** YOLO26-nano recipe as starting point; multi-scale train;
   FixRes fine-tune at deployment resolution; no flat label smoothing.
3. **OBB safety:** flip/scale/HSV primary; near-zero rotation/shear/perspective;
   visual verification of mosaic at tile boundaries (task #9).
4. **Rare-class rescue:** copy-paste only under the <3-example rule; budget
   50-100+ instances/class when planning labeling.
5. **Second pass if needed:** FRACAL trial; Seesaw/EQLv2 with tuning budget.
