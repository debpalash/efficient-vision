# Research synthesis → v1.1 plan

Fable 5 synthesis of four Sonnet 5 research sweeps (2026-07-18):
`docs/research/{backbones,obb-detection,distillation-quantization,training-and-data}.md`.

## Corrections to our existing plan (in order of impact)

### 1. RepConv BN placement was a latent INT8 catastrophe — fixed in design
Naive RepVGG fusion (our current `blocks.py`) produces outlier weight statistics;
INT8 collapses 20+pp (RepVGG-A0: 72.4→52.2%). **QARepVGG fix** (arXiv 2212.01593):
no BN on identity/1x1 branches, one BN after branch summation — INT8 within 1pp of
FP32, zero inference cost. Task #4 updated. This alone likely saves the 95% target
under quantization.

**Verification status (2026-07-18):** the QARepVGG block is implemented
(`efficientvision/models/blocks.py`, structural invariant pinned by
`tests/test_blocks.py::test_qarepvgg_bn_placement`), but our attempt to reproduce
the *mechanism* locally was **inconclusive** — `scripts/quant_friendliness.py`
found only ~3% difference in INT8 round-trip weight error between the two BN
placements on randomly-initialized blocks, which cannot explain a 20-point
collapse. The probe is too weak (random data, 40 steps, no real training), so this
neither validates nor refutes the paper. **The adoption currently rests on the
published result, not on a local measurement.** Re-run against trained weights in
task #13 and record the real number.

### 2. Wrong OBB baseline — re-baseline required
Our 129ms/73ms numbers are the *axis-aligned* detect task at 640px. The true
competitor is **YOLO26n-obb @1024px: 97.7ms published (Xeon)** — must be measured
locally (new task #17). All "beat YOLO26n" claims now reference the obb variant.

### 3. NMS-free ≠ big CPU win (in sparse scenes)
Rotated NMS at 5-20 objects is sub-ms. Keep NMS-free anyway: ONNX has no
rotated-NMS op (export blocker otherwise), deterministic latency. But the latency
budget must come from backbone/resolution, not from dropping NMS. Also budget the
measured one-to-one accuracy tax (~-0.5 AP).

### 4. Expect 1.2–1.5x from INT8, not 2–4x
Small models are memory/overhead-bound; Xeon-6 OpenVINO detection saw ~1.2x
end-to-end. VNNI required for the good case; ARM unverified. Resolution and
architecture remain the primary latency levers; INT8 is a multiplier, not the plan.

### 5. Angle encoding needs an ablation, not a decision
Literature validates low-channel trig encodings (FSC 2026), but YOLO26-obb itself
ships direct scalar regression + `sin²(2Δθ)·exp(-(log(w/h))²/9)` loss. No
controlled comparison exists. Task #6 now includes the A/B.

## Design deltas (adopted)

| Area | Change | Source |
|---|---|---|
| RepConv | QARepVGG BN placement; cap train-time branches | 2212.01593; MobileOne |
| Backbone | A/B grouped-conv + 2×2-kernel variants on our harness (MACpS loop); trial PConv in wide stages | CPUBone 2603.26425; FasterNet |
| Activation | ReLU6 A/B during QAT (bounded range quantizes better) | StarNet + 3 independent sources |
| Assignment | ProbIoU as o2o+o2m alignment metric; Progressive Loss (o2m weight 0.8→0.1); trial vertex-distance term | YOLO26-obb source; RHINO |
| OBB head | RN-normalize regression targets; per-channel quant on box-head convs; last reg layers may stay FP16 | YOLOv6+ 2025; synthesis |
| Distillation | PKD (feature, scale-invariant) + CrossKD (head); distill in FP32 multi-branch phase | 2207.02039; 2306.11369 |
| QAT | LSQ/LSQ+; fuse→QAT with FP32 model as teacher in the loop; selective QAT (skip stem/box head if loss concentrates) | YOLOv6 recipe |
| Training | YOLO26-nano recipe start point: ~245 epochs, light aug, mosaic off last 10 epochs, low WD, ~1-epoch warmup; multi-scale + FixRes fine-tune at deploy res; NO flat label smoothing | 2606.03748; FixRes |
| Long-tail | RFS+IRFS resampling + logit adjustment + cRT rebalancing pass (all near-free); Seesaw/EQLv2 only if insufficient | LVIS lineage; 2305.08069 |
| Augmentation | flip/scale/HSV primary; degrees/shear/perspective ≈0; visual mosaic verification (Ultralytics has open OBB corruption bugs: #10181, #9209) | community-verified |
| Rare classes | copy-paste only for <3-example classes; plan ≥50-100 instances/class | Ecology&Evolution 2025 |

## New experiments (cheap, high-information)

1. **Local YOLO26n-obb baseline** @1024 and 640 (task #17) — the real number to beat.
2. **Rotated-NMS microbenchmark** at realistic object counts — novel data; informs
   whether an o2m+NMS fallback head is worth keeping as an option.
3. **cos/sin vs direct-scalar angle** head A/B (task #6).
4. **ReLU vs ReLU6** under QAT (task #13).
5. **MACpS profiling** of block variants before locking backbone (task #4).

## Unchanged

Feasibility gate (#3) still precedes all optimization. Distillation still the
biggest accuracy lever. Resolution still the biggest latency lever. Two-model
fallback (separate cls/OBB) still on the table if unified underperforms.
