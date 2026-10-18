# Patch supervision v3 — pilot 11/10/2026

Migration code and CPU pilot are complete; **perception is still not usable for
closed-loop manipulation**. Kaggle inputs are prepared, GPU training has not been
run by this local session. BC v2 and all v2 datasets/checkpoints remain unchanged.

## 1. Root cause and experimental findings

The old learned offset used a single pixel, while the proposed runtime uses a
patch statistic. Mixing these two contracts changes the surface reference.
The migration fixes this mismatch, but it does not fix wrong heatmap pixels or
mixed foreground/background patches. Geometry gates are not semantic confidence.

There is also a checkpoint selection pitfall: invalid=1 metre penalized error
alone selected epoch 1, with test coverage 62.6% but mean accepted error 23.79cm
and zero targets within 1.5cm. Higher acceptance was not better grounding.
The trainer now prioritizes accurate coverage (valid AND error <=1.5cm / all
validation targets), then penalized error. Every future epoch is retained.

The first CPU run had saved only epochs 1 and 10. `selected.pt` is epoch 10,
selected using **validation only among those retained candidates**. It is not
the optimum among all ten epochs: logs show epoch 7 had 6 accurate validation
targets, but its weights were not retained by the original selection rule.
The corrected Kaggle trainer selects each epoch with the new rule; this local
pilot is an engineering diagnostic, not a claim of improvement.

## 2. Supervision and decoder contract

For projected target pixel u*, camera-center depth z*, and depth patch P:

\[
z_s=\operatorname{median}(D[P_{valid}]),\quad
\delta z^*=z^*-z_s,\quad
p_w=T_{wc}[(z_s+\hat{\delta z})K^{-1}[\hat u,\hat v,1]^T;1].
\]

Patch 7x7, >=6 valid pixels AND >=20% support, depth range 0.05–5m,
IQR <=0.10m, corrected XYZ inside configured workspace; eligible label requires
|offset| <=0.08m. No single-pixel fallback. Clamping is diagnostic, never acceptance.
Image-border padding is excluded. Robot mask pixels can be excluded by the decoder.

`data/grounding_patch_v3_256.h5` is a versioned copy with hash lineage, median
surface, offset, eligibility, support and IQR. At GT projected pixels,
2549/2588 targets (98.49%) pass label gates. This is an eligibility heuristic,
not independently verified visibility or predicted coverage.
Train: 1676 eligible targets; validation: all 458; test: all 428. Episode splits
are unchanged/disjoint. Evaluation includes rejected and occluded targets.
Training: seed42, 10 epochs, batch32, CPU2threads, AdamW0.0005, Gaussian sigma6px,
heatmap cross-entropy + SmoothL1(offset/0.05). The network still takes RGB/text;
depth is used by the decoder, not an RGB-D encoder.

## 3. Measured verification

Same all-target split and median decoder for old/new networks:

| Network | Split | Accepted / all | Mean / P95 error on accepted | Accurate <=1.5cm / all |
|---|---|---:|---:|---:|
| Old 256 offsets + patch decoder (diagnostic) | val | 171/458 (37.3%) | 8.65 / 20.42cm | 7/458 |
| Patch v3 epoch10 retained candidate | val | 167/458 (36.5%) | 8.69 / 19.66cm | 5/458 |
| Old 256 offsets + patch decoder (diagnostic) | test | 129/428 (30.1%) | 9.98 / 20.42cm | 6/428 |
| Patch v3 epoch10 retained candidate | test | 144/428 (33.6%) | 10.23 / 25.21cm | 3/428 |

Coverage changed, accuracy did not improve. Test heatmap error is 6.09px.
Do not compare these conditional means directly to the older visible-only
15.40cm test benchmark. JSON reports include per-target/phase errors, gates,
image/text ablations and all-target accuracy denominators.

Closed-loop seeds9000–9004: **0/5**, all rejected at initial perception before
any action, because both targets must be valid to initialize the manager.
This verifies fail-closed behavior; it does not validate task performance.
See [selected test](../../artifacts/patch_grounding_v3/selected_test.json),
[selected validation](../../artifacts/patch_grounding_v3/selected_val.json),
[loop](../../artifacts/patch_grounding_v3/closed_loop_selected.json).

## 4. Wrist and FK integration boundary

`simulation/multicamera.py` captures stamped synchronous head256/wrist224 RGB-D,
K/T, same simulation time and sequence. Its **MuJoCo MJCF FK backend** renders
robot-only geometry from qpos, compares robot depth with measured depth (1cm
agreement), and preserves nearer foreground surfaces. Simulator scene
segmentation is used only for mask IoU auditing, not policy inputs.

Seed9000 expert rollout: 22 snapshots/camera covering phases0–8; mean/min IoU
head0.999407/0.998383, wrist0.999961/0.999545. The original wrist mask missed
near-camera robot surfaces because of the landmark decoder's 5cm minimum.
Masking now accepts positive near depth separately from landmark-valid depth.
Audit capture including segmentation p50/p95 =108.08/129.40ms, excludes model
inference; this is not an asynchronous 10Hz runtime benchmark.

[Head overlay](../../artifacts/patch_grounding_v3/multicamera_mask_final_head_cam_phase0.html),
[wrist overlay](../../artifacts/patch_grounding_v3/multicamera_mask_final_right_wrist_cam_phase1.html),
[mask report](../../artifacts/patch_grounding_v3/multicamera_mask_final.json).

This is a synchronous simulation capture/mask component, **not Pinocchio/URDF**,
not a wrist-trained policy or head/wrist fusion controller. The v3 policy runner
uses head only; masks have not changed its training/runtime input contract.
Next: collect masked per-camera data, audit target visibility, train/evaluate
per-camera models, then integrate timestamp/validity-aware fusion. Do not feed
224 wrist images to the 256 head checkpoint. Pinocchio requires a matching
robot geometry/URDF and parity audit. Async capture remains pending.

## Reproduce and Kaggle handoff

Code: `scripts/prepare_patch_grounding.py`, `utils/patch_grounding_dataset.py`,
`scripts/train_patch_grounding.py`, `scripts/eval_patch_grounding.py`,
`scripts/select_patch_checkpoint.py`; runtime v3 in `models/grounding3d.py`.
Historical v2 decoder remains solely for baseline reproduction.

Upload these three prepared files to Kaggle:

1. `artifacts/kaggle_patch_v3_source.zip`
2. `data/grounding_patch_v3_256.h5`
3. `notebooks/kaggle_train_patch_grounding_v3.ipynb` (import as notebook)

Enable GPU; notebook installs MuJoCo for shared imports, no simulation rendering
is required for training. It uses 10 epochs, corrected checkpoint selection,
all-target val/test reports, and exports `patch_v3_outputs.zip`. Python notebook
cells were syntax checked; actual Kaggle execution is pending. Any new GPU run
must be evaluated on Dell; GPU loss is not a manipulation success metric.

```bash
# New training directory is required; keep prior artifacts.
.venv/bin/python -m scripts.train_patch_grounding \
  --dataset data/grounding_patch_v3_256.h5 --output weights/patch_new_run --epochs 10

# Existing retained CPU candidate, fail-closed experiment:
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/grounding_patch_v3_cpu_pilot/selected.pt \
  --episodes 5 --seed 9000 --output artifacts/patch_new_eval.json

# Synchronous cameras and FK mask overlays:
MUJOCO_GL=egl .venv/bin/python -m scripts.audit_multicamera_mask \
  --snapshots --output artifacts/multicamera_new_audit.json

.venv/bin/python -m pytest -q
```

Acceptance is still pending: proposed P95 <=1.5cm with declared coverage,
>=90% task success per held-out seed set. OT-CFM, learned monitor, adaptive
recovery, ONNX grounding and ROS2 are not implemented in this migration.

Verification: 38 tests passed, including patch-label outliers, all-target rejection
denominators, accuracy-first selection, runtime fail-closed behavior and
per-camera FK capture/mask invariants. Dataset hashes and packaged source match.
