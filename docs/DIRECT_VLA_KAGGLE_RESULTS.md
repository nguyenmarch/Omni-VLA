# Direct VLA Kaggle candidate — 14/10/2026

The first full GPU candidate has been imported into
`weights/direct_vla_kaggle/`; frozen P5/BC weights were not replaced.
The archive contains `best.pt`, `last.pt` and 100 training-log rows.

| Check | Result |
|---|---|
| Training completed | 100/100 epochs, CUDA Torch 2.11.0 |
| Selected checkpoint | epoch 99, validation action MSE 0.002865113 |
| Dataset | SHA-256 matches `direct_vla_v1.h5` |
| Schema/splits/robot normalization | verified against the local dataset |
| CPU | strict model load and finite forward pass on Torch 2.14.1 CPU |
| Smoke flag | false; this is the full trained candidate |

Importer: `python -m scripts.import_direct_vla` validates everything before
creating the candidate directory and refuses to overwrite existing outputs.
Checkpoint metadata does not include a training-source digest, so byte-for-byte
source equivalence cannot be independently established from the archive alone.

## Offline held-out camera episodes

271 test frames from four held-out teacher episodes: action MSE **0.004673**,
auxiliary phase accuracy **98.52%**, auxiliary bottle/handle center mean error
**4.28/3.46 cm**. These points/phase logits are auxiliary outputs, not oracle
inputs to action prediction. They are not interchangeable with P5 grounding
metrics. Highest phase action errors are release (0.03448) and retreat (0.01680).
Temporal frames within an episode are correlated; they are not 271 independent
rollout trials. Test was evaluated after validation-only checkpoint selection.

## Closed-loop protocol

Pilot: stow 2/5 on 90000–90004; open 5/5 on 91000–91004. The same frozen
checkpoint is evaluated on 50 seeds per task (90000–90049 and 91000–91049).
Pilot seeds are included in those sets, not counted as additional independent
trials. No weights, thresholds or controller behavior were tuned after seeing
pilot outcomes. Final results and input/report hashes are saved in
`DIRECT_VLA_KAGGLE_RESULTS.json` after the suite completes.

The action path receives measured RGB-D/calibration, text and robot-only history;
it does not receive object position, task phase or scripted actions. Evaluator
success flags only score/terminate episodes. Scene/objects/camera remain fixed,
with the usual ±1.5 cm bottle reset jitter. Opening success checks drawer opening
>0.16 m; it does not prove that no other object was moved, or that the system
understands unrestricted instructions. Stow requires bottle placement/release
and drawer closure according to the existing simulation predicate.

Matched development ablations on stow pilot seeds remove visual features or
replace the instruction with the opening task. Opening is a subgoal of stowing,
so a text swap scored only on drawer opening would not be a discriminative
language test. The stow criterion is used for the swap test instead. Zero visual
features are an out-of-training-distribution ablation, not a calibrated test of
all contributions of RGB versus depth. Concurrent independent CPU evaluation
jobs introduce resource contention; timing is diagnostic, not isolated latency.

## A diagnosed failure

Seed 90001 reaches the full 1,500-action budget with the drawer open, but bottle
placement false. The bottle ends outside the drawer at z=0.425 m. This shows
that successful opening does not establish grasp/transfer/place robustness.
Ground truth was inspected only after rollout for diagnosis and never supplied
to policy. A single case does not isolate the cause; rollout corrective examples
and per-stage diagnosis are needed before another training round.

Keep this as a candidate until both tasks meet observed IID success ≥90% and
Wilson lower 95% bound ≥80%, then test OOD, known-word paraphrases and multiple
training seeds. A low offline action loss alone does not satisfy those gates.
