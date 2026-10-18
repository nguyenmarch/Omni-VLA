# Direct VLA — RGB-D + language + robot history

This is a new learned vision-language-action policy, separate from the frozen
P5/BC bridge. It receives measured RGB-D/calibration, instruction tokens, and
50 robot proprioception values only. A spatial CNN, recurrent text encoder and
8-frame GRU history predict five normalized 14-D actions at 25 Hz. Images/replans
are every five actions (5 Hz simulation time), matching dataset collection.
No simulator object position, expert phase, hand-coded task manager or expert
action is used to choose actions at inference. Phase and 3D centers are auxiliary
training labels only. Task ID in evaluation selects the success metric only.

This implements a compact direct VLA architecture, not an open-world foundation
model or a verified high-success system. Two known-object tasks and a known word
vocabulary are currently supported. New objects/language, contact-rich uncapping,
real robots, asynchronous direct-policy inference and validated recovery remain
future integration work. The older async/force guards remain in their own runtime;
they are not silently claimed as features of this new evaluator.

## Data and CPU smoke

`data/direct_vla_v1.h5` contains 18 successful trajectories from 20 attempts,
1,110 RGB-D frames, real five-action chunks with tail masks, two tasks and
camera/light/friction/position variations. Successful episode splits are
11 train / 3 validation / 4 test, with disjoint seeds. Failed expert attempts are
retained in the collection report, excluded from learning. Dataset is small and
success-only, so it does not supply corrective/recovery states.

For this pilot, validation profiles are IID, test profiles are camera changes,
and training profiles are lighting/friction/position changes, following the
predetermined schedule. These are intentional different distributions; do not
claim that every domain appears in training or that the test is an IID score.
More demonstrations per task/domain and corrective examples are necessary for
high-quality rollout behavior. No test frame is used to fit robot normalization
or select a checkpoint. Training phrases are augmented within the training split;
held-out known-word paraphrases are provided for evaluation, not open vocabulary.

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.collect_vla \
  --episodes 20 --seed 80000 --randomize --output data/direct_vla_v1.h5
.venv/bin/python -m scripts.train_vla --dataset data/direct_vla_v1.h5 \
  --output weights/direct_vla_smoke_v2 --epochs 2 --batch-size 4 --max-batches 2
```

Smoke caps batches deliberately and is flagged `smoke_only` in checkpoints and
rollout reports. It verifies the pipeline, not convergence. Full training uses
all training/validation frames and selects `best.pt` by validation action MSE;
loss is not a substitute for closed-loop evaluation. Checkpoints include weights,
robot statistics, optimizer, Torch CPU/CUDA RNG, dataset hash, splits, timing and
config. Resume with identical dataset/model/history/batch/smoke settings:

```bash
python -m scripts.train_vla --dataset /path/direct_vla_v1.h5 \
  --output /path/direct_vla --epochs 100 --resume /path/direct_vla/last.pt
```

CPU smoke results, hashes and verification are recorded in
`docs/docs/policies/DIRECT_VLA_HANDOFF.json` after the local checks complete.

## Kaggle GPU handoff

1. Upload `artifacts/direct_vla_source.zip` as a Kaggle source dataset.
2. Upload `data/direct_vla_v1.h5` as a separate dataset.
3. Import `notebooks/kaggle_train_direct_vla.ipynb`, attach both datasets and enable GPU.
4. Run all cells; Save Version with outputs; download `direct_vla_outputs.zip`.
5. Extract into a new `weights/direct_vla_kaggle/` directory and evaluate locally.

Kaggle GPU training has completed for the first candidate: 100 epochs, best
checkpoint at epoch 99 with validation action MSE 0.002865. Dataset hash, splits,
normalization and CPU forward were verified on import. See
[GPU candidate report](./DIRECT_VLA_KAGGLE_RESULTS.md) for closed-loop results.
The notebook detects CUDA, trains the full 100 epochs and packages checkpoints/metrics. It can resume from
`last.pt` using the CLI above. Local artifacts are ignored by Git; transfer them
using the handoff checksums. Never replace the frozen P5/BC checkpoint.

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_direct_vla \
  --checkpoint weights/direct_vla_kaggle/best.pt --task 0 \
  --episodes 50 --seed 90000 --output artifacts/vla_stow_fresh.json
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_direct_vla \
  --checkpoint weights/direct_vla_kaggle/best.pt --task 1 \
  --episodes 50 --seed 91000 --output artifacts/vla_open_fresh.json
```

Reserve those seeds before inspecting outcomes. Use new ranges for each OOD
profile (`--profile camera|lighting|friction|position|combined`) and report them
separately. Check image/text/history dependence with `--ablation drop-image`,
`swap-text` and `drop-history` on matched development seeds. Explicit task-scoring
labels do not change policy outputs. RGB-D support failures and unsupported words
are rejected; rollout success uses ground truth only to score/terminate.

Acceptance for promoting a trained candidate: both tasks ≥90% observed IID,
Wilson lower 95% bound ≥80%, measured OOD degradation and meaningful ablations,
at least three training seeds, and no oracle input in the action path. If these
fail, collect corrective demonstrations and train another candidate using
validation only. Do not tune on the declared fresh test seeds.
