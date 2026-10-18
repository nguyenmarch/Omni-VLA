# Visual reliability work — 14/10/2026

This revision adds failure diagnostics, a controlled IID/OOD evaluation protocol,
bounded asynchronous RGB-D capture/inference, and an observable force detector.
BC/P5 weights and historical benchmark reports remain frozen. These engineering
changes do not establish 90% task success or open-world generalization.

## Failure diagnosis

Development seeds 12002/12015/12016/12018 were rerun with per-action phase,
goal error, observation age and train-state envelope diagnostics. No expert
label/action is used by the visual policy. The envelope is fit only on the train
episodes of `drawer_dagger_v2.h5`, separately for each phase; it measures the
fraction of 57 continuous condition coordinates outside train quantiles 1–99%.
This is a descriptive distribution check, not a calibrated OOD detector.

| Seed | Timeout phase | Final left/right goal distance | Mean coordinates outside envelope |
|---|---|---|---|
| 12002 | descend | 20.53 / 11.88 cm | 55.0% |
| 12015 | retreat | 50.36 / 2.16 cm | 49.3% |
| 12016 | transfer | 21.55 / 1.92 cm | 49.6% |
| 12018 | retreat | 30.28 / 0.84 cm | 48.1% |

P95 landmark age was 0.16 seconds of simulation time. Large terminal distances
are inconsistent with a small transition-tolerance adjustment being sufficient.
These observations support investigating corrective state coverage, phase entry
and perception bias; they do not identify a unique causal explanation. No
transition threshold was tuned against the new evaluation seeds.

Seed 12013 was rerun with three consecutive perception rejections permitted;
it still failed in descend. Rejection retries alone did not solve its failure.
Reports are under `artifacts/visual_reliability_20261014/`; full development traces
were generated under `/tmp`. Fresh-suite traces are retained beside each report.

## Runtime and phase diagnostics

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --seeds 12002 12015 \
  --trace artifacts/new_diagnosis.jsonl \
  --diagnostic-dataset data/drawer_dagger_v2.h5 \
  --output artifacts/new_diagnosis.json
```

Each report includes phase transitions, per-episode landmark errors, a trace
summary, Wilson 95% interval, work/cycle timing, observation age, deadline misses,
rejections and held ticks. Oracle landmarks are compared against the exact
capture snapshot only for scoring; they never enter the worker or controller.
Reports and traces are exclusive: existing paths are rejected before simulation.

Synchronous perception stays the default. `--async-perception` creates a worker
with its own cloned MjModel/MjData and renderer. Only copied integration-state
snapshots cross the boundary; rendering, inference and OpenGL cleanup run on the
worker. There is at most one running request and one replaceable pending request.
Episode-generation checks prevent old results from being published after reset.
Ages are checked in both simulation and wall time; future/stale results are
rejected. While waiting, Cartesian motion is zero and current finger opening is
held. The task phase does not advance using stale perception. Three consecutive
invalid results or a one-second simulation gap terminate the episode explicitly.
These limits are configurable and recorded in reports.

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --seeds 12000 \
  --async-perception --perception-every 1 --slip-monitor \
  --output artifacts/new_async.json
```

Async runs are paced at 25 Hz to prevent headless simulation from racing ahead
of perception. `--realtime` paces sync mode for a comparable timing experiment.
Physics/controller integration is still synchronous inside `env.step()`; this
is not an independent 250 Hz wall-clock servo. Deadline misses refer to the
40 ms action budget. Timing samples are retained in bounded buffers in GUI loop
mode; aggregate tick/success counts cover the run.

Development seed 12000 failed with async perception at both capture intervals
1 and 5. The initial every-5 test had P95 control work 11.93 ms and P95 observation
simulation age 0.28 s, with no 40 ms work-budget misses on that run. This is an
engineering timing check, not evidence that delayed observations preserve BC
success. The every-1 run used the corrected world-frame force detector and
reported no slip events, but timed out in transfer. Retain sync for the primary
baseline until delayed-observation training/evaluation succeeds.

## Generalization protocol

Prepare the plan **before inspecting outcomes**:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --output artifacts/new_visual_suite --seed 40000 \
  --iid-episodes 100 --ood-episodes 20
MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/new_visual_suite
```

The plan freezes seeds, policy/grounding/dataset hashes, source/assets/dependency
hash and runtime options. It rejects overlap with dataset seeds, previous rollout
reports and previously prepared plans under `artifacts/`. Use a fresh range when
changing/tuning the model. Completed profile reports can be resumed only when
plan identity and requested seeds match. An interrupted, incomplete profile must
be investigated rather than silently accepted as a complete benchmark.

The 14/10 plan reserves 30000–30099 for IID and 20 seeds each starting at
31000/32000/33000/34000/35000 for camera/lighting/friction/position/combined.
Profiles apply head-camera translation ±2 cm, light scale 0.65–1.35, sliding
friction scale 0.7–1.3, and initial bottle XY jitter ±4 cm. Model parameters are
restored before each profile application. Bottle perturbation happens only at
reset, never during manipulation. Camera matrices reflect the changed camera.
Report each profile independently; no pooling IID with OOD to meet a gate.
Summary gates distinguish observed IID success ≥90% from Wilson lower 95%
bound ≥80%. This is one training seed and the same objects; multiple training
seeds, new meshes/textures, calibration noise and broader task language remain
future experiments.

## Force detection and recovery scope

`WrenchSlipMonitor` consumes wrist force, finger opening, EE position and wrist
rotation from robot FK. It never accepts simulator object coordinates. Force is
transformed into the world frame and filtered with a 15-frame rolling median.
An open-gripper reference removes unloaded wrist weight; positive world-Z load
calibrates the payload for the scene's vertical gravity. Detection requires both
relative loss and support below 0.3 N, corroborated by finger opening/collapse.
This avoids treating normal grasp rotation as loss. High hand speed suppresses
decisions. Missing calibration remains `uncalibrated`. Grasp calibrates load;
carry phases enable detection; placement unloading does not trigger a stop.

Development seeds 14000–14004 were used to diagnose/calibrate the detector.
A separate check on 14500–14504 then detected **5/5 injected releases**, with
**0/5 clean-episode alerts**, and delay **0.40–0.48 s** from injection. This is
five scripted cases per group, not a broad robustness claim. The first detector
revision failed calibration (0/5 detections); intermediate prototypes produced
false positives. Reports retain these failures alongside the final holdout audit.
Physical regression tests cover an unperturbed carry and injected release, with
oracle object inputs poisoned for the detector.

`--slip-monitor` records detector states/events in the visual runner.
`--stop-on-slip` holds pose and terminates on a detection. This is a guarded stop.
The state/expert evaluator can now route detection through force as well:

```bash
.venv/bin/python -m scripts.eval_closed_loop --expert --recovery \
  --recovery-detector wrench --inject-slip --episodes 3 --seed 14500 \
  --output artifacts/new_wrench_recovery.json
```

This selects observable force/finger detection and bounded FSM preemption while
retaining **oracle scripted regrasp geometry**. The report labels both decisions
explicitly. Three injected-slip recovery trials achieved **0/3 full tasks**:
two recovery failures and one finish timeout. Improved detection has not solved
regrasp control. The visual runner does not invoke this oracle regrasp path.
An observable pose/grasp planner, recovery demonstrations, larger injection
studies and real force-sensor calibration remain unfinished.

## Source identity and timing scope

`evaluation_source.zip` and its checksum sidecar retain the exact code used by
the reserved IID/OOD and BC-state experiments. Force-monitor/recovery refinements
and resume-validation hardening were added afterward; force detection was disabled
in those large policy evaluations. To reproduce that plan byte-for-byte, restore
the source snapshot into a separate checkout, copy the data/weights bundle, and
run from that checkout. Current source requires a new plan and fresh seed range.
No BC/P5 weights, transition tolerances or policy actions were tuned on the new
IID/OOD outcomes. IID and OOD jobs used independent processes concurrently;
timing in those reports includes resource contention and should not be compared
with an isolated latency benchmark. Episode-level success/CI is the primary
outcome of this suite.

## Completed fresh-seed results

100 IID + 100 OOD visual trials and 100 state-BC trials are complete.
[Compact results, hashes and per-seed outcomes](./RELIABILITY_RESULTS_20261014.json).

| Mode/profile | Success | Wilson 95% interval |
|---|---|---|
| Visual iid | 75/100 | 65.70–82.45% |
| Visual camera | 7/20 | 18.12–56.71% |
| Visual lighting | 13/20 | 43.29–81.88% |
| Visual friction | 10/20 | 29.93–70.07% |
| Visual position | 3/20 | 5.24–36.04% |
| Visual combined | 4/20 | 8.07–41.60% |
| State BC oracle | 85/100 | 76.72–90.69% |

Both visual IID gates (observed ≥90%, lower 95% bound ≥80%) failed.
State BC observed success is 85%, but its lower bound is 76.72%, so the
population ≥80% claim remains unproven. OOD failures confirm that the fixed-scene
student/policy needs additional data/training; engineering guards alone did not
solve generalization. Preserve these seeds as completed evaluation history.
The next iteration should collect corrective visual-state examples and randomized
RGB-D teacher data, train/select using validation only, and reserve new evaluation
seeds before inspecting outcomes.
