# P5 temporal fallback and robot occlusion handling

This experiment compares the frozen P5/BC baseline with a candidate that uses
3D workspace/depth rejection and a short age-bounded temporal fallback. Temporal
smoothing and the projected arm mask are implemented as opt-in ablations, but
are disabled in the measured candidate because development paired runs showed
that both could delay or remove useful landmarks. P5 is a semantic RGB-D point-cloud student:
it does not decode one heatmap peak or read depth at one pixel. Its valid surface
cloud already aggregates many pixels, so the change uses robust cloud support
and a 3 m depth ceiling instead of introducing a misleading single-pixel patch
decoder.

The fallback keeps the last valid pair of landmarks for at most 15 invalid
inference calls and at most 0.4 simulated seconds. Fallback state resets at each
episode. Reports distinguish invalid-support causes, fallback calls, and
episodes using fallback. The optional arm mask projects Panda link forward-kinematic body
positions into the head camera and excludes a conservative image tube before
semantic cloud extraction. It requires the known robot model and camera
calibration; this simulation benchmark does not establish deployment on a
different robot or an uncalibrated real camera.

## Paired experiment

The immutable manifest is
[`artifacts/p5_temporal_fallback_suite/plan.json`](../artifacts/p5_temporal_fallback_suite/plan.json).
It reserves 100 IID seeds and 20 seeds each for camera, lighting, and position
profiles. Each seed runs once through baseline and once through candidate, with
the same BC checkpoint, P5 checkpoint, scene, and policy settings. Results
report Wilson 95% intervals, fallback use, rejection causes, episode rescues and
regressions, and an exact paired McNemar p-value. The historical 75/100 IID and
7/20 camera, 13/20 lighting, 3/20 position results remain the original
benchmark; this paired run also measures a fresh baseline on the new seeds.

Run the prepared suite with:

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_temporal_fallback_suite
```

Prepare a new, immutable seed set with:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --output artifacts/p5_temporal_fallback_suite --seed 50000 \
  --iid-episodes 100 --ood-episodes 20 \
  --profiles iid camera lighting position
```

## Results

Full paired run completed on 2026-10-14. Baseline and candidate were evaluated
on the same frozen seeds: IID 50000–50099, camera 51000–51019, lighting
52000–52019, and position 53000–53019. The candidate used fallback only, with
the 0.4 s maximum age. All Wilson intervals below are 95% intervals.

| Profile | Historical result | Fresh baseline | Candidate | Candidate fallback calls / episodes | Paired rescues / regressions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| IID | 75/100 (75%) | 71/100 (71%; 61.5–79.0%) | 71/100 (71%; 61.5–79.0%) | 1 / 1 | 0 / 0 |
| Camera OOD | 7/20 (35%) | 10/20 (50%; 29.9–70.1%) | 10/20 (50%; 29.9–70.1%) | 0 / 0 | 0 / 0 |
| Lighting OOD | 13/20 (65%) | 10/20 (50%; 29.9–70.1%) | 10/20 (50%; 29.9–70.1%) | 1 / 1 | 0 / 0 |
| Position OOD | 3/20 (15%) | 5/20 (25%; 11.2–46.9%) | 5/20 (25%; 11.2–46.9%) | 18 / 5 | 0 / 0 |

The ≥80% IID target was **not met**: the observed candidate rate is 71%, and
its Wilson lower bound is 61.5%. Paired successes did not change in any profile.
The fallback reduced position-profile hard perception rejections from 19 to 18,
but did not rescue an episode; the remaining failures were dominated by
approach/descend/transfer timeouts. In IID, hard rejections remained 3, matching
baseline. The fallback activates rarely in IID and cannot fix phase control
timeouts or compensate for stale geometry on larger position shifts. Treat this
as a reliability guard against short invalid-frame bursts, not evidence of
increased task success or demonstrated OOD generalization.

The JSON reports contain per-seed traces, phase transitions, observation ages,
failure reasons, and fallback counts:
[`summary.json`](../artifacts/p5_temporal_fallback_suite/summary.json), with
profile reports under `baseline/` and `candidate/`. To rerun the exact paired
benchmark, use the `run` command above; it resumes from the immutable plan and
retains the same seeds/checkpoints.
