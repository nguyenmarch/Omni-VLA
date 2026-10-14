# P5 temporal fallback and robot occlusion handling

This experiment compares the frozen P5/BC baseline with a candidate that uses
3D workspace/depth rejection, temporally smoothed landmarks, a short age-bounded
fallback, and a projected arm mask. P5 is a semantic RGB-D point-cloud student:
it does not decode one heatmap peak or read depth at one pixel. Its valid surface
cloud already aggregates many pixels, so the change uses robust cloud support
and a 3 m depth ceiling instead of introducing a misleading single-pixel patch
decoder.

The fallback keeps the last valid pair of landmarks for at most 15 invalid
inference calls and at most 0.4 simulated seconds. Fallback state resets at each
episode. Reports distinguish invalid-support causes, fallback calls, and
episodes using fallback. The arm mask projects Panda link forward-kinematic body
positions into the head camera and excludes a conservative image tube before
semantic cloud extraction. It requires the known robot model and camera
calibration; this simulation benchmark does not establish deployment on a
different robot or an uncalibrated real camera.

## Paired experiment

The immutable manifest is
[`artifacts/p5_temporal_mask_suite/plan.json`](../artifacts/p5_temporal_mask_suite/plan.json).
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
  --output artifacts/p5_temporal_mask_suite
```

Prepare a new, immutable seed set with:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --output artifacts/p5_temporal_mask_suite --seed 40000 \
  --iid-episodes 100 --ood-episodes 20 \
  --profiles iid camera lighting position
```

## Results

Full paired evaluation is in progress. The output JSON files are written when
each profile finishes; `summary.json` is written after all selected profiles
complete. No success-rate claim is made before then.
