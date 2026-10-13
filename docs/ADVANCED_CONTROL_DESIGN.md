# Omni-VLA advanced embodied control — 10/10/2026

Update 11/10/2026: [PATCH_GROUNDING_V3.md](PATCH_GROUNDING_V3.md) records migrated
supervision, CPU retraining/coverage and fail-closed runtime. Accuracy is still
insufficient. Sync wrist224/MuJoCo FK mask audit exists; Pinocchio/fusion/async
remain pending. The implementation/audit paragraphs below describe the original
10/10 snapshot.

Target: >=90% success per drawer/bottle/cap task. This is a design and staged
migration, not an achieved result or a hardware deployment approval.

## 1. Root cause analysis

State BC v2 succeeds 92%/84%; 256 vision bridge normal is 0/5. Paired test mean
XYZ error 15.40cm, handle 18.69cm, approach 15.53cm. Perception errors and the
new task manager have not been isolated. Current CFM has independent pairing,
not OT; its H16 results were worse than BC. Fixed-gain controller and privileged
recovery do not satisfy the advanced design.

| Pillar | Present implementation | Remaining work |
|---|---|---|
| Head 256/wrist 224 | Head 256 collected/trained; wrist camera declared | Separate per-camera sizes, timestamps, async capture and fusion |
| Robust depth | New patch decoder + validation audit | Recalibrate offset labels; integrate versioned new runtime |
| FK self-body mask | None | Pinocchio URDF/CAD, camera projection, z-buffer and IoU audit |
| Correlated OT-CFM H16/A14 | Independent CFM baseline only | Correlated prior + actual OT + new visual/language condition |
| Learned completion/recovery | Rule manager/scripted recovery | Predicate labels/model, recovery dataset/policy and adaptive gains |
| ONNX/ROS2 | State-policy ONNX exports | New network/sampler exports; stamped bimanual ROS2 interfaces |

256 currently gives ~0.62–0.65cm/pixel at target depth. To meet <0.6cm, change
camera distance/FOV and recalibrate/recollect; resolution alone cannot certify
this. Wrist views reduce some occlusions but can themselves be occluded.
Masking robot pixels does not reconstruct a hidden target.

One thread owns mutable physics/controller MjData. A render worker owns its
own snapshot/MjData and OpenGL context; latest-only frames/chunks carry episode
ID, sequence, simulation/wall timestamp and age. Never concurrently mutate
the same MjData. [MuJoCo simulation/threading](https://mujoco.readthedocs.io/en/stable/programming/simulation.html),
[rendering](https://mujoco.readthedocs.io/en/stable/python.html#rendering).

## 2. Algorithms and mathematical contracts

### Robust depth and FK masking

\[
z_s=Q_q(\{D_i:i\in P_{valid}\}),\quad q\in\{0.5,0.2\},\qquad
p_w=T_{wc}[(z_s+\delta z)K^{-1}[u,v,1]^T;1].
\]

P is a 5×5/7×7 neighborhood, excludes invalid/out-of-image and self-body pixels.
Support/IQR gates reject empty or mixed surfaces. No single-pixel fallback.
Median can select background if it dominates; low quantiles can select a
closer wrong object. Workspace-invalid points are rejected. Clamped points
are diagnostic output only; clipping must not turn an invalid estimate into
a valid action target. Defaults use the controller's conservative workspace.
Depth-offset labels must use the same aggregation as inference.

Pinocchio computes link transforms T_wl(q); CAD triangles are projected through
K*T_cw*T_wl and rasterized with z-buffer. Compare robot depth with measured
depth before masking, to preserve foreground objects overlapping a link's 2D
silhouette. FK alone does not produce a pixel mask. Robot-only segmentation
is evaluation ground truth in sim, not object geometry passed to the policy.
[Pinocchio geometry](https://docs.ros.org/en/melodic/api/pinocchio/html/geometry-models_8cpp_source.html).

Head/wrist fusion uses validity, confidence, world-coordinate consistency and
timestamp; wrist activation at <=10cm also requires visibility. If both views
are invalid/stale, invalidate chunk and hold/reobserve. No fabricated landmarks.

### Correlated OT-CFM

\[
C_{ij}=\rho^{|i-j|},\quad0\le\rho<1,\quad LL^T=C+\epsilon I,
\qquad x_0=L\varepsilon,\ \varepsilon\sim N(0,I),\ x_0\in R^{16\times14}.
\]

Store covariance/rho/normalization/noise seed in model metadata; use identical
prior in train and inference. Correlation does not guarantee physical velocity,
acceleration or torque limits. Executor validates those separately.

Use actual minibatch OT within compatible task/context groups. Declare matching
cost; condition/instruction/tail mask follow reassigned target indices. Matching
across unrelated states must not be labeled a validated conditional controller.

\[
x_t=(1-t)x_0+tx_1,\quad
\mathcal L=\frac{\sum M\|v_\theta(x_t,t,c)-(x_1-x_0)\|^2}{\sum M},\quad
x_{k+1}=x_k+N^{-1}v_\theta(x_k,k/N,c),\ N\in\{2,4,6\}.
\]

[TorchCFM variants](https://github.com/atong01/conditional-flow-matching).
Replan after 1–2 actions; preempt on stale perception/contact/recovery. H16 at
25Hz covers 0.64s and must not be executed blindly. Consistency distillation
is a separate experiment if full sampler fails <=15ms CPU latency.

### Learned condition monitor and force control

Learn P(is_grasped, is_drawer_opened, cap_loosened | image features, q/dq,
gripper, filtered wrench, instruction, history). Measure false completion,
calibration and detection delay. Learned completion replaces semantic FSM;
a bounded safety supervisor still owns hold, deadline, retry and emergency stop.

\[
\tau=J^T(K(t)e-D(t)\dot x)+\tau_{bias}+\tau_{null},\qquad
D_i\approx2\zeta_i\sqrt{K_i m_{eff,i}}.
\]

Use compensated wrench in declared tool/world frames; K/D have bounded values
and rates. F_z/tau_z thresholds are fixture-specific measured parameters.
Torque stop needs debounce/contact checks; torque alone does not certify cap
release. Autonomous DAgger uses expert corrective labels on learner states in
simulation; learner predictions are not their own ground truth.

ROS2: named JointState, Image+CameraInfo+TF with timestamps; action messages
carry both arm poses/grippers, frame, duration, episode/sequence and validity.
`/cmd_vel_cartesian` must not silently mix delta poses and velocities. Define
an explicit stamped bimanual message or documented TwistStamped conversion
and gripper topics. Long tasks use Actions with cancel/feedback.

## 3. Concrete implementation and migration

New [robust_depth.py](../utils/robust_depth.py): vectorized patch gather,
masked quantile, support/IQR/workspace checks, shape/calibration guards, no
single-pixel fallback. [audit_robust_depth.py](../scripts/audit_robust_depth.py)
evaluates both variants on all validation frames including occlusion.

The component is experimental and **not wired into the legacy runtime yet**.
Legacy offsets were trained using single-pixel labels. Dataset supervision,
preprocessing and checkpoint schema must migrate together; historical runners
remain baseline reproduction tools, not the new production architecture.
Pinocchio mask, wrist fusion, async workers, OT/noise, learned monitor/recovery,
adaptive gains, distillation and ROS2 have not been implemented by this change.

Validation audit, 458 targets with old offsets:

| Decoder | Valid coverage | Valid-only mean XYZ | Within 1.5cm / all targets |
|---|---:|---:|---:|
| Median 7×7 | 171/458 (37.3%) | 8.65cm | 7/458 |
| Q20 7×7 | 213/458 (46.5%) | 6.34cm | 6/458 |

These conditional means cannot be compared directly to the visible-only
15.40cm benchmark. Rejected targets count as failures for overall coverage;
neither variant is deployment-ready. [Audit report](../artifacts/advanced_perception/robust_depth_val.json).
Tests: 33 passed, including center outlier, invalid depths, mixed surfaces,
all-robot mask, image border and rejection of workspace-invalid coordinates.

## 4. Verification gates

- Perception: proposed P95 <=1.5cm on declared visible targets; publish invalid/
  occluded coverage and per-camera/object/phase errors. Roundtrip alone does
  not certify physical calibration.
- Mask: proposed IoU >=0.9 on robot-only sim segmentation; report false masking
  of targets and sensitivity to joint/TF lag.
- Runtime: measured >=5Hz refresh, then test 10Hz; p50/p95/p99 frame age, drops
  and deadlines. Full policy sampler p95 <=15ms at 2/4/6 Euler on CPU under
  render load. Start at 25Hz actions; 50Hz needs new action_dt data/model.
- Export: max absolute error <1e-5 for same inputs/noise/chunk, including complete
  sampler and each new head. Quantile preprocessing may stay on CPU with its
  own tested parity boundary if ONNX cannot export that operator.
- Tasks: >=90% observed success per task on each of two held-out seed sets,
  at least 100 trials/set; publish Wilson CI, failure phase, collisions/forces,
  retries and perturbation recovery. CI lower bound >=90% is a stronger gate
  and is not implied by observing 90/100.
- Data: episode-disjoint splits, train-only statistics, image/action alignment,
  schema/hash validation, perturbation/recovery demos and no oracle policy input.

Targets above are acceptance proposals, not measured full-system achievements.
