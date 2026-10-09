# Trạng thái implementation — 09/10/2026

Đã chuyển scaffold sang pipeline simulation/data/training/export/evaluation chạy
được local. **Chưa hoàn tất toàn bộ mục tiêu VLA contact-rich**: BC Kaggle đã train và kiểm tra trên Dell, nhưng BC drawer đạt target 80% theo tỷ lệ quan sát trên hai bộ seed; CFM gate chưa đạt, grounding chưa nối vào policy,
uncapping hai tay chưa tích hợp và recovery cần cải thiện.

## Những gì đã triển khai

| Phần | Implementation | Kết quả kiểm chứng |
|---|---|---|
| Setup | `pyproject.toml`, `.venv`, config resolver, module CLIs | Editable install, pip check và path portability |
| Simulation | Hai Panda từ upstream collision assets; bottle free-body, drawer slide; RGB-D và wrist/head cameras | MJCF compile, 10,000-step stability, seeded reset, Gymnasium checker |
| Controller | Cartesian torque impedance, SO(3) error, Jacobians, bias compensation, torque limits, posture damping, gripper servos | Free-space tracking mục tiêu <1 cm trong test |
| Expert | Approach/grasp/pull/lift/transfer/place/release/close FSM | 99/100 full drawer task, seeds 3000–3099 |
| Data | Streaming HDF5, clean vs executed actions, split episode, train-only stats, chunk tail masks | 20 expert episodes và 12 corrective episodes; validation scripts pass |
| BC | MLP, optional phase heads, baseline một action; DAgger hỗ trợ | BC Kaggle v2: 46/50 same seeds, 42/50 fresh seeds |
| CFM | Vector field, time embedding, optional noise/condition skips, Euler sampler | Train/export chạy; 0/5 full tasks ở checkpoint thử nghiệm |
| Export | BC action output / CFM vector field ONNX, JSON normalizer metadata | Deterministic forward parity max abs error khoảng 1e-7–1.6e-6 |
| Kaggle | Notebook BC → CFM → ONNX, source ZIP, resumable artifacts | BC v2 outputs đã import; CPU evaluation 92%/84% trên hai bộ seeds |
| Grounding | Registered CNN, hashed-word text embeddings, spatial cross-attention head, depth-based 3D API | 5-epoch CPU pilot: pixel error 10.17 → 5.96 trên validation visible targets |
| Screw fixture | Hinge/slide, pitch constraint 6 mm/vòng, disable constraint sau 2 vòng | Max thread error ~0.0204 mm, release pose jump 0 |
| Recovery | Slip debounce, hold/reground/regrasp, bounded retry, chunk preemption | Tilted regrasp experiment 1/3 full tasks sau forced slip |

## Artifacts có sẵn trên máy

- `data/drawer_v1.h5`: 20 successful expert episodes, 10,674 steps; 12/4/4 split.
- `data/dagger_01.h5`: corrective labels ở learner/expert mixed rollouts;
  11/12 trajectories thành công nhưng labels/failures giữ riêng.
- `data/drawer_dagger_v1.h5`: 32 episodes, 16,771 labeled steps; corrective data
  chỉ vào train, base validation/test không thay đổi.
- `data/drawer_rgbd_v1.h5`: 6 successful RGB-D pilot episodes; grounding train
  hiện chỉ dùng head camera. Wrist orientation đã được chỉnh sau pilot; collect
  lại RGB-D nếu dùng wrist stream của scene hiện tại để train.
- `weights/bc_single/best.pt`, `weights/bc_single/policy.onnx` và JSON: baseline
  local một action, CPU Torch/ONNX đều đã rollout.
- `weights/bc_kaggle/best.pt`, `policy.onnx`, `policy.json`: outputs Kaggle đã
  import; 150 epochs, best epoch 138, dataset hash khớp `drawer_dagger_v1.h5`.
- `weights/cfm_skip/best.pt`, `weights/cfm_skip/policy.onnx` và JSON: CFM H=16,
  còn thiếu task-quality convergence.
- `weights/grounding_cpu/best.pt`: finite-instruction grounding pilot.
- `artifacts/scene_preview.png`: visual scene; collision geometry cũng được render.
- `notebooks/kaggle_train.ipynb`, `artifacts/kaggle_source.zip`: cloud handoff.

Dữ liệu `.h5`, weights `.pt`/`.onnx` và ZIP bị gitignore, vẫn tồn tại local.
Metadata JSON và benchmark reports có thể version trong repository. Source ZIP
không chứa environments, credentials, datasets hay model weights.

## Kết quả và giới hạn benchmark

### Scripted expert

[100-trial report](../artifacts/expert_100.json): success **99/100**, Wilson 95%
interval khoảng **94.55–99.82%**. Bottle initial XY jitter ±1.5 cm; friction,
lighting và robot geometry cố định. Không suy rộng kết quả sang OOD hoặc ±4 cm.
Success hiện là drawer đã mở, bottle center nằm trong cavity khi gripper thả,
và drawer đóng tới <3.5 cm; benchmark cho phép bottle nằm nghiêng, không yêu cầu
upright stow. Success dùng ground truth của evaluator.

Simulation/controller chạy trên CPU theo 500/250 Hz simulation time; action
25 Hz. Report state-only có real-time factor khoảng 3.93 trong lần expert run
với các job khác cùng chạy. Không phải hard real-time benchmark, không tính ảnh
hay ROS2.

### Learned baseline

[BC single-action report](../artifacts/bc_single_20.json): **8/20 (40%)**, Wilson
95% khoảng **21.88–61.34%**, CPU sampler p50 ~0.68 ms / p95 ~1.09 ms. Đây là
privileged state baseline có oracle object geometry và phase task manager;
không phải VLA. Failures chủ yếu approach/retreat/placement và lệch khỏi phân
phối expert. DAgger và phase heads cải thiện so với checkpoint đầu nhưng chưa
đạt gate 80% của MVP.

[BC Kaggle report](../artifacts/bc_kaggle_eval.json): **14/50 (28%)**, seeds
5000–5049, Wilson 95% **17.47–41.67%**, CPU sampler p50 **0.454 ms** / p95
**0.857 ms**, 2 threads, execute horizon 1. Failures: approach 27, retreat 3,
descend 3, place 3. Đây vẫn là privileged state/oracle phase baseline,
không phải VLA. Khác seed set và số episodes với baseline local 8/20 nên chưa
kết luận model nào tốt hơn. Best validation MSE **0.001893** ở epoch 138/150;
ONNX parity trên Dell **3.58e-7**. Loss thấp chưa đảm bảo closed-loop success.
[Import metadata](../artifacts/bc_kaggle_import.json) lưu hashes và train config.

[ONNX pilot](../artifacts/bc_single_onnx_eval.json): **2/3** với sampler p50
~0.25 ms. Sample quá nhỏ để so sánh backend. ONNX/PyTorch numerical parity
không đảm bảo task rollout giống tuyệt đối trong contact dynamics; đã có
khác biệt outcome/step count trên cùng seed. Không quảng bá các số latency này
là end-to-end VLA inference.

[CFM skip report](../artifacts/cfm_skip_eval.json): **0/5** full tasks. Một số
trials mở được drawer nhưng không hoàn tất transfer; loss/sample convergence
và corrective data cần tiếp tục. Chưa có OT pairing và chưa chứng minh CFM hơn BC.

### BC Kaggle v2

[Comparison](../artifacts/bc_kaggle_v2_comparison.json): same seeds 5000–5049
**46/50 (92%)**, Wilson 95% **81.16–96.85%**; fresh seeds 7000–7049
**42/50 (84%)**, Wilson 95% **71.49–91.66%**. Same-set improvement **+64
percentage points** vs v1. Target 80% đạt theo tỷ lệ quan sát trên cả hai bộ;
CI của fresh set vẫn chứa giá trị dưới 80%, nên chưa khẳng định population
success >=80%. Sampler CPU 2 threads: p50 **0.465/0.476 ms**, p95
**0.839/0.899 ms**. Chỉ đo state policy sampling, không tính end-to-end vision.
Không bật recovery hoặc teacher action fallback; oracle state và phase context
vẫn có. Approach failures giảm từ 27 xuống 3 trên same set.

Checkpoint `weights/bc_kaggle_v2/best.pt` khớp dataset v2, phase sampling
balanced/capped đúng config. Log có hai lượt 150 epochs; current checkpoint
chọn epoch 136 của lượt cuối, không phải train 300 epochs liên tục. ONNX parity
trên Dell **4.77e-7**. [Import metadata](../artifacts/bc_kaggle_v2_import.json).

### Grounding / recovery / screw

Grounding error 5.96 pixel là trên **133 visible validation targets** của một
validation episode, hai finite text instructions, head images 128². Labels là
object centers; chưa đo contact-surface error 3D, OOD, paraphrase hay image/text
ablation. Không có pretrained DINOv2 hoặc LLM trong runtime hiện tại.

[Forced-slip recovery report](../artifacts/recovery_tilt.json): **1/3** full
rollouts thành công. Injection mở left gripper 18 actions và perturb velocity,
không đổi object position. Detection chủ yếu dùng oracle relative position,
wrench được lọc nhưng chưa làm tiêu chí slip chính. Recovery còn yếu với chai
nằm ngang; retry/hold và failure exit đã triển khai, chưa đạt robustness gate.

Screw fixture chỉ xác minh constraint của cơ cấu xấp xỉ. Cap sau release vẫn
còn hinge/slide DOFs; chưa free-body detach, chưa có hai tay fixturing/uncapping,
chưa có wrench-based thread-slip benchmark. Không dùng kết quả fixture thay
cho kết quả full task.

## Verification

- **14 tests pass**, gồm Gym reset/timing, long-rollout stability, tracking,
  camera projection/unprojection và wrist direction, expert physical rollout,
  dataset split/statistics/tail masks/cache parity, CFM determinism, spatial
  grounding shape/depth requirement, recovery debounce/retry và config timing.
- PyTorch/ONNX deterministic forward parity đã check với cùng inputs.
- Resume smoke: dừng sau epoch 1 trong kế hoạch 3 epochs rồi resume;
  mọi model parameter giống bit-for-bit với run 3 epochs liên tục.
- Dataset validator pass cho expert, merged corrective và RGB-D pilots.
- Dependencies: [environment versions](../artifacts/environment.json); pip check
  không báo dependency conflict. GUI viewer chưa được mở trong agent session;
  offscreen EGL được kiểm tra và scene preview đã inspect.

## Corrective round 2

Đã chẩn đoán 27 approach failures qua trace, thu 25 mixed DAgger episodes
(seeds 6000–6024, 23/25 mixed success), replay toàn bộ dataset cũ với state
error 0 và merge thành `data/drawer_dagger_v2.h5`: 57 episodes, 30,902 steps,
49/4/4. Original held-out arrays giữ nguyên. CPU smoke, phase-balanced config,
notebook và candidate evaluation CLI đã chuẩn bị. GPU outputs v2 đã import và đánh giá candidate: **46/50 same seeds**,
**42/50 fresh seeds**; [comparison](../artifacts/bc_kaggle_v2_comparison.json). [Chi tiết và lệnh](BC_CORRECTIVE_ROUND_2.md).

## Công việc còn lại theo thứ tự

1. BC v2 đạt observed target 80% trên same/fresh sets (92%/84%). Freeze
   checkpoint v2 làm baseline; mở rộng nhiều training seeds và randomization
   để đo độ ổn định, giữ dữ liệu benchmark tách khỏi train.
2. Mở rộng corrective states/domain randomization và context/history; đạt BC
   held-out gate trước khi tối ưu CFM. Đánh giá đủ episodes và training seeds.
3. Đạt CFM closed-loop gate, so với BC cùng dữ liệu; OT chỉ thêm sau khi giữ
   đúng pairing của condition/targets. So sánh H=1/H=16 phải ghi rõ khác horizon.
4. Collect RGB-D dưới camera revision hiện tại, train/ablate grounding, bỏ
   object oracle khỏi policy và phase manager khi đánh giá VLA thật.
5. Tích hợp screw-cap fixture vào bimanual scene với free-body detach, expert
   uncapping và dataset tương ứng; kiểm chứng sensor/contact fidelity.
6. Cải thiện fallen-bottle recovery bằng pose-aware grasping, thu recovery
   demonstrations và kiểm thử nhiều perturbation cases.
7. Tích hợp async inference và profiling deadlines; ROS2/C++ sau khi các gate
   learned control/contact-rich đạt yêu cầu. ROS2 workspace hiện còn trống.

README chứa các lệnh runnable. Architecture/build plan vẫn mô tả roadmap và
các gate, không phải mọi milestone đã đạt.
