# Trạng thái implementation — 13/10/2026

Đã chuyển scaffold sang pipeline simulation/data/training/export/evaluation chạy
được local. **Chưa hoàn tất toàn bộ mục tiêu VLA contact-rich**: BC Kaggle đã train và kiểm tra trên Dell, BC drawer đạt target 80% theo tỷ lệ quan sát trên hai bộ seed; CFM gate chưa đạt, P5 grounding đã đạt gate 3D trong scene cố định nhưng system closed-loop ≥90% chưa được chứng minh,
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
| CFM | Vector field, time embedding, optional noise/condition skips, Euler sampler | CFM Kaggle v2: Euler 4 50%/50%, 6 52%/46%, 8 60%/52% same/fresh |
| Export | BC action output / CFM vector field ONNX, JSON normalizer metadata | Deterministic forward parity max abs error khoảng 1e-7–1.6e-6 |
| Kaggle | Notebook BC → CFM → ONNX, source ZIP, resumable artifacts | BC v2 outputs đã import; CPU evaluation 92%/84% trên hai bộ seeds |
| Grounding P5 | Replay-certified semantic teacher, RGB-D student, known-vocabulary text encoder, calibrated cloud + scene-cylinder refinement; nối frozen BC v2 | Test visible 423/423 coverage, mean 0,29 cm, p95 0,83 cm; CPU capture+infer p50 66,69 ms; loop 15/20 trên seed mới, chưa đạt 90%. [P5 report](P5_PERCEPTION.md). Patch v3 trước đó 144/428 accepted, mean 10,23 cm |
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
- `data/grounding_rgbd_v2_256.h5`, `data/grounding_p5_labels.h5` và
  `weights/p5_student/best.pt`: source 12 episodes, replay-certified teacher
  masks và P5 checkpoint; [offline/CPU/loop reports](P5_PERCEPTION.md).
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

Các số grounding cũ dưới đây mô tả CNN pilot 128 trước P5. Grounding error
5.96 pixel là trên **133 visible validation targets** của một
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

- **44 tests pass**, gồm Gym reset/timing, long-rollout stability, tracking,
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

## CFM v2 handoff

Đã thêm config phase-balanced 150 epochs và notebook riêng CFM/dataset v2.
Candidate CLI giữ BC frozen, kiểm tra import/parity và benchmark hai sets với
4/6/8 Euler (300 episodes), có resume completed reports. Noise được cố định
theo seed/action step giữa settings/backends. CPU 3-epoch smoke kiểm tra pipeline;
GPU outputs đã import; full benchmark 300 episodes hoàn tất. [Kết quả](CFM_KAGGLE_V2.md).

## CFM v2: kết quả thực nghiệm

GPU train 150 epochs; checkpoint chọn epoch 133. Import hash/config pass;
ONNX/PyTorch 4/6/8 Euler parity đạt tolerance. CFM 4 Euler same/fresh 25/50 và
25/50; 6 Euler 26/50 và 23/50; 8 Euler 30/50 và 26/50. Không setting nào đạt
80% hoặc hơn BC v2; sampler p50 khoảng 1.44–2.62 ms, p95 2.52–4.25 ms trên
CPU 2 threads. Giữ BC v2 unchanged. [Report](../artifacts/cfm_kaggle_v2/comparison.md).

Đây là một seed training và hai cấu hình model/chunk khác nhau. Không suy
rộng kết quả thành kết luận CFM nói chung kém BC. Nếu tuning tiếp: CFM H=1
ablation và diagnostics first-action/phase; chưa thực hiện. VLA end-to-end vẫn
chưa tích hợp.

## CFM H1 ablation handoff

Đã chẩn đoán first-action của BC/H16 trên 2,080 validation states với fixed
noise, và trace 6 selected rollouts (success/steps khớp benchmark). Lỗi early
phases tăng mạnh ngoài expert trajectories; chưa chứng minh chunk là nguyên
nhân. Config/notebook H1 chỉ đổi chunk; smoke local kiểm tra pipeline. Candidate
CLI hỗ trợ expected chunk và reference comparison để giữ BC/H16 frozen.
GPU H1/full benchmark hoàn tất. [Chi tiết](CFM_H1_V2.md).

## CFM H1: kết quả

Checkpoint Kaggle đúng chunk1, 150 epochs, best epoch 144. Full benchmark
300 trials: Euler 4 same/fresh 35/50 và 39/50; 6 37/50 và 32/50;
8 30/50 và 22/50. Cải thiện H16 ở 4/6, không nhất quán ở 8; chưa đạt 80%
trên cả hai sets. Sampler p50 1.21–2.48 ms, p95 1.97–3.76 ms. BC/H16
weights unchanged. Full first-action validation comparison bổ sung: H1 có
error thấp hơn H16 nhưng còn cao hơn BC ở approach; place error trên validation
giảm theo Euler steps trong khi place failures closed-loop tăng. Không suy
rộng thành root cause chunk hoặc kết luận CFM nói chung. Giữ BC v2 cho
vision-language integration. [Report](../artifacts/cfm_h1_kaggle_v2/comparison.md).

## Patch grounding v3 — 11/10/2026

Đã migrate median-patch supervision, CPU pilot10epochs, all-target coverage
val/test, runtime invalid gates và Kaggle handoff. Test candidate epoch10:144/428
accepted, mean10.23cm/P95 25.21cm,3/428within1.5cm; loop0/5 từ chối perception
trước action. Chưa cải thiện định vị. Sync head256/wrist224 + MuJoCo MJCF FK
mask audit có; Pinocchio, masked per-camera training/fusion, async còn thiếu.
[Report và selection pitfall](PATCH_GROUNDING_V3.md).
