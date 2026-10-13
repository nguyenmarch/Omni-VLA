# Omni-VLA

Dự án học điều khiển hai tay robot, hướng tới visual-language affordance,
flow matching và manipulation có tiếp xúc lực. Runtime local dùng **CPU Dell
Latitude 5421 + Python/MuJoCo**; training GPU chạy bằng notebook **Kaggle**.

Bản hiện tại có simulation **hai Franka Panda, chai và drawer có slide joint**,
Cartesian impedance torque control, scripted expert và pipeline HDF5 → BC/CFM
→ checkpoint/ONNX → CPU rollout. Expert kẹp/vận chuyển chai bằng tiếp xúc vật lý;
không teleport vật thể hoặc dùng weld giả để nắm chai.

Pilot mới: [Patch grounding v3](docs/PATCH_GROUNDING_V3.md) — đã migrate nhãn,
train CPU và đo coverage; test accepted error 10.23cm, closed-loop 0/5.
Kaggle notebook/input sẵn; wrist224 + MuJoCo FK mask đang ở mức audit.
Mốc code, config, hash dữ liệu/checkpoint, report và seed đã chốt tại
[reproducible baseline](docs/REPRO_BASELINE.md).

## Trạng thái đã kiểm chứng

- Expert drawer benchmark: **99/100** trials, seeds 3000–3099, random vị trí chai
  ±1.5 cm. Đây là kết quả **scripted expert**, không phải learned VLA success.
  [Report](artifacts/expert_100.json).
- Dataset `data/drawer_v1.h5`: 20 successful demonstrations, 10,674 action steps,
  split 12 train / 4 validation / 4 test theo episode.
- BC một action đạt **8/20** CPU rollouts; CFM còn **0/5** full tasks ở checkpoint thử nghiệm. BC và CFM đã train thử trên CPU và export ONNX; parity max absolute error
  khoảng `1.6e-6` ở các checkpoint đầu. Validation loss và rollout success được
  ghi riêng; checkpoint ban đầu chưa đạt learned-control success gate.
- RGB-D offscreen, calibration và camera round-trip đã kiểm tra; CNN grounding
  có language conditioning và script train riêng. Chưa xác nhận VLA end-to-end.
- Screw fixture kiểm chứng ren 6 mm/vòng và release constraint sau hai vòng;
  chưa tích hợp free-body uncapping vào benchmark hai tay.
- Recovery có slip debounce, buffer preemption và scripted regrasp. Recovery
  thử nghiệm còn thất bại ở một số tình huống; xem report trong `artifacts/`.
- Người dùng đã train BC trên Kaggle 150 epochs; outputs được import vào
  `weights/bc_kaggle/`. CPU closed-loop đạt **14/50 (28%)**, seeds 5000–5049;
  chưa đạt gate 80%. [Report](artifacts/bc_kaggle_eval.json).

Không dùng các số `89.6%`, `<18 ms` của bản đề xuất ban đầu làm benchmark hiện tại.

Vòng corrective 2 đã chuẩn bị: **25 DAgger episodes**, dataset v2 **57 episodes**
(49/4/4), notebook train lại và CLI so sánh 100 CPU trials sau khi nhận outputs.
Xem [hướng dẫn vòng 2](docs/BC_CORRECTIVE_ROUND_2.md). BC Kaggle v2 đã đánh giá: **46/50 (92%)** trên seed cũ và **42/50 (84%)**
trên seed mới, đạt target 80% theo tỷ lệ quan sát. [Comparison](artifacts/bc_kaggle_v2_comparison.json).

## Cài đặt trên Dell

Python 3.10+; môi trường kiểm chứng hiện tại dùng Python 3.12. ROS2 chưa cần thiết.
Robot collision assets và upstream license được lưu trong repository.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/python -m pip install -e '.[train,deploy,test]'
.venv/bin/python -m pytest -q
```

Mọi CLI dùng `python -m ...` để import portable; chạy từ repo root hoặc sau khi
editable-install. Không cần tải lại Panda assets. `scripts/fetch_robot.py` chỉ
phục vụ refresh upstream khi cần, có thay đổi assets/config thì benchmark lại.

## Chạy simulation và thu dữ liệu

```bash
# Viewer có hai tay robot; collector ghi dữ liệu và report
.venv/bin/python -m scripts.collect_demonstrations --num-episodes 3 --gui --output data/gui_demo.h5

# Dataset state baseline; dùng output path mới mỗi lần collect
.venv/bin/python -m scripts.collect_demonstrations --num-episodes 150 --output data/drawer_train.h5
.venv/bin/python -m scripts.validate_dataset data/drawer_train.h5

# RGB-D + intrinsics/extrinsics; EGL đã chạy được trên máy hiện tại
MUJOCO_GL=egl .venv/bin/python -m scripts.collect_demonstrations --num-episodes 6 --images --output data/rgbd_train.h5
```

Nếu EGL không hoạt động trên máy khác, thử backend phù hợp driver như OSMesa;
OSMesa cần thư viện hệ thống tương ứng. State-only mode không tạo renderer.

Timing: physics 500 Hz, controller 250 Hz, actions 25 Hz theo simulation time.
Mỗi action advance 20 physics steps; không khẳng định hard real-time OS.
Action `[14]` là hai bộ `[dx,dy,dz,rx,ry,rz,g]`: normalized xyz ×0.015 m,
world rotation vector ×0.08 rad, `g ∈ [-1,1]` map sang finger opening 0–0.04 m.

State condition `[67]` gồm arm q/dq, grippers, hai wrist wrench, end-effector
positions, oracle object geometry và phase context. Baseline này dùng thông tin
đặc quyền của simulation; evaluator ghi rõ `learned_state_oracle_context`.

## Train trên Kaggle

1. Chạy `scripts.package_kaggle` để tạo `artifacts/kaggle_source.zip`.
2. Upload source ZIP và HDF5 đã validate thành Kaggle datasets.
3. Mở [notebook](notebooks/kaggle_train.ipynb), attach hai datasets, chọn GPU khả
   dụng và chạy các cells. Notebook detect CUDA; không giả định loại GPU/quota.
4. Save Version với outputs; tải các thư mục `bc`, `cfm` về Dell.

```bash
.venv/bin/python -m scripts.package_kaggle

# Các lệnh training bên trong Kaggle / hoặc CPU smoke tại local
python -m scripts.train_kaggle --dataset /path/drawer_v1.h5 --policy bc --output /kaggle/working/bc
python -m scripts.train_kaggle --dataset /path/drawer_v1.h5 --policy cfm --output /kaggle/working/cfm
```

CFM hiện dùng independent noise/data pairing với loss vector field và Euler
4/6/8 steps. Chưa có OT batch matching, vì vậy không gắn nhãn OT-CFM.
BC baseline mặc định dùng một action và phase-specific heads; CFM dùng chunk 16. BC có optional phase-specific heads; phase context nằm trong baseline schema.

BC config: `configs/train_bc_kaggle.yaml`; CFM config: `configs/train_cfm_kaggle.yaml`. CLI tự chọn config theo `--policy`.

Checkpoint gồm model, normalization, optimizer, scheduler, scaler, Torch RNG,
dataset hash, episode splits và config. Resume phải dùng cùng dataset/model
config, first-action weight và **tổng số epochs**:

```bash
# Chạy 1 epoch trong kế hoạch 80 epochs rồi dừng có chủ đích
python -m scripts.train_kaggle --dataset /path/drawer_v1.h5 --policy bc --epochs 80 --max-epochs-per-run 1 --output weights/bc
python -m scripts.train_kaggle --dataset /path/drawer_v1.h5 --policy bc --epochs 80 --resume weights/bc/last.pt --output weights/bc
```

Dataset state/action nhỏ được cache trong RAM; ảnh không được nạp toàn bộ.
Nếu BC lệch khi tự rollout, thu corrective labels bằng DAgger thay vì chỉ tăng epochs:

```bash
python -m scripts.collect_demonstrations --num-episodes 12 --seed 500 --dagger-checkpoint weights/bc/best.pt --output data/correction.h5
python -m scripts.merge_datasets --base data/drawer_v1.h5 --correction data/correction.h5 --output data/merged.h5
python -m scripts.train_kaggle --dataset data/merged.h5 --policy bc --first-action-weight 8 --output weights/bc_corrected
```

DAgger thực thi learner 4/8 steps, còn lại expert; lưu `action` là expert label
và `executed_action` riêng. Corrective episodes thêm vào train, giữ nguyên held-out
split của dataset gốc. Kết quả rollout trộn teacher/learner không phải pure policy success.

## CFM Kaggle vòng v2

Import [notebook CFM](notebooks/kaggle_train_cfm_v2.ipynb), upload source ZIP mới
và `drawer_dagger_v2.h5`; config riêng `configs/train_cfm_dagger_v2.yaml`.
Sau khi lấy `cfm_dagger_v2_outputs.zip` về Dell:

```bash
.venv/bin/python -m scripts.evaluate_cfm_candidate --archive cfm_dagger_v2_outputs.zip
```

Chạy 300 CPU trials (hai bộ seed × 4/6/8 Euler × 50), so với frozen BC v2.
GPU training và đủ 300 CPU trials đã hoàn tất: CFM Euler 4 **50%/50%**,
Euler 6 **52%/46%**, Euler 8 **60%/52%** trên same/fresh seeds. CFM hiện tại
chưa đạt 80%; giữ BC v2 **92%/84%** làm baseline.
[Report](artifacts/cfm_kaggle_v2/comparison.md), [chi tiết](docs/CFM_KAGGLE_V2.md).

## CFM H1 ablation

Đã chẩn đoán first-action theo phase trên validation và rollout H16; notebook
`notebooks/kaggle_train_cfm_h1_v2.ipynb` thử chunk1 với các settings khác giữ nguyên.
Source ZIP mới + dataset v2, rồi import/eval vào paths riêng khi nhận output.
GPU H1 và 300 CPU trials đã hoàn tất: Euler 4 **70%/78%**,
6 **74%/64%**, 8 **60%/44%** same/fresh. Chưa đạt 80% trên cả hai sets;
giữ BC v2 làm policy chính.
[Report](artifacts/cfm_h1_kaggle_v2/comparison.md),
[hướng dẫn và chẩn đoán](docs/CFM_H1_V2.md).

## Xem BC v2 trên Dell

```bash
.venv/bin/python -m scripts.eval_closed_loop --checkpoint weights/bc_kaggle_v2/best.pt --seeds 5000 5006 5026 --execute-horizon 1 --gui --speed 1 --output artifacts/bc_v2_visual_review.json
```

Speed 1 là tốc độ thực; 0.5 xem chậm hơn. Đóng viewer dừng phiên.
[Xem hướng dẫn và các điểm cần quan sát](docs/BC_VISUAL_REVIEW.md).

## Export và closed-loop evaluation CPU

```bash
python -m scripts.export_policy --checkpoint weights/bc/best.pt --output weights/bc/policy.onnx
python -m scripts.eval_closed_loop --checkpoint weights/bc/best.pt --execute-horizon 1 --episodes 50 --output artifacts/bc_eval.json
python -m scripts.eval_closed_loop --checkpoint weights/bc/policy.onnx --execute-horizon 1 --episodes 50 --output artifacts/bc_onnx_eval.json
python -m scripts.eval_closed_loop --checkpoint weights/cfm/policy.onnx --euler-steps 6 --episodes 50 --output artifacts/cfm_onnx_eval.json
python -m scripts.eval_closed_loop --expert --episodes 100 --seed 3000 --output artifacts/expert_repeat.json
```

ONNX export dùng batch=1 cho CPU deployment. BC export action chunk; CFM export
vector field và evaluation chạy Euler wrapper. Giữ `.onnx` **cùng `.json` metadata**.
Execute horizon mặc định 2 actions; mỗi phase transition xóa chunk buffer.
Learned evaluation không gọi teacher action để bù lỗi; task manager có oracle
phase/context được khai báo trong report. Report gồm failures, stage success,
Wilson interval, sampler p50/p95 và real-time factor. Latency sampler chưa bao gồm
perception, render hoặc ROS2 communication.

## Grounding, screw fixture và recovery experiments

```bash
python -m scripts.check_screw_fixture
python -m scripts.eval_closed_loop --expert --recovery --inject-slip --episodes 10 --seed 9000 --output artifacts/recovery_eval.json
```

Grounding train dùng ảnh, finite text instructions và simulator center labels;
metric là pixel error trên các target visible. Module yêu cầu depth thật và
camera extrinsics khi unproject 3D. Pilot mới nối predicted landmarks vào BC v2,
thay oracle object/phase bằng camera và finite-task manager; xem
[Vision-language CPU pilot](docs/VISION_LANGUAGE_PILOT.md) để xem kết quả
3D, closed-loop và image/text ablations; [pilot 256×256](docs/GROUNDING_256_PILOT.md)
thu lại RGB-D và train grounding mới, giữ BC v2 làm baseline. Recovery hiện dùng oracle geometry
cho regrasp và phải benchmark thêm.
`--inject-slip` mở gripper tạm thời và perturb velocity để gây mất grasp; không
đổi object position. Screw fixture dùng hinge/slide approximation; release vẫn
còn articulated DOFs, chưa phải cap free-body rời chai.

## Tài liệu và cấu trúc

- [Architecture](docs/ARCHITECTURE.md), [Build plan](docs/BUILD_PLAN.md): mục tiêu và gates.
- [Implementation report](docs/IMPLEMENTATION_STATUS.md): kết quả local và phần còn lại.
- `simulation/`: scene composition, env, expert, recovery, Panda assets.
- `models/`: impedance controller, BC, CFM và visual grounding.
- `utils/`: dataset/schema conventions, camera geometry, checkpoints.
- `scripts/`: collect, validate, train, export, eval và Kaggle packaging.
- `tests/`: correctness về physics, geometry, dataset và sampling.
- `artifacts/`: reports và scene preview; `data/`, `weights/`: local artifacts.

Panda assets từ [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie),
license Apache-2.0 trong [asset directory](simulation/assets/panda/LICENSE).
ROS2/C++ integration là milestone sau khi learned closed-loop và contact-rich
extensions đạt các acceptance gates; workspace ROS2 chưa có runtime packages.
