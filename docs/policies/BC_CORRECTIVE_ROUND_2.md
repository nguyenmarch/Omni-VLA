# BC corrective round 2

## Kết quả chẩn đoán

Checkpoint `weights/bc_kaggle/best.pt` được chạy lại trên seeds 5000–5049 với
trace trước mỗi action. Success/steps của cả 50 episode giống report ban đầu:
14/50 thành công, 27 timeout approach. Expert chỉ được dùng làm nhãn so sánh,
không thực thi hành động thay learner trong benchmark này.

Ở 27 failures, khoảng cách cuối trung vị tới mục tiêu: left 21.10 cm,
right 18.09 cm; lỗi góc trung vị 0.583/0.125 rad. Threshold chuyển phase là
1.2 cm cho mỗi tay, chỉ dựa vào vị trí. Trong 50 bước cuối, gripper phải có
closing fraction trung vị 100%, action gripper trung vị khoảng -0.875 trong
khi expert yêu cầu +1. Left distance có std trung vị 3.72 cm; right 0.025 cm.
Vì vậy các failures không chỉ thiếu vài mm ở threshold: có drift/dao động
ở tay trái, đứng lệch mục tiêu ở tay phải và dự đoán grip sai. Đây là quan sát,
chưa chứng minh nguyên nhân cụ thể của sai lệch. Không đổi timeout/predicate
để làm đẹp benchmark.

- Trace: `artifacts/bc_kaggle_trace.jsonl` (lớn, gitignore).
- Summary: `artifacts/bc_kaggle_diagnosis.json`.
- Evaluation: `artifacts/bc_kaggle_diagnostic_eval.json`.

## Pilot corrective

`data/dagger_kaggle_02.h5`: seeds 6000–6024, 25 episodes, 14,131 steps.
Learner thực thi 24/32 bước theo chu kỳ toàn episode trong approach/descend;
expert thực thi 8 bước còn lại và tất cả các phase sau. Tại mọi trạng thái,
`action` là expert label; `executed_action` và `learner_executed` cho biết
hành động thật. Tổng learner steps: approach 975, descend 825. Failed episodes
vẫn giữ valid corrective labels. Mixed success 23/25 không phải pure BC success.
Không collect trên seeds 5000–5049 để tránh thêm benchmark states vào train.

Replay toàn bộ 32 episode của dataset cũ với executed actions trong scene hiện
tại khớp chính xác mọi state/condition (max error 0). Audit cho phép merge
state-only qua scene hash revision; không xác minh ảnh/camera compatibility
hoặc mọi trajectory chưa từng ghi. Không sửa metadata dataset gốc.

Merged dataset `data/drawer_dagger_v2.h5`: 57 episodes, 30,902 steps;
49 train / 4 val / 4 test. Tất cả arrays của original val/test không đổi.
Merge checks seeds held-out, schema/timing/action conventions và namespace
collisions; scene mismatch cần audit bound với dataset hash.

## Train Kaggle

Upload ba artifacts:

1. `artifacts/kaggle_source.zip`: source mới, gồm các CLIs và config vòng 2.
2. `data/drawer_dagger_v2.h5`: attach làm Input.
3. `notebooks/kaggle_train_dagger_v2.ipynb`: import thành notebook.

Chỉ attach một source version; bỏ source cũ để autodetection không ambiguous.
Bật GPU + Internet, chạy cells theo thứ tự. Validator phải ra 57/49/4/4.
Config `configs/train_bc_dagger_v2.yaml` giữ BC H=1, hidden128, phase heads,
67D state và 150 epochs. Train lại từ đầu vì dữ liệu và normalization thay đổi;
không resume checkpoint v1. Sampling cân bằng phase, capped bởi min_count=100:
phase finish chỉ có 2 labels nên không được upweight như một phase đầy đủ.
Val vẫn dùng phân phối gốc và train-only stats.

14 tests pass; CPU smoke 3 epochs, ONNX parity 2.38e-7; phase-balanced resume
khôi phục model bit-for-bit như run liên tục. Manifest: `artifacts/bc_corrective_round_2.json`.
CPU smoke xác minh pipeline; không thay thế lượt GPU và không cung
cấp bằng chứng đạt success gate. Train và closed-loop metrics phải tách biệt.

## Khi lấy output mới về Dell

Đặt ZIP mới ở root với tên `bc_dagger_v2_outputs.zip`, chạy:

```bash
.venv/bin/python -m scripts.evaluate_bc_candidate \
  --archive bc_dagger_v2_outputs.zip
```

CLI kiểm tra dataset hash, import vào path mới `weights/bc_kaggle_v2/`, chạy
50 seeds so sánh 5000–5049 và 50 fresh seeds 7000–7049. Candidate chỉ thực thi
learned actions; oracle state/phase context vẫn có. Reports:
`artifacts/bc_kaggle_v2_same.json`, `bc_kaggle_v2_fresh.json`,
`bc_kaggle_v2_comparison.json`. Target 80% trên cả hai bộ được ghi rõ.
Bộ comparison đã được dùng để chẩn đoán, nên fresh seeds quan trọng; không
coi đây là bằng chứng OOD hay VLA. Candidate GPU v2 đã import và bước này đã chạy xong; xem kết quả bên dưới.

## Tái tạo dữ liệu

```bash
.venv/bin/python -m scripts.collect_demonstrations \
  --num-episodes 25 --seed 6000 \
  --dagger-checkpoint weights/bc_kaggle/best.pt \
  --dagger-phases approach descend --dagger-cycle 32 --dagger-learner-steps 24 \
  --output data/dagger_kaggle_02.h5
.venv/bin/python -m scripts.audit_state_replay \
  --dataset data/drawer_dagger_v1.h5 --output artifacts/drawer_v1_state_replay.json
.venv/bin/python -m scripts.merge_datasets \
  --base data/drawer_dagger_v1.h5 --correction data/dagger_kaggle_02.h5 \
  --state-scene-audit artifacts/drawer_v1_state_replay.json \
  --output data/drawer_dagger_v2.h5
.venv/bin/python -m scripts.validate_dataset data/drawer_dagger_v2.h5
```

Collector/merge yêu cầu output mới; các files trên đã có local.

## Kết quả GPU v2 và CPU evaluation

Checkpoint ở `weights/bc_kaggle_v2/best.pt`, epoch 136 của lượt cuối. Metrics
log có hai segments 1–150; không diễn giải 300 rows thành một run 300 epochs.
Dataset hash khớp v2, phase-balanced config đúng; ONNX parity trên Dell 4.77e-7.

| Bộ | Success | Wilson 95% | Sampler p50/p95 |
|---|---|---|---|
| Same seeds 5000–5049 | 46/50 (92%) | 81.16–96.85% | 0.465/0.839 ms |
| Fresh seeds 7000–7049 | 42/50 (84%) | 71.49–91.66% | 0.476/0.899 ms |

Observed target 80% đạt trên cả hai sets; fresh CI vẫn chứa success dưới 80%.
Baseline v1 cùng seeds đạt 14/50: tăng 64 percentage points; approach failures
giảm 27→3. State/oracle context vẫn giữ, không recovery/teacher fallback.
Results chỉ áp dụng reset randomization hiện tại; CFM/VLA gates vẫn chưa đạt.
Reports và hashes nằm ở `artifacts/bc_kaggle_v2_comparison.json` và
`artifacts/bc_kaggle_v2_import.json`. Giữ v2 làm baseline khi thử CFM.
