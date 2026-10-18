# P5 H1: paired harness và telemetry

Ngày: **2026-10-15**. Trạng thái: **done**.
Phạm vi: H1 trong [P5_CONTROL_HANDOFF.md](P5_CONTROL_HANDOFF.md).
H2 terminal servo và H3 calibrated contact chưa triển khai trong lượt này.
Panel dưới đây xác nhận harness, không phải nghiệm thu IID ≥80% hoặc OOD.

**Cấu hình và protocol mới**

`scripts.evaluate_visual_suite` tạo `omni-visual-eval-plan-v3`. Tên variant
không còn quyết định legacy perception. Mỗi variant lưu options đã resolve đầy
đủ trong hai khối `perception_options` và `controller_options`; task instruction,
checkpoint, scene/physics và seed set được giữ chung trong một plan.

[p5_controller_h1.json](../configs/p5_controller_h1.json) có:

| Variant | Perception | Transfer | Contact |
| :--- | :--- | :--- | :--- |
| B | P5 fallback bật, mask/EMA tắt | 12 mm | off |
| T_tolerance_debug | Như B | 25 mm | off |
| Raw_contact_debug | Như B | 25 mm | raw_debug |

B/T khác **đúng một option**: `controller_options.transfer_tolerance`.
Hai variant debug chỉ tái hiện các ablation đã biết; T ở đây chưa phải terminal
servo H2 và Raw không phải calibrated detector H3. `calibrated` tiếp tục fail-fast.

Plan lưu source/config/checkpoint/dataset hashes, snapshot và hash reference
report, seed-file hash, instruction và baseline name. `plan.seal.json` khóa
checksum của plan sau prepare. Không sửa trực tiếp plan để thay candidate.

- `development`: dùng cho chẩn đoán và chọn tham số; seed đã dùng cần reference
  report tường minh để replay.
- `regression_replay`: mỗi profile bắt buộc có `PROFILE=REPORT` làm tham chiếu,
  đủ các seed được chọn, cùng checkpoint và cấu hình baseline tương thích.
- `holdout`: không nhận replay references; seed không được trùng training,
  development hay các plan/evaluation đã đăng ký trong artifacts/configs.
- Training seed overlap luôn bị từ chối, kể cả khi có reference. Dataset
  training bổ sung có thể đăng ký bằng `--additional-training-dataset`.

Baseline replay phải khớp `success`, `steps`, `phase`, `failure_reason` theo
seed với reference; drift dừng run trước paired analysis. Report thiếu episode,
sai seed/order, khác effective options, source hoặc checkpoint bị từ chối.

Sau mỗi profile hoàn tất, receipt khóa report/trace hashes và plan hash. Resume
xác minh receipts, số tick, thứ tự trace và physical success trước khi tổng hợp.
Các profile đã hoàn tất không chạy lại policy/physics. Trace/report dang dở
không bị ghi đè; dùng output mới để chạy lại. Plan v1/v2 và report cũ giữ làm
historical artifacts; CLI v3 dùng config mới thay cho các flags candidate v2.

**Telemetry đã có**

[eval_visual_policy.py](../scripts/eval_visual_policy.py) ghi trace
`omni-visual-trace-v2` cho từng tick thực thi, kể cả lúc hold/đợi perception;
tick dừng trước khi thực thi được đánh dấu `executed=false`.

- Phase/substate, phase age, transition reason và phase sau update. Khi chưa có
  substate riêng, ghi phase legacy hoặc `await_perception`.
- Action trước/sau controller override, hold reason và trạng thái usable của
  perception. Không có policy prediction thì action trước override là null.
- EE position/orientation world, velocity sai phân tại action rate, goal error
  tổng/XY/Z, gripper aperture và một số sensor values sau bước physics.
- Wrench site/world cho cả hai cổ tay. Wrench residual **null** và calibration
  status là `disabled` hoặc `unavailable_H3`; H1 không giả định đã bù lực.
- Tuổi observation theo sim/wall clock; source sim age cộng cả cache age nếu
  P5 dùng fallback. Source wall age của cache chưa biết được ghi null.
- Sim/wall timestamps, work time trước trace write và deadline flag tương ứng.
  Deadline aggregate đo vòng điều khiển kể cả trace write, trước pacing sleep.

`--scoring-diagnostics` bật khối `scoring_only`; suite v3 luôn bật. Contact
pairs và tổng normal force theo geom pair lấy từ contact constraints đang hoạt
động. Geometry chai/hộc, đặt vật và physical success chỉ được thu sau khi đã
chọn action hoặc sau khi manager ra quyết định chuyển pha.

[visual_scoring.py](../utils/visual_scoring.py) không được policy/manager dùng
làm input. Nó ghi riêng force claim, lệnh mở kẹp, aperture thực tế và vị trí
chai tại những thời điểm đó. Quy ước:

- `false_contact_triggers`: force claim không có contact chai–đáy hộc thực tại
  sample đó; chạm thành không được tính là support đáy.
- `release_commands_outside_env_envelope` và `actual_openings_outside_env_envelope`:
  dùng envelope đặt chai trong env để audit. Một release hợp lệ ở độ cao thả
  nhỏ không bị gắn lỗi chỉ vì chưa chạm đáy.
- Latency từ onset contact đáy trong pha place tới hold/release command/actual
  opening có cả sim và wall time. Thiếu onset hoặc response phù hợp là **null**,
  không ghi giả bằng 0. Contact chỉ được sample tại action ticks, không phải
  mọi physics substep; chưa có bảo đảm hard real time.

Summary có phase duration p50/p95, reach count/rate, deadline misses, contact/
release audit và latency sample counts. Paired results giữ danh sách rescued/
regressed seed, nguyên nhân baseline cố định, audit candidate và exact McNemar.
Chuyển timeout từ transfer sang retreat vẫn là thất bại, không được tính rescue.

**Panel MuJoCo xác nhận H1**

Seed replay **76000, 76003**, reference là report H0. Cả B/T/Raw dùng cùng
checkpoint và perception. Baseline B khớp reference cả kết quả lẫn số bước.

| Variant | Success | Rescue vs B | Regression vs B | False contact | Lệnh release ngoài envelope |
| :--- | :--- | ---: | ---: | ---: | ---: |
| B | 1/2 | — | — | 0 | 0 |
| T_tolerance_debug | 1/2 | 1: 76003 | 1: 76000 | 0 | 0 |
| Raw_contact_debug | 0/2 | 0 | 1: 76000 | 2 | 2 |

Raw có 1 actual gripper opening ngoài envelope; force claims có 2 false
positive. Đây là các audit riêng, không quy mọi regression cho cùng một nguyên
nhân. Kết quả tái hiện vấn đề cũ và xác nhận paired reporting; sample này không
được dùng suy ra success rate IID hay khả năng generalization.

Deadline misses gồm telemetry: B **194/965** ticks, T **282/1406**, Raw
**241/1199**. Perception còn đồng bộ và đây là profiling CPU, không phải kết
luận hệ thống đáp ứng deadline. H1 bổ sung đo đạc, chưa giải quyết scheduling.

**Kiểm chứng và artifacts**

```bash
.venv/bin/python -m pytest -q \
  tests/test_p5_h1.py tests/test_p5_h0.py \
  tests/test_visual_bridge.py tests/test_visual_runtime.py
```

Kết quả: **51 passed**. Tests bao gồm variant isolation, seed leakage/replay,
training seed không được whitelist, baseline drift, thay source/config/plan/
checkpoint/dataset/seed-file, sửa report/trace, mất receipt, trace dang dở,
episode thiếu/trùng/sai thứ tự, Position-only plan, paired final success,
frame của wrench, fallback source age và real physics contact force.

Chạy resume trên artifact thật với `evaluate` được thay bằng hàm ném lỗi nếu
bị gọi vẫn thành công: không rerun policy/physics, summary checksum giữ nguyên.
Plan bị đổi tolerance sau prepare bị từ chối. Config baseline H0 giữ nguyên hash.

Artifacts tại `artifacts/p5_h1_20261015/`:

- [plan.json](../artifacts/p5_h1_20261015/plan.json) và `plan.seal.json`.
- [summary.json](../artifacts/p5_h1_20261015/summary.json): paired results và telemetry aggregate.
- [verification.json](../artifacts/p5_h1_20261015/verification.json): resume, config isolation và hash checks.
- `B/`, `T_tolerance_debug/`, `Raw_contact_debug/`: `iid.json`, `iid.jsonl`,
  `iid.receipt.json` theo từng variant.

Source hash trước prepare, sau physics và khi resume khớp. Báo cáo/trace cũ
không bị ghi đè. `compileall` và `git diff --check` đạt.

**Lệnh đã kiểm chứng**

Chạy từ repo root; dùng output mới vì artifacts đã có là immutable:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --variants-config configs/p5_controller_h1.json \
  --purpose regression_replay \
  --seed-file configs/p5_h1_replay_seeds.json \
  --reference-report iid=artifacts/p5_h0_20261015/baseline.json \
  --profiles iid --output artifacts/p5_h1_repeat

MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_h1_repeat

# Cùng lệnh run lần nữa chỉ kiểm tra/resume các profile đã hoàn tất.
MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_h1_repeat
```

Để chuẩn bị holdout mới, bỏ `--seed-file`/`--reference-report`, chọn
`--purpose holdout --seed <chưa_dùng> --iid-episodes 100 --ood-episodes 20`
và các profile mong muốn. Prepare sẽ chặn seed đã đăng ký. Đóng băng config
candidate H2/H3 trước khi làm việc này; không dùng debug variants hiện tại
để gọi là nghiệm thu controller mới. Sau prepare, đổi bất kỳ source/config
đầu vào nào cần plan mới.

Bước tiếp theo: **H2 terminal transfer ổn định**; H3 calibrated contact cũng
có thể bắt đầu trên harness này, nhưng hiện vẫn chưa có detector hiệu chuẩn.
