# P5 H2: terminal transfer ổn định

Ngày: **2026-10-15**. Trạng thái: **H2 done — đạt gate development**.
Phạm vi: H2 trong [P5_CONTROL_HANDOFF.md](P5_CONTROL_HANDOFF.md).
Đây là vòng controller với BC/P5 cố định, không phải nghiệm thu IID 100 seed.

**Triển khai**

[simulation/transfer.py](../simulation/transfer.py) thêm `TerminalTransfer` với
chuỗi `bc_transfer → settle → ready_to_place`. Chỉ có hiệu lực ở phase 4 khi
chọn `--transfer-mode terminal`; mặc định tiếp tục là `legacy`. Phase one-hot
và input BC vẫn 67-D. Các impedance gains, scene, checkpoints, logic success
và release/retreat legacy không thay đổi. Contact mode của B/T đều `off`.

Khi cả hai EE vào lân cận goal, servo chụp loaded pose, orientation và action
BC lúc vào settle. Các thành phần đã đủ gần giữ loaded pose cùng feedforward
BC để duy trì tải; những thành phần sai số còn lại được Cartesian feedback
đưa về goal từ camera. Đặc biệt, handle center không được coi là một target
chính xác bắt tay phải bỏ loaded grasp pose để dịch thêm vài mm.

Target env là delta từ EE đo hiện tại. Servo xuất `linear_velocity * dt / .015`
và `angular_velocity * dt / .08`, không tích phân target thêm lần nữa. Gripper
hai tay tiếp tục nhận lệnh đóng; hold theo measured finger width có thể làm
mất preload lên chai/handle. Orientation được giữ quanh loaded orientation,
có BC angular feedforward và feedback correction.

| Tham số mặc định của terminal mode | Giá trị |
| :--- | :--- |
| Entry radius, max error giữa hai EE | 30 mm |
| Ready tolerance, max error giữa hai EE | 12 mm |
| Ready measured speed, cả hai tay | ≤0,025 m/s |
| Stable window | 3 tick tại 25 Hz |
| Cartesian command speed / acceleration | 0,15 m/s / 0,75 m/s² |
| Angular command speed / acceleration | 0,5 rad/s / 2 rad/s² |
| Orientation error tối đa khi ready | 0,12 rad |
| Settle budget / no-progress window | 150 / 40 tick |
| Source sim age tối đa, gồm fallback age | 0,4 s |
| Goal jump tối đa so với lúc vào settle | 30 mm |

Các bound command áp dụng trong servo. Limiter được khởi tạo từ measured EE
velocity; không coi bound này là bảo đảm gia tốc vật lý của toàn robot hoặc
slew limit xuyên qua policy BC trước khi vào servo. Emergency hold do stale
perception/goal jump ưu tiên dừng và có thể giảm tốc command nhanh hơn slew
limit bình thường. Config vượt action delta của env tại `dt` hiện hành bị từ chối.

**Điều kiện chuyển pha và failure**

- Source usable và đủ tươi, goal còn nhất quán, cả hai EE đủ gần và đủ chậm,
  orientation còn hợp lệ qua đủ stable ticks.
- Độ mở hộc **hiện tại** >16 cm; không chỉ dựa vào cờ từng mở trước đó.
- Clearance dùng perceived bottle/handle và EE: bottle footprint nằm trong
  envelope có trừ bán kính chai, đáy chai dự đoán trên rim + margin; EE trái
  vẫn đủ cao. Các kích thước chai/fixture là giả định hữu hạn của scene này.
- Stale source hoặc goal jump làm hold và reset stable streak. Manager vẫn
  theo dõi terminal budget ở tick hold, không ép place khi mắt bị stale.
- Budget hết: `visual_transfer_settle_timeout`; không tiến bộ:
  `visual_transfer_settle_no_progress`. Cả hai đều là thất bại, không phải
  transition thành công. Các state/cached pose/preload reset theo episode.

Không dùng contact ground truth, vị trí object trong simulator hoặc expert
action để quyết định servo/guard. Feedforward là action của chính BC. GT trong
trace tiếp tục ở khối `scoring_only` của H1.

**Những gì development đã loại bỏ**

Bản đầu servo toàn bộ pose về landmark center đạt guards ở transfer nhưng gây
2/2 timeout place trên panel 76000/76003. Vì vậy nó không được chọn dù đã hết
timeout transfer. Bản giữ loaded components và BC preload sau đó đạt 2/2 trên
panel này. Đó là quá trình chọn trên development, không phải holdout.

Artifacts các prototype là `artifacts/p5_h2_dev_v1/` và `p5_h2_dev_v2/`.
Replay đầy đủ trước khi bổ sung guard current-opening nằm ở
`artifacts/p5_h2_development_20261015/`. Nghiệm thu H2 chỉ dùng bản cuối bên dưới.

**Kết quả bản cuối**

Replay paired trên 76000–76004 đã hoàn tất, có completion receipts và source
hash khớp plan. Baseline tái lập đúng success mask, số bước và failure reason
của H0. Hai variant chỉ khác `transfer_mode` sau khi normalize config.

| Seed | B: legacy | T: terminal | Kết quả paired |
| :--- | :--- | :--- | :--- |
| 76000 | Success, 453 tick | Success, 641 tick | Giữ success |
| 76001 | Success, 470 tick | Success, 619 tick | Giữ success |
| 76002 | Timeout retreat, 747 tick | Timeout retreat, 922 tick | Chưa cứu được |
| 76003 | Timeout transfer, 512 tick | Success, 549 tick | **Rescue** |
| 76004 | Success, 440 tick | Success, 629 tick | Giữ success |
| Tổng | **3/5 (60%)** | **4/5 (80%)** | **1 rescue, 0 regression** |

Transfer timeout giảm **1 → 0**; cả 5 episode của T đi tới place. Không có force
trigger, release command hay actual opening ngoài placement envelope trong
audit của hai variant. Điều này chỉ mô tả panel này, không chứng minh release
an toàn trên mọi state. Seed 76002 vẫn timeout retreat; H2 không sửa release/
retreat và không tính việc qua transfer là task success.

Tại 5 transition của T, settle dùng **16–19 tick**, stable streak đều **3 tick**;
max error của hai EE tại mỗi transition nằm trong **6,78–9,66 mm**, max measured
speed **0,0130–0,0160 m/s**, max orientation error **0,0143–0,0152 rad**.
Current opening **0,1930–0,1967 m**; mọi source/goal/position/speed/orientation/
clearance guard đều pass. Kiểm tra toàn bộ command settle xác nhận giới hạn
linear/angular speed và slew, grip vẫn đóng.

Ở tick đầu place trên 4 seed cả B/T đều tới place (76000/76001/76002/76004),
median fraction của 57 feature nằm ngoài train envelope theo phase giảm từ
**30,70% → 21,05%**; median max normalized excess giảm **0,689 → 0,302**.
Envelope là khoảng quantile 1–99% riêng từng feature từ train split, không phải
ước lượng density hay bằng chứng state hoàn toàn nằm trong phân phối BC.
Candidate vẫn còn feature ngoài envelope. Trace lưu diagnostic này, không
đưa nó vào điều khiển.

**Đánh đổi:** trên 3 seed success cũ, median tổng bước tăng **453 → 629**
(18,12 → 25,16 s simulation tại 25 Hz), chủ yếu vì place dài hơn sau terminal
transfer. H2 đạt ổn định trên development, chưa chứng minh giảm tổng thời gian.

Wilson 95% CI của B là **23,1–88,2%**, T là **37,6–96,4%**. Đây là 5 seed
development đã dùng để chọn thiết kế; observed 80% **không** chứng minh mục tiêu
IID ≥80/100, population ≥80%, hay cải thiện Position/Camera/Lighting OOD.
Cohort regression và fresh holdout vẫn thuộc H5 sau H3/H4.

Chứng cứ bản cuối:

- [Manifest](../artifacts/p5_h2_final_20261015/plan.json) và
  [paired summary](../artifacts/p5_h2_final_20261015/summary.json).
- [B report](../artifacts/p5_h2_final_20261015/B/iid.json),
  [T report](../artifacts/p5_h2_final_20261015/T_terminal/iid.json),
  [T tick trace](../artifacts/p5_h2_final_20261015/T_terminal/iid.jsonl).
- [Verification](../artifacts/p5_h2_final_20261015/verification.json): transition
  guards, command/slew bounds, source hash, H0 config hash không đổi; resume
  không chạy lại physics và giữ nguyên summary checksum.

**Kiểm chứng tự động**

```bash
.venv/bin/python -m pytest -q \
  tests/test_p5_h2.py tests/test_p5_h1.py tests/test_p5_h0.py \
  tests/test_visual_bridge.py tests/test_visual_runtime.py
```

**62 passed**. Tests H2 kiểm tra bounded Cartesian/angular command và slew,
quy ước delta env, giữ grip/preload, contiguous stable ticks, speed/source-age
guard, clearance/orientation/opening/goal jump, emergency hold, no-progress,
budget không ép place, reset và input 67-D. Có test độ mở hộc giảm từ 17 xuống
15 cm khi historical `opened` vẫn true: terminal không được chuyển phase.

**Config và tái tạo**

- [p5_controller_h2_dev.json](../configs/p5_controller_h2_dev.json): B/T cùng
  perception và mọi tham số; chỉ khác `transfer_mode`.
- [p5_h2_development_seeds.json](../configs/p5_h2_development_seeds.json):
  5 seed development cũ, không dùng để tuyên bố fresh generalization.
- Artifact bản cuối: `artifacts/p5_h2_final_20261015/`.

Lệnh đã dùng; để chạy lại, chọn output mới:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --variants-config configs/p5_controller_h2_dev.json \
  --purpose regression_replay \
  --seed-file configs/p5_h2_development_seeds.json \
  --reference-report iid=artifacts/p5_h0_20261015/baseline.json \
  --profiles iid --output artifacts/p5_h2_repeat

MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_h2_repeat
```

Để chạy riêng terminal mode bằng evaluator, thêm `--transfer-mode terminal`
và giữ `--place-contact-mode off`. H3 calibrated contact và H4 verified release/
retreat vẫn là task riêng; không bật raw force 8 N để bù thất bại pha sau.
