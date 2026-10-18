# P5 H3: observer lực đặt chai đã hiệu chuẩn

Ngày: **2026-10-15**. Trạng thái: **H3 done — đạt gate development**.
Cập nhật 2026-10-16: H4 đã mở calibrated release/retreat opt-in; xem
[P5_H4_RESULTS.md](P5_H4_RESULTS.md). Các mô tả shadow/fail-fast bên dưới là
snapshot tại thời điểm nghiệm thu H3; `off --placement-observer` vẫn là shadow.
Phạm vi: observer và audit H3 trong [P5_CONTROL_HANDOFF.md](P5_CONTROL_HANDOFF.md).
Đây là kiểm chứng detector trên development, không phải nghiệm thu success IID/OOD.

**Code và cách dùng**

- [simulation/placement_contact.py](../simulation/placement_contact.py):
  `PlacementContactObserver`, API `reset_episode()` và `update(robot, dt)`.
- [scripts/evaluate_placement_contact.py](../scripts/evaluate_placement_contact.py):
  thu dữ liệu có tiếp xúc thật, audit replay, tái hiện false positive đã biết
  và replay action của visual rollout trong physics.
- [configs/p5_placement_observer.json](../configs/p5_placement_observer.json):
  các tham số development đã pin, trùng defaults dùng bởi visual runtime.
- Evaluator hỗ trợ `--placement-observer`: chạy **shadow**, thêm telemetry,
  không thay action, không cấp quyền nhả kẹp. Cờ mặc định tắt; input BC vẫn 67-D.
- [p5_controller_h3_shadow.json](../configs/p5_controller_h3_shadow.json):
  B/S chỉ khác boolean `placement_observer`; cùng checkpoints, perception và
  baseline controller H0, contact mode `off`.

Observer nhận `wrench[12]`, `wrist_rotations[2,3,3]`, `ee[6]`,
`ee_velocity[2,3]`, `gripper[4]`, `placement_stage` từ finite-task manager.
Không đọc object pose, contact, success, teacher action hay landmark camera.
Collector dùng scripted expert và fixture goals để tạo dữ liệu; chúng không
được truyền vào observer. Nhãn contact chỉ được đọc sau prediction.

H3 trả `calibration_status`, expected/filtered/residual wrench, estimated load
loss, support confidence/streak, `collision` veto và `hold_requested`.
`support_confidence` là tỷ lệ mất tải đã chuẩn hóa khi các guard hợp lệ,
**không** phải xác suất contact đã được statistical calibration.

**Hiệu chuẩn và logic detector**

MuJoCo force sensor đo tương tác child–parent trong site frame, bao gồm các
lực tác động lên hệ. Vì vậy raw wrist Fz không phải lực tiếp xúc đáy riêng.
Nguồn: [MuJoCo force/torque sensor reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html#sensor-force).

Force và torque đều được quay từ wrist site sang world; torque vẫn tính quanh
cùng wrist origin, không chuyển sang moment quanh world origin. Hiệu chuẩn
cục bộ dùng reference rỗng và reference mang chai để bù tổng gravity/bias trong
grasp hiện tại; không nhận dạng riêng một global tool inertial model.

1. Stage 0/1, fingers mở ≥32 mm, speed/acceleration đủ nhỏ: tích lũy 12 mẫu,
   lấy median empty wrench. Bỏ mẫu sensor chưa ổn định có world Fz ≤1 N.
2. Stage 3/4, EE trái Z>0,69 m, grip còn giữ vật, các guard động học/lực/
   orientation hợp lệ: lấy 12 mẫu carry. Chỉ chấp nhận reference ít nhiễu,
   apparent payload trong 0,4–3 N. Reference freeze khi vào place.
3. Tính `residual = filtered_world - expected_free_world` và
   `load_loss = -residual_z`. Trong grasp này, đáy đỡ vật làm Fz **giảm**.
4. Support cần cả raw và filtered unloading cùng chiều, đủ lớn, lực ổn định,
   tốc độ/acceleration thấp, grip/orientation còn hợp lệ và không có collision
   veto trong 3 tick liên tiếp. Không chỉ xét median của Fz.
5. Lực ngang hoặc moment lớn tạo veto. Veto chỉ được re-arm khi trở về stage
   free-space hoặc tải mang danh định đã hồi phục và lực yên qua 5 tick.
   Side friction vẫn làm mất tải thì không được tự đổi thành support.

| Tham số | Giá trị bản cuối |
| :--- | :--- |
| Filter | Rolling median 5 tick |
| Support on / off | max(0,5 N, 60% / 35% apparent payload) |
| Stable force window | Max std của các force component ≤0,4 N |
| Support debounce | 3 tick |
| Contact speed / acceleration tối đa | 0,02 m/s / 0,8 m/s² |
| Calibration speed / acceleration tối đa | 0,16 m/s / 0,6 m/s² |
| Lateral force / moment veto | >1 N / >0,12 Nm |
| Local orientation / grip change tối đa | 0,2 rad / 6 mm |

Missing/bad/saturated sensor xóa reference và support streak, trả
`sensor_invalid`. Orientation ngoài miền hoặc grip thay đổi trả trạng thái
unknown tương ứng và support false. Không học reference từ contact ở place.
Các buffer đều bounded và reset theo episode.

Observer được gọi sau mỗi physics step, trước capture/inference tiếp theo,
kể cả tick visual đang stale hoặc chưa tạo được manager. Trace thêm
`placement_observer` trước action và `placement_after_update` sau physics;
residual world trong khối diagnostic là 6-D của wrist trái, phải chưa hiệu chuẩn.
Test integration có physics thật xác nhận camera stale không chặn force update
và không tạo phase transition.

**Ranh giới với H4**

H3 chỉ đưa ra bằng chứng lực và yêu cầu hold, chưa thực thi hold/correction hoặc
release từ bằng chứng đó. `--place-contact-mode calibrated` tiếp tục fail-fast,
với thông báo cần H4 geometry/open/retreat verification; không âm thầm dùng raw
force. Khi bật shadow cùng `off`, baseline pose release vẫn giữ nguyên và
không tiêu thụ support/unknown từ observer.

H4 phải ghép tín hiệu này với geometry còn mới, valid placement pose, xác nhận
mở kẹp rồi retreat clearance. Support từ wrench đơn lẻ không định danh được
vật đang tựa đáy hộc, bàn hay một bề mặt khác. Unknown không được cấp quyền
nhả theo force. Không gọi shadow mode là candidate F/TF đã hoàn chỉnh.

**Dữ liệu calibration development và kết quả**

Panel bản cuối gồm 76000–76004 × 3 profile, **15 trial / 6.024 sensor sample**.
Mỗi trial có grasp/lift/transfer bằng friction thật, 50 tick carry để ổn định
reference, rồi 140 tick probe. Không weld, teleport object trong rollout hoặc
tiêm force vào sensor.

Positive bottom probe dùng jig: `drawer_floor` được nâng **12 cm ngay trong
XML trước reset/compile**, để chai chạm được bề mặt khi grasp còn giữ ổn định.
Free/side dùng scene gốc. XML jig và hash được lưu cùng manifest; không sửa
scene/checkpoint production. Đây là kiểm chứng tiếp xúc thật trên jig, chưa
chứng minh detector nhận được mọi contact đáy trong rollout scene gốc.

Nhãn positive yêu cầu đồng thời có bottle–floor và bottle–finger contact. Chai
đã rơi xuống đáy sau khi tuột kẹp không được tính là support trong grasp.

| Profile | Trial | GT positive tick | Support tick | False support tick | Missed bottom trial |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Free space | 5 | 0 | 0 | **0** | — |
| Side wall | 5 | 0; có 270 side-contact tick | 0 | **0** | — |
| Raised bottom jig | 5 | 494 | 457 | **0** | **0/5** |

Đã phát hiện bottom ở **5/5** trial. Latency đầu contact → support là
**0,28–0,32 s simulation (7–8 tick)** và **0,0288–0,0513 s wall** trong collector
CPU chạy nhanh hơn realtime. 37 positive tick chưa báo support nằm trong các
đoạn chờ detector ổn định. P50/P95 riêng `observer.update()` khoảng
**0,249 / 0,468 ms** trên run này.

Collision veto có 20 tick trong một free trial và 23 tick trước contact của
một bottom trial: detector còn bảo thủ với transient/grasp disturbance.
Các request đó chưa tác động action ở shadow. H4 phải audit tác động hold và
re-arm, không coi mọi veto là ground-truth side collision.

**Ca đối chứng và thiết kế đã loại bỏ**

- H0 seed 76000/step 230 được replay bằng full action trace H1, bao gồm 5 tick
  await-perception ban đầu mà trace H0 không lưu. Fz tái lập chính xác
  **8,925802705 N**, không có bottle–drawer contact. Observer đã **calibrated**,
  load loss **−0,3596 N**, support **false**, không có false support trước claim.
- Bản đầu median + speed guard vượt qua jig panel, nhưng visual shadow audit
  phát hiện **13 false-support tick** ở seed 76000 khi lực dao động do hai tay
  va nhau. Artifact `artifacts/p5_h3_shadow_20261015/` giữ nguyên kết quả này;
  bản đó **không được chọn**.
- Bản cuối thêm raw/filtered agreement, variance guard và contact speed 0,02
  m/s. Đánh đổi latency jig từ 0,16 s lên 0,28–0,32 s simulation để loại ca trên.
- Replay 10 trajectory B/T của H2 trên 76000–76004: **5.982 tick**, mọi EE
  position tái lập với sai số **0**, success cuối giống reference, **0 false
  support tick**. Các trajectory này không có support-in-grasp được detector
  xác nhận; bằng chứng sensitivity nằm ở positive jig riêng, không dùng việc
  luôn trả false trên visual trajectory để tuyên bố detector tốt.

Đây đều là development/replay seed đã dùng; không gọi là holdout hay OOD.

**Visual runtime bản cuối**

Paired B/S trên 76000/76003 đã hoàn tất, có completion receipts và source hash
khớp manifest. B tái lập H0; B/S có toàn bộ **965 tick** trùng action, vị trí EE
sau step và transition. Cả hai cùng success seed 76000 ở 453 tick, cùng timeout
transfer seed 76003 ở 512 tick: **1/2**, **0 rescue / 0 regression** như kỳ vọng
của shadow. Không dùng run này để kết luận H3 làm tăng success rate.

Shadow cập nhật observer **965/965 physics tick**, **0 support / 0 false
support tick**, P50/P95 riêng update khoảng **0,253 / 0,460 ms**. Cả B/S có
**194/965 deadline miss** (>40 ms work/tick), chủ yếu là nút thắt visual còn
đồng bộ; riêng observer nhanh không làm toàn hệ thống thành realtime.
Resume đã kiểm tra receipts/hashes và giữ nguyên summary checksum, không
chạy lại policy/physics.

**Kiểm chứng và artifacts**

```bash
.venv/bin/python -m pytest -q \
  tests/test_p5_h3.py tests/test_p5_h2.py tests/test_p5_h1.py \
  tests/test_p5_h0.py tests/test_visual_bridge.py tests/test_visual_runtime.py
```

**84 passed**: frame/sign, calibration validity, support debounce/hysteresis,
side force/moment veto và load-restored re-arm, oscillating wrist load dù EE
chậm, speed/acceleration/orientation/grip guards, reset/bad sensor, oracle
poisoning, config validation, sensor tick độc lập camera freshness; cùng các
regression tests H0/H1/H2. Audit offline tái chạy prediction từ `robot` đã lưu,
không đưa nhãn scoring vào observer.

- [Calibration manifest](../artifacts/p5_h3_final_calibration_20261015/manifest.json),
  [report](../artifacts/p5_h3_final_calibration_20261015/report.json),
  [sensor samples](../artifacts/p5_h3_final_calibration_20261015/samples.jsonl).
- [Known-case report](../artifacts/p5_h3_final_known_76000_20261015/report.json).
- [10 frozen-action replays](../artifacts/p5_h3_frozen_action_audit_20261015/summary.json).
- [Runtime paired summary](../artifacts/p5_h3_final_shadow_20261015/summary.json),
  [shadow trace](../artifacts/p5_h3_final_shadow_20261015/S_shadow/iid.jsonl).
- [Verification tổng](../artifacts/p5_h3_verification_20261015.json): 84 tests,
  exact sensor replay trên 15 trial, known-case, 10 frozen-action trajectory,
  paired runtime, source hash thống nhất, H0 config không đổi và resume không
  chạy lại physics.

**Lệnh tái tạo**

Dùng output mới; scripts từ chối ghi đè. Calibration không cần GPU/checkpoint:

```bash
.venv/bin/python -m scripts.evaluate_placement_contact collect \
  --config configs/p5_placement_observer.json \
  --seeds 76000 76001 76002 76003 76004 \
  --output artifacts/p5_h3_calibration_repeat

.venv/bin/python -m scripts.evaluate_placement_contact audit \
  --dataset artifacts/p5_h3_calibration_repeat

.venv/bin/python -m scripts.evaluate_placement_contact replay-known \
  --trace artifacts/p5_h1_20261015/Raw_contact_debug/iid.jsonl \
  --reference-report artifacts/p5_h0_20261015/raw_debug.json \
  --output artifacts/p5_h3_known_repeat

.venv/bin/python -m scripts.evaluate_placement_contact replay-actions \
  --trace artifacts/p5_h2_final_20261015/B/iid.jsonl \
  --reference-report artifacts/p5_h2_final_20261015/B/iid.json \
  --seed 76000 --output artifacts/p5_h3_actions_repeat
```

Visual paired replay cần P5/BC artifacts giống H0/H2:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --variants-config configs/p5_controller_h3_shadow.json \
  --purpose regression_replay --seed-file configs/p5_h1_replay_seeds.json \
  --reference-report iid=artifacts/p5_h0_20261015/baseline.json \
  --profiles iid --output artifacts/p5_h3_shadow_repeat

MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_h3_shadow_repeat
```

Không suy từ latency collector sang deadline wall-clock của visual runtime:
render/inference đồng bộ còn có thể chặn vòng điều khiển. Chưa có cam kết
hard realtime, sensitivity ở mọi grasp/orientation, hay success ≥80% do H3.
