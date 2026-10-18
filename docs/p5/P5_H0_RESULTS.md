# P5 H0: khóa baseline và công tắc contact

Ngày: **2026-10-15**. Trạng thái: **done**.
Phạm vi: H0 trong [P5_CONTROL_HANDOFF.md](./P5_CONTROL_HANDOFF.md).
Cập nhật: H1 đã hoàn tất, xem [P5_H1_RESULTS.md](./P5_H1_RESULTS.md) cho CLI suite v3.
Đây là xác nhận baseline/reproduction trên development, chưa phải benchmark
IID 100 episode hoặc bằng chứng đạt success ≥80%.

**Thay đổi đã thực hiện**

- `VisualTaskManager`, visual evaluator và suite dùng mặc định transfer **12 mm**,
  `place_contact_mode=off`. Pose-based release và release/retreat legacy được giữ.
- CLI có `--place-contact-mode {off,calibrated,raw_debug}`. `off` bỏ hoàn toàn
  nhánh force transition; không cần ngưỡng giả 1.000.000 N để vô hiệu hóa nữa.
- `raw_debug` phải bật tường minh để tái hiện trigger lỗi cũ. Nó vẫn phát
  `wrench_contact`, nhưng report ghi rõ mode diagnostic.
- `calibrated` là mode dành cho H3: hiện từ chối ngay trước khi load model/chạy
  môi trường, với thông báo cần H3. Chưa có detector hiệu chuẩn trong H0 và
  không âm thầm thay bằng raw-force.
- `reset_episode(points)` xóa phase/age, geometry, contact streak/Fz, transition
  reason và các cờ hoàn thành/thất bại. Evaluator tiếp tục tạo manager riêng
  cho mỗi episode và reset perception cache như trước.
- Report và suite manifest lưu contact mode; resume kiểm tra mode tương thích.
  H1 vẫn cần triển khai cấu hình controller riêng theo variant.
- [Config baseline](../../configs/p5_control_baseline.json) cố định arguments,
  checkpoint hashes, development seeds/mask và reference reports.
- [Replay H0](../../scripts/replay_p5_h0.py) thực thi config, đối chiếu theo seed,
  kiểm tra contact bằng oracle chỉ để scoring, lưu provenance và từ chối
  output directory đã tồn tại.

**Kết quả replay**

| Seed | Success kỳ vọng | Success replay | Steps | Kết thúc |
| :--- | :--- | :--- | ---: | :--- |
| 76000 | true | true | 453 | success |
| 76001 | true | true | 470 | success |
| 76002 | false | false | 747 | visual_timeout_retreat |
| 76003 | false | false | 512 | visual_timeout_transfer |
| 76004 | true | true | 440 | success |

Baseline đạt **3/5**, mask `[true, true, false, false, true]`; **0 contact
trigger**. `success`, `steps`, `phase`, `failure_reason` khớp reference cho
cả 5 seed. Perception vẫn fallback bật, mask/EMA tắt và cache age 0,4 s.

Raw-debug seed 76000, transfer 25 mm, threshold 8 N, debounce 3:

- Phát đúng một trigger ở **step 230**, world Fz **8,925802705 N**.
- Không có contact body pair `bottle`–`drawer` tại trigger; tái hiện đúng
  false positive đã biết, chỉ khi người chạy chọn `raw_debug`.
- Episode kết thúc `visual_timeout_retreat` ở step **601**, khớp reference.
  Đây là reproduction của lỗi, không phải một rescue.

Replay runner trả `h0_passed: true` và tất cả checks đều true. Source hash
trước/sau run giống nhau; provenance có config/reference/data hashes,
dependency versions, git commit/status và arguments thực tế.

**Kiểm chứng**

Lệnh sau đạt **27 passed**:

```bash
.venv/bin/python -m pytest -q \
  tests/test_p5_h0.py tests/test_visual_bridge.py tests/test_visual_runtime.py
```

Bao gồm baseline force-off với wrench lớn/lỗi tại manager, pose release legacy,
transfer mặc định 12 mm, xóa pending contact giữa episode, calibrated fail-fast,
raw-debug debounce và resume từ chối report khác mode. Fixture test resume cũ
được cập nhật cho chữ ký/config hiện hành. `compileall` và `git diff --check` đạt.

**Artifacts và lệnh chạy lại**

Kết quả đã chạy nằm ở `artifacts/p5_h0_20261015/`:

- [summary.json](../../artifacts/p5_h0_20261015/summary.json): gates, mask và drift.
- [manifest.json](../../artifacts/p5_h0_20261015/manifest.json): provenance.
- [baseline.json](../../artifacts/p5_h0_20261015/baseline.json) và `baseline.jsonl`.
- [raw_debug.json](../../artifacts/p5_h0_20261015/raw_debug.json) và `raw_debug.jsonl`.
- [contact_audit.json](../../artifacts/p5_h0_20261015/contact_audit.json): scoring-only.

Chạy lại từ repo root với output mới:

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.replay_p5_h0 \
  --config configs/p5_control_baseline.json \
  --output artifacts/p5_h0_repeat
```

Để chỉ chạy baseline bằng evaluator:

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --policy weights/bc_kaggle_v2/best.pt \
  --grounding weights/p5_student/best.pt \
  --seeds 76000 76001 76002 76003 76004 \
  --place-contact-mode off --transfer-tolerance .012 \
  --output artifacts/p5_h0_baseline_repeat.json
```

Các output trên phải chưa tồn tại. Artifact cũ cần có khi chạy replay để xác
nhận hash và đối chiếu. Các report trong tài liệu này là baseline H0; phép so
controller mới dùng harness v3 đã kiểm chứng ở H1.
