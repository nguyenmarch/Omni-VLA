# P5 H4: verified release và retreat

Ngày: **2026-10-16**. Trạng thái: **H4 done — đạt gate development và physics audit**.
Phạm vi: H4 trong [P5_CONTROL_HANDOFF.md](P5_CONTROL_HANDOFF.md).
Checkpoint BC/P5, input 67-D, 10 phase token, impedance gains, scene production
và scoring success vật lý giữ nguyên. Chưa phải nghiệm thu IID/OOD H5.

**Triển khai**

[simulation/verified_placement.py](../simulation/verified_placement.py) thêm
`VerifiedPlacement`, chỉ bật bằng `--place-contact-mode calibrated`.
Baseline `off` tiếp tục dùng release/retreat legacy; `raw_debug` chỉ để tái hiện
trigger raw-force lỗi. Calibrated mode tự bật H3 observer qua normalized options.

FSM ở các phase hiện có:

| Phase | Substate | Điều kiện tiến tiếp |
| :--- | :--- | :--- |
| 5 — place | `support_or_valid_pose` | Geometry, retained grasp, speed và evidence hợp lệ ổn định |
| 5 — contact veto | `collision_hold → collision_correction` | Hold rồi correction Z có giới hạn; không nhả khi veto |
| 6 — release | `release → verify_open` | Ra lệnh mở rõ ràng, giữ pose, xác nhận cả hai finger mở + unload + camera mới |
| 7 — retreat | `retreat_clear` | Servo Z về clearance, kẹp trái mở, tay phải còn giữ handle |
| 8 — close | `retreat_verified` | Chỉ cho BC đóng hộc khi clearance còn hợp lệ; trái tiếp tục hold ở cao độ an toàn |

Phase token đơn lẻ không cấp quyền mở/retreat: FSM còn kiểm tra authorization
và `open_verified`. `reset_episode()` xóa toàn bộ authorization, anchors,
limiter, budget và H3 calibration. Không đọc GT object/contact/success trong
controller; các trường đó chỉ vào scoring/audit sau action.

**Authorization và xác nhận nhả**

- Geometry phải usable và source sim age **≤0,24 s**, gồm fallback age. Cơ chế
  wall-age của visual runtime vẫn hoạt động. Current opening >16 cm, perceived
  bottle footprint trong vùng đã trừ bán kính, đáy chai không quá thấp hoặc
  cách floor quá **35 mm**. Center jump >30 mm bị veto.
- `verified_valid_pose` cho phép thả một khoảng nhỏ trong envelope hợp lệ,
  không tự nhận đã tiếp xúc đáy. Cần EE gần placement goal, grasp còn giữ,
  H3 calibrated, vận tốc cả hai tay ≤0,025 m/s trong **3 tick**.
- `calibrated_support` cần H3 support thật sự và geometry sát floor (drop
  ≤10 mm), cùng các guard về freshness/grip/speed. Raw Fz 8 N không cấp quyền.
  Unknown/bad calibration không dùng làm bằng chứng nhả theo force.
- Action BC bị ép đóng cả hai gripper ở phase 5. Khi đã authorize, controller
  mở trái rõ ràng (`+1`), giữ phải đóng (`-1`) và giữ loaded pose/orientation.
- Chỉ chuyển sang retreat sau khi **cả hai finger trái >32 mm**, đủ stable
  tick; raw/filtered load đã giảm, force window ổn định; perceived bottle còn
  trong placement envelope và tách khỏi EE. Cần image source sau thời điểm
  aperture bắt đầu mở, không tái dùng ảnh trước khi mở kẹp để xác nhận.
- Khi mất camera trước lệnh mở thì giữ đóng; sau khi lệnh mở đã phát thì giữ
  mở và hold, không tự đóng lại trên chai. Bad sensor hoặc budget hết tạo
  failure reason, không ép transition.

**Servo, collision và ngân sách**

Giữ anchor EE/orientation đo lúc authorize, cùng preload BC của tay phải để
giữ loaded handle pose. Tay trái retreat theo Z đến **0,78 m**, giữ XY anchor,
không tiếp tục chạy BC phase 7 khi EE đã vượt goal. Target env vẫn là delta
từ EE hiện tại, không tích phân target lần hai. Giới hạn command trong servo:
**0,15 m/s**, **0,75 m/s²**, **0,5 rad/s**, **2 rad/s²**. Đây không phải bảo đảm
gia tốc vật lý toàn robot hoặc slew xuyên qua policy BC.

Clearance cần sai số Z <12 mm, speed ≤0,025 m/s, XY drift <20 mm, phải giữ
grasp/pose, trái còn mở; giữ 3 tick trước close. Trong phase 8, trái vẫn do
servo giữ ở clearance; phải dùng BC để đóng hộc. Nếu clearance mất, lệnh đóng
phải bị chặn. Emergency stale/failure hold có thể giảm command nhanh hơn
slew limiter bình thường.

Collision veto H3 dẫn tới hold 10 tick, rồi correction Z lên 8 mm mỗi lần,
tối đa 2 lần/16 mm; mỗi correction có budget 50 tick. Không mở kẹp để thoát
veto. Open budget 60 tick, verify budget 50 tick, stale geometry 25 tick,
retreat 200 tick và no-progress 50 tick. Có reason riêng cho open timeout,
verification timeout, perception/calibration/sensor invalid, collision timeout,
retreat no-progress, mất aperture và clearance.

H4 còn phân biệt veto đang có lực mới và veto chỉ còn latch lịch sử. Chỉ re-arm
veto lịch sử khi 5 tick liên tiếp có raw/filtered load trở lại trong ±0,25 N,
force window yên, lực ngang/moment <80% giới hạn H3, EE đủ chậm và camera còn
mới với footprint trong hộc. Va chạm mới lập tức bật lại veto; mất tải do ma
sát thành không đáp ứng điều kiện này. Không sửa calibration hay thresholds H3.

Mọi tham số được pin trong controller options của manifest. CLI cung cấp các
cờ `--release-*`; không đổi baseline contract H0.

**Protocol và các run không được chọn**

Panel v1 phát hiện confound ở tick 0: render đầu dao động quanh wall-age 0,4 s,
variant này nhận ảnh ngay, variant kia hold 5 tick. F có thể khác B trước H4,
nên không dùng số success đó để kết luận về controller.

Thêm protocol tường minh `--startup-hold-steps 5` vào suite/evaluator: chỉ hold
5 action đầu của **episode đầu tiên** trong mỗi run variant, giữ mặc định 0
cho cách chạy cũ. Đây là prefix đã quan sát trong trace baseline H0/H2; mọi
variant dùng cùng protocol, được pin riêng trong plan/report và kiểm tra khi
resume. Không nới freshness/deadline để ép metric.

Run v2 bị harness dừng trước paired analysis: B seed 76002 gặp wall-age 3,35 s
tại step 232, đổi sang stale hold và drift số bước so với reference. Report
được giữ tại `artifacts/p5_h4_dev_v2/`; không tính thành regression controller.
V3 có cùng startup protocol nhưng T gặp stale hold ở phase 1 (wall-age 0,71 s),
F cũng khác B trước H4. Những run này chỉ dùng để chẩn đoán, không phải phép
so sánh controller hợp lệ. V3 còn lộ vấn đề H3 latch lịch sử khi tải đã về
danh định; H4 bổ sung re-arm có điều kiện như trên. Source thay đổi khi v3
đang chạy nên harness từ chối tổng hợp; các report trung gian được giữ nguyên.

Tách phép đo controller bằng `--observation-age-clock simulation`: trong
accelerated MuJoCo, physics đứng trong khi render/host bị block nên wall-age
không phản ánh độ già của state đã mô phỏng. Mode này vẫn chặn sim age và
fallback age, giữ cadence/action dt; wall-age, `wall_stale_results` và deadline
miss tiếp tục được đo. Mode được pin chung cho mọi variant trong plan/report
và kiểm tra khi resume. **Mặc định vẫn là `both`**, kiểm tra cả wall/sim age;
benchmark simulation-clock không chứng minh độ ổn định wall-clock deployment.

**Kết quả và kiểm chứng**

Run được chọn: `artifacts/p5_h4_dev_v4/`, paired B/T/F/TF trên development
76000–76004, scene gốc, simulation clock và startup prefix chung. Đủ completion
receipts, source hash khớp; resume không rerun physics. B và T có **toàn bộ
action/EE/transition** trùng các trace đã đóng băng của H2 (2.622 / 3.360 tick),
không chỉ trùng success mask. F/B và TF/T trùng trajectory trước phase 5.

| Seed | B | T (H2) | F (H4) | TF (H2 + H4) |
| :--- | :--- | :--- | :--- | :--- |
| 76000 | Success, 453 | Success, 641 | Success, 464 | Success, 654 |
| 76001 | Success, 470 | Success, 619 | Success, 474 | Success, 637 |
| 76002 | Timeout retreat, 747 | Timeout retreat, 922 | **Success, 513** | **Success, 716** |
| 76003 | Timeout transfer, 512 | **Success, 549** | Timeout transfer, 512 | **Success, 585** |
| 76004 | Success, 440 | Success, 629 | Success, 539 | Success, 658 |
| Tổng | **3/5** | **4/5** | **4/5** | **5/5** |

Số sau kết quả là tick tại 25 Hz. F/B: **1 rescue, 0 regression**; TF/B:
**2 rescue, 0 regression**. Riêng tác động H4 khi đã có H2 (TF/T): cứu thêm
76002, giữ cả 4 success cũ. Seed F 76003 thất bại ở transfer trước H4.
Đây là rescue của **development panel**, chưa phải số được cứu trong cohort
26 timeout cũ/100 IID ở các seed 50000–50099.

Physics audit trên 9 placement hoàn tất của F/TF:

- **0** retreat trước aperture mở thực; **0** đóng hộc trước clearance.
- **0** release command hoặc actual opening ngoài envelope của env.
- **1.476 tick** servo được kiểm tra linear/angular command bounds, không vi phạm.
- **9** release theo `verified_valid_pose`; **0** positive force-release event.
  Vì vậy chưa có phép đo latency/effectiveness của force-triggered release trên
  scene gốc. Guard nhánh force được test ở unit level; H3 có positive contact
  trên jig riêng. Không quy rescue cho một Fz trigger chưa phát.
- H4 F re-arm latch lịch sử ở 3 episode; correction tối đa 1 lần trên panel
  này. TF có correction 1 lần ở 76002/76003/76004, không phải retry vô hạn.

Đánh đổi: trên 3 success cũ của B, median tổng tick **453 → 474** khi dùng F;
TF **654** so với T **629**. Riêng F 76004 tăng 440 → 539 tick do hold/correction
và retreat ổn định. Chưa chứng minh giảm tổng thời gian trên mọi episode.

Wilson 95% CI: B **23,1–88,2%**, F/T **37,6–96,4%**, TF **56,6–100%**.
5/5 là development đã dùng để chọn thiết kế, không chứng minh population ≥80%,
IID ≥80/100 hay cải thiện OOD. H5 vẫn cần frozen-candidate replay/holdout.

Smoke riêng ở **clock `both`** (không bỏ wall freshness): TF seed 76000 đạt
success ở **654 tick**, action/EE/transition trùng episode TF đầu của panel.
Có **132/654 deadline miss**, nên cả smoke thành công này cũng không chứng
minh realtime. Panel simulation-clock có 4–129 wall-stale result/variant dưới
tải host; các số đó được giữ trong report, không giấu bằng success rate.

```bash
.venv/bin/python -m pytest -q \
  tests/test_p5_h4.py tests/test_p5_h3.py tests/test_p5_h2.py \
  tests/test_p5_h1.py tests/test_p5_h0.py \
  tests/test_visual_bridge.py tests/test_visual_runtime.py
```

Các counterexample H4 gồm force
không có geometry, source stale/fallback quá tuổi, grasp mất, phase token chưa
authorize, collision correction quá budget, kẹp chưa mở/không unload, ảnh cũ
trước khi mở, retreat chưa clear và no-progress. Tests H0/H1/H2/H3 tiếp tục chạy.
**112 tests pass**, gồm hai ca cơ học bên dưới và checks pin startup/age-clock
trong report/resume. Simulation-clock test vẫn buộc hold khi sim age quá hạn,
kể cả khi wall freshness chỉ đang được đo.
Test physics thật trên scene gốc dùng ideal landmarks để cô lập cơ học:
scripted expert chỉ tạo grasp prelude, sau đó H4 điều khiển release/retreat;
không dùng contact GT để authorize. Ca thường có bottle–floor contact sau nhả
và đạt clearance trước close; ca chặn joint range của finger để mô phỏng kẹt
phát `visual_release_open_timeout`, không chuyển retreat. Benchmark B/T/F/TF
riêng bên dưới dùng landmark P5 học được, không dùng ideal/GT landmarks.

Chứng cứ:

- [Plan](../artifacts/p5_h4_dev_v4/plan.json),
  [paired summary](../artifacts/p5_h4_dev_v4/summary.json),
  [physics audit](../artifacts/p5_h4_dev_v4/h4_audit.json).
- [TF report](../artifacts/p5_h4_dev_v4/TF/iid.json),
  [TF trace](../artifacts/p5_h4_dev_v4/TF/iid.jsonl),
  [test log](../artifacts/p5_h4_dev_v4/tests.log).
- [Both-clock smoke](../artifacts/p5_h4_both_clock_smoke_20261016.json).
- [Verification tổng](../artifacts/p5_h4_verification_20261016.json): guards,
  H0 contract/scene/H3 thresholds không đổi, replay sensor H3 15 trial còn
  giống hệt, source/hash/receipt và resume không chạy lại physics.

**Lệnh tái tạo**

Chọn output mới; evaluator/audit không ghi đè report:

```bash
.venv/bin/python -m scripts.evaluate_visual_suite prepare \
  --variants-config configs/p5_controller_h4_dev.json \
  --purpose regression_replay --seed-file configs/p5_h2_development_seeds.json \
  --reference-report iid=artifacts/p5_h0_20261015/baseline.json \
  --profiles iid --startup-hold-steps 5 --observation-age-clock simulation \
  --output artifacts/p5_h4_repeat

MUJOCO_GL=egl .venv/bin/python -m scripts.evaluate_visual_suite run \
  --output artifacts/p5_h4_repeat

.venv/bin/python -m scripts.audit_verified_placement \
  --suite artifacts/p5_h4_repeat
```

Đánh giá deployment riêng có thể bật `--place-contact-mode calibrated` và
`--transfer-mode terminal` để dùng TF. Đây là opt-in development cho tới H5;
chưa chứng minh IID ≥80/100, population ≥80%, OOD hoặc hard realtime.
