# P5 control: kế hoạch triển khai và handoff

Ngày lập: **2026-10-15**. Cập nhật: **2026-10-16**.
Trạng thái: **H0–H4 done; H5–H7 todo**.
Chứng cứ H0: [P5_H0_RESULTS.md](P5_H0_RESULTS.md).
Chứng cứ H1: [P5_H1_RESULTS.md](P5_H1_RESULTS.md).
Chứng cứ H2: [P5_H2_RESULTS.md](P5_H2_RESULTS.md).
Chứng cứ H3: [P5_H3_RESULTS.md](P5_H3_RESULTS.md).
Chứng cứ H4: [P5_H4_RESULTS.md](P5_H4_RESULTS.md).
Nguồn quyết định: [P5_CONTROL_RESEARCH.md](P5_CONTROL_RESEARCH.md).
Người tiếp nhận bắt đầu ở **H5** sau khi đọc kết quả H0–H4, thực hiện theo các dependency, ghi kết quả
vào artifact riêng cho từng lần chạy.

**Mục tiêu và tiêu chí hoàn thành**

| Mục tiêu | Chứng cứ cần bàn giao | Điều kiện nghiệm thu |
| :--- | :--- | :--- |
| IID | Baseline/candidate trên cùng seed, success vật lý và Wilson 95% CI | Cohort regression cũ đạt ≥80/100; holdout mới đạt observed ≥80%, báo đầy đủ baseline mới và CI |
| Rescues/regressions | Danh sách seed được cứu, bị hỏng, nguyên nhân và trace release | Cohort cũ có cải thiện ròng ≥9; báo riêng rescue trong 26 timeout được hỏi và mọi regression trong 71 success cũ |
| Position | Paired 20 seed cũ và ít nhất 20 seed holdout mới, số episode tới từng phase | Cohort cũ vượt 5/20; holdout mới cải thiện ròng so với baseline trên cùng seed |

Đặt mục tiêu development là **zero regression**, đặc biệt lỗi release không hợp
lệ do detector mới. Nếu có regression, phải có audit nguyên nhân trước khi
chọn candidate; không ẩn chúng bằng tỷ lệ success tổng. Sample holdout 20 episode
chỉ cung cấp bằng chứng ban đầu cho Position; báo CI và paired test để thể hiện
độ bất định. Observed ≥80% không đồng nghĩa chứng minh population ≥80%:
với N=100, cần ≥88 success để cận dưới Wilson 95% vượt 80%.

**Thông tin bắt buộc khi tiếp nhận**

- IID tham chiếu: **71/100**; Position: **5/20**.
- IID có 18 timeout `transfer`, 4 `descend`, 4 `retreat`, 2 `approach` và
  1 episode perception rejection. Nhóm **26** là transfer + descend + retreat;
  tổng timeout là 28. `descend` là hạ để gắp chai; hạ vào hộc tên `place`.
- Cả 18 transfer timeout IID và 2 transfer timeout Position từng đi qua vùng
  sai số hai EE dưới 20 mm. Nới tolerance đơn thuần vẫn có thể tạo regression.
- Trigger raw `abs(Fz) >= 8 N` đã có false positive: seed 76000 phát ở 8,93 N
  khi chai chưa tiếp xúc hộc; lực nền mang chai đã khoảng 8,58 N.
- Trên development 76000–76004: baseline 12 mm/force off đạt 3/5;
  transfer 25 mm/force off đạt 2/5 (1 rescue, 2 regressions);
  transfer 25 mm/raw force 8 N đạt 0/5 (3 regressions).
- Chỉ 7/20 Position episode tới transfer. Chỉ sửa từ transfer trở đi có trần
  7/20 trên cohort này nếu mọi diễn biến trước transfer giữ nguyên.
- Harness trước H1 dùng cùng controller options cho baseline/candidate và đổi
  perception theo tên variant. H1 đã thay bằng manifest v3 có config riêng,
  xem kết quả H1; các report cũ vẫn là historical artifacts.

**Đầu vào và code liên quan**

| Thành phần | Đường dẫn | Vai trò |
| :--- | :--- | :--- |
| BC v2 | `weights/bc_kaggle_v2/best.pt` | Checkpoint cố định cho vòng controller |
| P5 | `weights/p5_student/best.pt` | Perception cố định cho vòng controller |
| Báo cáo cohort cũ | `artifacts/p5_temporal_fallback_suite/{baseline,candidate}/` | JSON + JSONL theo profile |
| Audit nghiên cứu | `artifacts/p5_control_research/` | Raw reports, contact audit và `research_summary.json` |
| Phase manager | `simulation/vision.py` | Goals, transition, action adaptation |
| Visual evaluator | `scripts/eval_visual_policy.py` | Vòng điều khiển, observations, traces và scoring |
| Suite | `scripts/evaluate_visual_suite.py` | Manifest, seed, paired reports và resume |
| Impedance/env | `models/impedance_controller.py`, `simulation/bimanual_env.py` | Thực thi Cartesian action và sensor |
| Wrench monitor | `simulation/recovery.py` | Tham khảo calibrated load; placement detector phải độc lập với slip |
| Diagnostic trace | `utils/visual_diagnostics.py` | Mở rộng telemetry theo từng tick |
| Collector | `scripts/collect_demonstrations.py` | Collector cũ dùng oracle condition và expert phase |
| Randomization | `simulation/randomization.py` | Position hiện jitter chai ±4 cm XY |

Artifacts và checkpoint lớn có thể không được Git quản lý. Khi chuyển máy,
kiểm tra sự hiện diện, hash và khả năng đọc trước khi bắt đầu. Lưu `git status`,
commit, source hash, dependency versions và dataset/checkpoint hashes vào run
manifest. Giữ các thay đổi có sẵn trong workspace; không reset chúng để tái lập.

**Thứ tự thực hiện**

| Task | Phụ thuộc | Đầu ra |
| :--- | :--- | :--- |
| H0 — Khóa baseline và công tắc controller — done | — | Config baseline và replay development đã khớp; xem kết quả H0 |
| H1 — Harness paired và telemetry — done | H0 | Manifest v3, telemetry và paired replay đã kiểm chứng; xem kết quả H1 |
| H2 — Terminal transfer ổn định — done | H1 | Candidate T, guards và paired development 4/5 vs B 3/5; xem kết quả H2 |
| H3 — Calibrated placement force — done | H1 | Observer shadow, 15 trial calibration/audit, known false positive và visual regression; xem H3 results |
| H4 — Verified release và retreat — done | H3 | Candidate F/TF opt-in, 112 tests, original-scene physics/paired audit; xem H4 results |
| H5 — Chọn candidate và nghiệm thu controller | H2, H4 | Development ablations, regression replay, fresh holdout |
| H6 — Visual DAgger cho lỗi trước transfer | H1; dùng controller đã đóng băng từ H5 | Dataset/BC mới, CPU smoke và notebook Kaggle |
| H7 — Nghiệm thu sau training | H6 | Paired report checkpoint mới trên holdout riêng |

H2–H4 đã hoàn tất; tiếp theo là nghiệm thu/chọn candidate ở H5. Nếu H5 không đạt mục tiêu, giữ trạng
thái **chưa đạt**, ghi rõ lỗi còn lại và tiếp tục bằng dữ liệu corrective ở H6;
không đổi tiêu chí success để vượt gate.

**H0 — Khóa baseline và loại bỏ cấu hình trigger lỗi khỏi mặc định**

1. Thêm chế độ contact tường minh: `off`, `calibrated`, `raw_debug`.
   API đã có qua `--place-contact-mode`; H4 đã mở `calibrated` cho verified
   release/retreat, tự bật H3 observer. `off --placement-observer` vẫn là shadow.
   `raw_debug` chỉ dùng để tái hiện lỗi; không dùng làm mặc định hay
   candidate nghiệm thu.
2. Baseline controller: transfer tolerance 12 mm, force trigger off,
   release/retreat legacy. Baseline B dùng P5 fallback bật, mask/EMA tắt,
   cùng checkpoint/perception config với T/F/TF.
3. Tham chiếu replay B với các report `candidate/` của temporal fallback suite:
   IID 71/100, Position 5/20. Các success theo seed đã giống legacy baseline
   trong nghiên cứu; nguyên nhân rejection có thể khác. Cố định một tham chiếu
   cho mỗi phép so sánh, không trộn report hai perception modes.
4. Tái hiện development 76000–76004, B phải đạt đúng success mask
   `[true, true, false, false, true]`. Tái hiện riêng raw trigger seed 76000
   để detector mới có một ca false positive bắt buộc phải loại bỏ.

Hoàn tất khi cờ off thực sự không phát contact transition, các state reset theo
episode, và baseline replay khớp. Nếu drift, giải thích và xử lý trước H5.
H0 đã đạt các điều kiện này; xem [P5_H0_RESULTS.md](P5_H0_RESULTS.md) và
`artifacts/p5_h0_20261015/summary.json`. Chưa chạy lại cohort IID 100/Position 20
trong H0; đó là phần nghiệm thu H5.

**H1 — Sửa harness để đo đúng tác động controller**

- Manifest có schema version mới; lưu config đầy đủ cho từng variant thay vì
  suy ra legacy perception từ tên `baseline`. Tách `perception_options` và
  `controller_options`; mọi variant cùng seed, checkpoint và physics.
- Phân biệt `development`, `regression_replay`, `holdout`. Cho replay seed cũ
  bằng manifest tham chiếu rõ ràng; seed holdout mới phải không trùng training,
  development hay evaluation đã sử dụng. Không bỏ kiểm tra seed toàn cục để
  lách replay.
- Lưu source/config/checkpoint hashes; reject resume nếu khác config, thiếu
  episode hoặc sai seed/order. Giữ cơ chế từ chối ghi đè report cũ.
- Trace mỗi tick cần: phase/substate, action trước/sau override, EE pose và
  vận tốc, goal error XY/Z, perception source age, raw/world/residual wrench,
  trạng thái hiệu chuẩn, contact debounce, gripper aperture, transition reason.
- Scoring-only trace thêm contact pairs/force, vị trí chai và trạng thái đặt
  vật. Policy/manager không được đọc các trường oracle này.
- Aggregate: phase durations, phase reach rates, deadline misses, latency từ
  contact tới hold/release, false contact, release ngoài vùng hợp lệ và
  rescue/regression theo seed. Phân loại lý do từ baseline đã đóng băng.

Hoàn tất khi một run B/T cố ý khác đúng một option thể hiện khác biệt đó trong
manifest và report; không thể resume bằng option khác. Test paired reporting
phải tính rescue bằng success vật lý cuối episode, không bằng transition.
H1 đã đạt: 51 tests và panel B/T/raw-debug trên 2 seed replay, kèm kiểm tra
resume không rerun physics. B/T chỉ khác transfer tolerance; T này là ablation
debug, chưa phải terminal servo H2. Xem [P5_H1_RESULTS.md](P5_H1_RESULTS.md).

**H2 — Terminal transfer ổn định**

Triển khai substate `bc_transfer → settle → ready_to_place`, không thêm phase
token vào input checkpoint 67-D. Điểm vào settle có thể bắt đầu thử ở vùng
20–30 mm; đây chỉ là khoảng tìm kiếm development, không phải ngưỡng đã nghiệm thu.

- Dùng EE, robot model và landmark camera để servo tới goal; giới hạn tốc độ,
  gia tốc và bước action. Đúng quy ước env hiện tại: mỗi Cartesian action là
  delta từ EE đo hiện thời, scale 0,015 m; không tích phân nhầm target lần nữa.
- Giữ grip và orientation thích hợp; không tự tăng impedance gains trong
  cùng ablation đầu tiên.
- Chuyển place khi cả hai tay đạt envelope XY/Z, vận tốc đủ thấp và giữ ổn
  định trong một cửa sổ. Landmark stale hoặc clearance chưa xác nhận thì hold.
- Có budget settle, phát hiện không tiến bộ và lý do thất bại riêng. Không
  ép chuyển place chỉ vì hết budget, không chỉ tăng timeout chung.
- Log tư thế/vận tốc tại transition; kiểm tra state đầu place có nằm ngoài
  phân phối BC đã gặp hay không.

Hoàn tất khi T không lặp regression đã biết trên development success cũ,
không chuyển khi robot còn chuyển động nhanh/ngoài clearance, và giảm được
thất bại transfer trong development. Tolerance-only 25 mm giữ làm ablation
chẩn đoán; không mặc định chọn vì thời gian pha ngắn hơn.

H2 đã đạt gate development: `--transfer-mode terminal` giữ input 67-D và
baseline mặc định `legacy`; 62 tests pass. Replay paired 76000–76004 có B 3/5,
T 4/5, cứu seed 76003, không regression trong 3 success cũ; cả 5 transition
đạt guards và command bounds. Không dùng force trigger. Seed 76002 còn timeout
retreat, tổng thời gian trên success cũ tăng; chưa có nghiệm thu IID/OOD H5.
Xem [P5_H2_RESULTS.md](P5_H2_RESULTS.md), config
`configs/p5_controller_h2_dev.json` và
`artifacts/p5_h2_final_20261015/{summary,verification}.json`.

**H3 — Calibrated placement force**

Tạo observer placement riêng, có thể đặt ở module mới
`simulation/placement_contact.py`; tái sử dụng thao tác biến đổi wrench của
recovery khi phù hợp. API tối thiểu: `reset_episode()`, `update(robot, dt)` và
output gồm calibration status, residual, support confidence, collision status.

- Xây dữ liệu hiệu chuẩn bàn tay rỗng/mang chai trong free space và các contact
  đáy/thành. Dùng GT để gán nhãn audit, không đưa GT vào observer.
- Biến đổi đúng site frame → world frame; bù bias và tải dự kiến. Hiệu chuẩn
  dấu: contact đáy có thể làm tải mang tại cổ tay giảm, không mặc định residual
  phải tăng dương. Unknown calibration phải trả về unknown và không cấp quyền nhả.
- Filter, debounce/hysteresis và điều kiện vận tốc được chọn trên calibration
  development. Kiểm tra lực ngang/moment để chặn nhầm side-wall collision.
- Observer cập nhật từ robot sensor mỗi tick, kể cả khi visual result stale.
  Geometry chưa đủ tươi thì ngăn chuyển phase; force vẫn có thể yêu cầu stop/hold.
- Đo cả latency theo simulated time và wall time. Tách force update khỏi
  freshness gate chưa làm render/inference đồng bộ thành thời gian thực;
  nếu blocking làm vượt deadline thì xử lý scheduling/async trước khi tuyên bố
  đáp ứng nhanh ở wall clock.

Hoàn tất khi seed 76000 không còn báo support ở sự kiện false positive đã biết,
không có false support trên panel free-space/side-wall development, có phát
hiện contact đáy thật và báo detection latency/missed contacts. Không giải quyết
false positive bằng detector luôn trả false. Reset/bad sensor/calibration
unknown không được dẫn tới release.

H3 đã đạt gate development, xem [P5_H3_RESULTS.md](P5_H3_RESULTS.md).
Observer cục bộ tại `simulation/placement_contact.py` cập nhật độc lập camera
và có raw/filtered agreement, variance/speed guard, side-force/moment veto,
load-restored re-arm, reset và unknown handling. **84 tests pass**.
Panel 76000–76004 × free/bottom/side có **0 false support**, phát hiện bottom
**5/5**, latency **0,28–0,32 s simulation**; positive dùng floor jig nâng 12 cm
từ reset, không thay scene production. Seed 76000/step 230 đã calibrated và
từ chối đúng false claim 8,9258 N. 10 visual action replay có 5.982 tick,
0 false support; paired visual B/S shadow trùng cả 965 action/EE/transition.
Không thay release/hold nên chưa có cải thiện success được gán cho H3.
Artifacts ở `artifacts/p5_h3_final_calibration_20261015/`,
`artifacts/p5_h3_final_shadow_20261015/` và
`artifacts/p5_h3_verification_20261015.json`.

**H4 — Nhả kẹp và rút tay có xác nhận**

Thêm substate `support_or_valid_pose → release → verify_open → retreat_clear`.
Điều kiện pose thả hợp lệ cho phép khoảng thả nhỏ nếu task cần; thiếu contact
không tự động khiến một release theo pose của baseline thành nhả non.

H4 đã nối observer H3 với geometry đủ tươi, hold/correction và
`place_contact_mode=calibrated`. Chỉ bật `--placement-observer` cùng `off` thì
vẫn là shadow. Calibrated mode mới có quyền ghi đè action và xác nhận release.
Không dùng confidence riêng để cấp release; unknown phải chặn force-based
authorization. Audit lại trên scene gốc: sensitivity của H3 được đo bằng jig,
và có conservative collision veto trong một số free-space transient.

- Contact chỉ cấp quyền release khi được observer và geometry guards xác nhận.
  Va thành dẫn đến hold hoặc correction có giới hạn, không mở kẹp.
- Ra lệnh mở kẹp rõ ràng, giữ pose trong lúc mở; xác nhận aperture đo được và
  trạng thái load/vision phù hợp trước khi rút.
- Retreat theo Z tới clearance hợp lệ rồi mới cho phép close drawer. Giữ tay
  phải và drawer grip trong suốt thao tác; không chỉ đổi phase one-hot rồi
  mặc định BC sẽ thực hiện đúng chuỗi mới.
- Timeout mở kẹp, retreat không tiến bộ và mất perception phải có reason riêng;
  không nhảy qua trạng thái xác nhận để tiếp tục.

Hoàn tất khi test mô phỏng chứng minh gripper mở trước retreat, không close
drawer khi tay trái chưa clear, và audit phân biệt false contact với release
ngoài envelope. Scoring success vật lý trong env giữ nguyên.

H4 đã đạt gate development: **112 tests pass**, gồm physics scene gốc với
ideal landmarks để cô lập cơ học và ca joint-stop jam. Paired learned-P5
76000–76004 có **B 3/5, T 4/5, F 4/5, TF 5/5**; F cứu 76002 và giữ 3 success
cũ; TF giữ 4 success của T, cứu thêm 76002. Không release ngoài envelope,
retreat trước aperture mở hoặc close trước clearance trong audit. Không đổi
H0 contract/scene/H3 thresholds. Xem [P5_H4_RESULTS.md](P5_H4_RESULTS.md) và
`artifacts/p5_h4_verification_20261016.json`.

Panel này pin `--startup-hold-steps 5 --observation-age-clock simulation`
để tách controller khỏi cold render/host pauses; B/T tái lập nguyên trajectory
H2. Mode mặc định vẫn `both`; smoke both-clock 76000 success nhưng còn deadline
miss. Mọi 9 placement của F/TF dùng valid-pose release; positive force-release
scene gốc chưa được panel này exercise. Không gọi 5/5 là nghiệm thu ≥80% IID.

**H5 — Development, replay và holdout cho controller**

| Variant | Transfer | Contact/release | Perception/checkpoint |
| :--- | :--- | :--- | :--- |
| B | Legacy 12 mm | Force off, legacy release | Cố định giống nhau |
| T | Terminal servo + guards | Như B | Như B |
| F | Như B | Calibrated observer + verified release/retreat | Như B |
| TF | Như T | Như F | Như B |

1. Development: dùng seed đã biết để tái hiện lỗi; bổ sung ít nhất 20 IID và
   20 Position seed development đã đăng ký, tách khỏi holdout. Chạy B/T/F/TF,
   chọn theo success vật lý, regression và contact audit. Không chọn dựa trên
   giảm timeout riêng lẻ. Tất cả tuning kết thúc tại bước này.
   Pin startup/age-clock chung cho mọi variant; báo tách controller simulation
   và runtime both-clock. Bổ sung targeted contact thật ở scene gốc để kiểm tra
   nhánh `calibrated_support` và latency, không chỉ valid-pose release như H4.
2. Đóng băng candidate. Replay B và candidate trên IID 50000–50099 và
   Position 53000–53019. Kiểm tra toàn bộ 71 success cũ. Nếu sửa code/cấu hình
   sau replay, ghi đây là vòng development tiếp theo và tạo manifest mới.
3. Holdout: ít nhất 100 IID + 20 Position seed mới, chạy paired B/candidate.
   Camera và Lighting thêm 20 seed/profile để kiểm tra ảnh hưởng; không pool
   OOD với IID. Giữ holdout chưa đọc kết quả cho đến khi đóng băng candidate.
4. Sau holdout chỉ báo kết quả. Nếu fail, quay về development và dùng holdout
   mới cho vòng xác nhận sau; không tuning rồi tiếp tục gọi cùng cohort là fresh.

Công thức cohort cũ: `success_new = 71 + rescues_all - regressions_all`.
Để đạt ≥80 cần net ≥9. Với Position cohort cũ, cần >5/20; với holdout mới,
so candidate với baseline mới, không lấy 5/20 làm baseline của một seed set khác.
Ghi rõ số ca trong 26 timeout được cứu; không bỏ 2 approach timeout và perception
failure khỏi bảng tổng. Chuyển từ timeout transfer sang timeout retreat vẫn là fail.

**H6–H7 — Position OOD và visual DAgger**

Đây là vòng training riêng để xử lý lỗi trước transfer, không gộp vào ablation
controller H5. Nếu chỉ nhánh controller đạt IID nhưng Position còn thấp, phần
công việc này vẫn cần thiết để hoàn thành mục tiêu Position của handoff.

- Collector mới rollout bằng landmark P5, observation age và visual phase
  manager thực. Student inputs phải đúng những gì runtime nhận; teacher có
  thể dùng oracle để tạo corrective action labels trong simulation.
- Tập trung approach/descend và những state mới của settle/release/retreat.
  Bao gồm failure và corrective continuation có nhãn hợp lệ, giữ provenance
  teacher action/executed action/phase. Không sao chép expert phase vào input
  rồi gọi đó là visual rollout.
- Tăng đa dạng vị trí chai, camera và noise/occlusion quan sát trong data
  collection; train/val/test tách theo episode và seed.
- Thử relative features ở ablation riêng. Thay condition dimension/schema
  phải tạo config, normalization stats và checkpoint mới; checkpoint 67-D cũ
  không nhận input schema mới tại inference.
- Bàn giao CPU smoke để xác nhận data/model pipeline và notebook Kaggle cho
  training BC v3. Lưu split manifest, source/data hashes, train seed, config,
  stats và checkpoint xuất ra. Ban đầu giữ P5 cố định để cô lập policy training;
  nếu phải retrain P5 thì báo thành vòng ablation riêng.
- H7 so paired BC v2/v3 trên cùng controller đã đóng băng và fresh seeds riêng;
  báo riêng tác động controller và tác động training. Nếu train phủ ±4 cm,
  Position cũ là stress suite trong vùng train. Đặt thêm extrapolation split
  chưa train, trong workspace khả thi, trước khi tiếp tục tuyên bố OOD.

Hoàn tất khi trả lời được lỗi approach/descend giảm bao nhiêu, perception reject
còn bao nhiêu, success Position tăng bao nhiêu và IID có regression hay không.
Không suy ra generalization từ training loss hoặc oracle-only evaluation.

**Kiểm chứng cần triển khai**

Unit/integration tests tập trung vào hành vi có rủi ro: baseline compatibility,
episode reset, force compensation/frame/sign, calibration unknown, side contact,
stale vision, chuyển phase khi robot đang chạy, mở kẹp trước retreat, resume
manifest sai config và tính paired rescue đúng. Thêm simulation tests có contact
thật để tránh test chỉ phản chiếu điều kiện `if` trong implementation.

Các lệnh sau đã tồn tại, dùng để tiếp nhận và kiểm tra phần bridge/runtime:

```bash
cd /home/nguyen-cong/Desktop/omni_vla
.venv/bin/python -m scripts.eval_visual_policy --help
.venv/bin/python -m scripts.evaluate_visual_suite --help
.venv/bin/python -m pytest -q tests/test_visual_bridge.py tests/test_visual_runtime.py
git diff --check
```

Lệnh tái tạo raw-force và tolerance-only hiện có nằm trong tài liệu nghiên cứu.
Lệnh prepare/run manifest v3 đã được kiểm chứng ở
[P5_H1_RESULTS.md](P5_H1_RESULTS.md) và
[P5_H2_RESULTS.md](P5_H2_RESULTS.md). Dùng interface v3 cho B/T/F/TF;
không dùng report từ interface paired cũ để thay thế phép đo có config riêng.

**Bộ bàn giao cuối cùng**

- Code + tests cho config, observer, terminal servo, release/retreat và harness.
- Config B/T/F/TF, immutable manifests, calibration data và dependency record.
- JSON/JSONL mỗi variant/profile; contact audit và danh sách rescued/regressed
  seeds có lý do. Artifact lớn kèm đường dẫn/hash và hướng dẫn khôi phục trên máy mới.
- Báo cáo `docs/P5_CONTROL_RESULTS.md`: IID/OOD success, Wilson CI, paired
  McNemar, failure breakdown, release false positives, phản ứng lực và deadline.
- Với vòng training: dataset provenance, CPU smoke, notebook Kaggle,
  checkpoint/stats/splits và báo cáo BC v2/v3 trên holdout riêng.
- Trạng thái từng task: `todo`, `in_progress`, `done` hoặc `failed_gate`, kèm
  evidence path. Chỉ ghi `done` cho task thực sự có deliverable và kiểm chứng;
  chỉ tuyên bố đạt ba mục tiêu khi số liệu nghiệm thu đáp ứng bảng đầu tài liệu.

**Nội dung có thể chuyển nguyên cho người tiếp nhận**

> H0–H4 đã hoàn tất; đọc `docs/P5_H0_RESULTS.md`, `docs/P5_H1_RESULTS.md`,
> `docs/P5_H2_RESULTS.md`, `docs/P5_H3_RESULTS.md`, `docs/P5_H4_RESULTS.md`,
> sau đó triển khai H5 theo
> `docs/P5_CONTROL_HANDOFF.md`. Harness v3 đã
> có config riêng theo variant. Baseline đã khóa bằng `configs/p5_control_baseline.json`.
> Dùng `docs/P5_CONTROL_RESEARCH.md` cùng `artifacts/p5_control_research/` làm
> bằng chứng đầu vào. Raw Fz 8 N và tolerance-only 25 mm đã có regression, cần
> giữ các guard verified release/retreat và geometry khi nghiệm thu.
> Terminal servo H2 đã có, bật bằng `transfer_mode=terminal`; panel development
> cuối đạt 4/5 vs baseline 3/5, 1 rescue/0 regression, chưa thay thế H5.
> Observer H3 ở `simulation/placement_contact.py`; H4 tại
> `simulation/verified_placement.py` đã mở calibrated release/retreat opt-in.
> Development B/T/F/TF đạt 3/5, 4/5, 4/5, 5/5 với simulation clock chung;
> chưa thay thế H5. Kiểm tra thêm force-positive release trên scene gốc và
> runtime both-clock; không suy realtime hay population ≥80% từ panel này.
> Giữ checkpoint
> cố định cho vòng controller; sau đó H6 → H7 bổ sung visual DAgger để xử lý
> Position trước transfer. Bàn giao code, tests, manifests, báo cáo paired theo
> seed và fresh holdout; không tính hết timeout hay chuyển phase là task success.
