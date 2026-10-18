# P5 H5: paired controller development and gate status

Ngày cập nhật: **2026-10-19**. Trạng thái: **development, regression replay và diagnostic holdout đã hoàn tất; TF failed_gate; H5 chưa nghiệm thu**.

H5 bắt đầu bằng 20 seed IID mới và 20 seed Position mới, chạy paired trên B/T/F/TF.
BC v2 và P5 được giữ cố định. Plan dùng startup hold chung 5 action và
simulation observation clock để cô lập controller trong MuJoCo tăng tốc; đây
không phải phép đo runtime both-clock. Seed development lần lượt là 100000–100019
và 100100–100119. Chúng mới, không trùng evaluation/training seed đã đăng ký.

## Kết quả development

| Profile | B | T | F | TF | TF so với B |
| :--- | ---: | ---: | ---: | ---: | :--- |
| IID, n=20 | 11/20 (55%), Wilson 95% 34,2–74,2% | 10/20 (50%) | 10/20 (50%) | **15/20 (75%), 53,1–88,8%** | 8 rescue, 4 regression, net +4 |
| Position, n=20 | **6/20 (30%), Wilson 95% 14,5–51,9%** | 3/20 (15%) | 6/20 (30%) | 5/20 (25%), 11,2–46,9% | 2 rescue, 3 regression, net −1 |

TF cải thiện IID so với B nhưng vẫn dưới observed target 80% trên sample này.
Position giảm một success ròng. TF còn regression trên cả hai profile nên chưa
đạt mục tiêu zero regression, không đủ điều kiện để gọi là candidate đã đóng băng.
Các CI rộng; development seeds đã dùng để so sánh, không phải holdout.

Paired seed TF:

- IID rescue: 100000, 100003, 100004, 100005, 100009, 100010, 100012, 100013.
- IID regression: 100001, 100016, 100018, 100019.
- Position rescue: 100112, 100114.
- Position regression: 100103, 100118, 100119.

Failure breakdown cho thấy Position có 7/20 approach timeout và 2/20 perception
rejection giống baseline, cùng hai descend timeout. Các lỗi trước transfer nằm
ngoài tác động của terminal servo. TF có bốn `visual_place_guard_timeout` ở
Position; F có `visual_placement_calibration_timeout`; TF cũng có một
`visual_placement_collision_timeout` ở IID. Các guard này fail closed, nhưng
vẫn là thất bại task và phải được xử lý/audit trong development tiếp theo.

Trong tất cả variant/profile ở panel này có 0 false contact trigger, 0 release
command ngoài placement envelope và 0 actual opening ngoài envelope. Đây là
audit panel development, không chứng minh an toàn cho mọi trạng thái. Deadline
miss vẫn phổ biến: B IID 1.957/9.739 ticks; TF IID 2.332/11.609 ticks. Kết quả
không chứng minh hard real time.

## Regression replay và holdout

Đã tạo compatibility reference v3 riêng trong
`artifacts/p5_h5_reference_v3_20261018/`. Report/trace historical gốc không bị
sửa; metadata B được bổ sung từ `configs/p5_control_baseline.json` và H0
contract, có lưu source hash. Protocol replay dùng startup hold 0 và age clock
`both`. Harness xác nhận B khớp theo từng seed (success, steps, failure reason)
với reference cũ trên cả hai cohort.

| Profile | B/reference | TF | Rescue | Regression | Net | Wilson 95% TF |
|---|---:|---:|---:|---:|---:|---:|
| IID seeds 50000–50099 | 71/100 | **46/100** | 13 | 38 | **−25** | 36,6–55,7% |
| Position seeds 53000–53019 | 5/20 | 5/20 | 2 | 2 | **0** | 11,2–46,9% |

IID rescue: 50020, 50022, 50023, 50035, 50054, 50061, 50062, 50063, 50071,
50073, 50082, 50086, 50094. IID regression: 50002, 50003, 50004, 50005, 50013,
50014, 50015, 50017, 50018, 50021, 50027, 50031, 50033, 50034, 50037, 50038,
50039, 50041, 50042, 50043, 50045, 50049, 50051, 50053, 50055, 50057, 50059,
50060, 50070, 50072, 50075, 50083, 50087, 50088, 50091, 50093, 50095, 50098.
Position rescue: 53002, 53005. Position regression: 53006, 53018.

TF IID failure breakdown: 41 `visual_placement_calibration_timeout`, 3
`visual_placement_collision_timeout`, 3 `visual_place_guard_timeout`, 4 descend
timeouts, 2 approach timeouts and 1 perception rejection. Position TF có 2
placement calibration timeout, 7 approach timeout, 4 perception rejection và
2 descend timeout. Trong paired IID, 35/38 regressions là ba loại placement
guard; tất cả 13 rescue thuộc timeout transfer/descend/retreat. TF không có
false contact hoặc release/opening ngoài envelope. B có 1 actual opening ngoài
envelope trên IID; đây là baseline audit event, không phải candidate trigger.
McNemar exact hai phía: IID `p=0.00062`, Position `p=1.0`.

Kết luận: candidate TF **failed gate**. IID giảm từ 71% xuống 46%, net regression
−25; Position không cải thiện net; IID target ≥80 và yêu cầu net rescue ≥9 đều
không đạt. Các failure placement guard fail closed, nhưng không được đổi timeout
thành success hoặc nới guard sau khi đã xem cohort replay.

TF đã bị loại khỏi nghiệm thu sau replay nhưng được giữ nguyên code/config để
chạy một **diagnostic paired holdout**, theo yêu cầu tiếp tục H5. Không thay đổi
TF dựa trên holdout; kết quả không thể đảo ngược failed replay gate và không
được xem là xác nhận một candidate được chấp nhận. Seed set đã seal trước khi
chạy: 110000–110099 IID, 111000–111019 Position; protocol `both`, startup hold 0.
Run: `artifacts/p5_h5_holdout_20261018/`.

| Profile | B | TF | Rescue | Regression | Net | McNemar exact p | Wilson 95% TF |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID, n=100 | 63/100 | 59/100 | 22 | 26 | **−4** | 0,6655 | 49,2–68,1% |
| Position, n=20 | 4/20 | 3/20 | 0 | 1 | **−1** | 1,0 | 5,2–36,0% |

Holdout IID rescue: 110006, 110012, 110013, 110016, 110017, 110018, 110023,
110025, 110030, 110031, 110033, 110035, 110038, 110056, 110057, 110059,
110060, 110063, 110067, 110087, 110093, 110096. IID regression: 110000,
110008, 110019, 110026, 110029, 110032, 110045, 110046, 110048, 110049,
110051, 110053, 110058, 110062, 110064, 110065, 110068, 110075, 110076,
110077, 110080, 110082, 110083, 110084, 110094, 110099. Position regression:
111008. Wilson 95% cho B là 53,2–71,8% IID và 8,1–41,6% Position; cả B và TF
đều không đạt observed IID 80% gate.

TF IID có 25 `visual_placement_calibration_timeout` và 5
`visual_place_guard_timeout`; đây là 30/26 regressions không tương ứng vì một
số failure placement thuộc các episode vốn fail baseline. So sánh paired cho
thấy 26 baseline success bị mất, trong đó placement calibration/guard chiếm
phần lớn. TF Position regression seed 111008 kết thúc ở
`visual_placement_calibration_timeout`. Không có false contact, release command
ngoài envelope hay actual opening ngoài envelope TF ở holdout. B có một actual
opening ngoài envelope IID. Không có lỗi audit này không bù lại task failure.

Kết luận holdout: TF không cải thiện paired success tổng thể; nó giải quyết
22/25 baseline transfer/descend/retreat timeout được tính trong nhóm mục tiêu,
nhưng gây 26 regression IID và 1 regression Position. Holdout chỉ là chẩn đoán
sau khi candidate đã trượt replay, không phải xác nhận độc lập cho candidate
hợp lệ. Cohort seed này đã tiêu thụ; không tái sử dụng cho candidate đã chỉnh.
Để tiếp tục H5 cần audit trace placement, xử lý calibration/guard timeout trên
development, tạo candidate mới, rồi chạy replay đạt gate trước khi đăng ký
holdout mới.

## Artifacts và lệnh tái hiện

- Plan và sealed manifest: `artifacts/p5_h5_dev_20261018/plan.json`.
- Paired summary: `artifacts/p5_h5_dev_20261018/summary.json`.
- Per-variant reports/traces: `artifacts/p5_h5_dev_20261018/{B,T,F,TF}/`.
- Regression replay: `artifacts/p5_h5_regression_20261018/` (plan, summary,
  B/TF reports, JSONL traces và completion receipts).
- Normalized historical references with source hashes:
  `artifacts/p5_h5_reference_v3_20261018/`.
- Diagnostic holdout plan/summary/seed set: `artifacts/p5_h5_holdout_20261018/`
  và `configs/p5_h5_holdout_seeds.json` (đã hoàn tất; seed cohort đã tiêu thụ).
- Candidate comparison config: `configs/p5_controller_h5_candidates.json`.
- Seed list: `configs/p5_h5_development_seeds.json`.

Run đã hoàn tất và có completion receipts cho cả hai profile của B/T/F/TF.
Không chạy test suite trong lượt H5 này.

H5 còn việc: (1) audit trace/release và sửa nguyên nhân calibration/placement
timeout trong development, (2) chạy development paired mới và đóng băng candidate
vượt regression criteria, (3) replay IID/Position với manifest mới, (4) chỉ khi
replay đạt mới đăng ký holdout mới. H5 hiện **failed_gate/incomplete**, không
được ghi là nghiệm thu. H6–H7 vẫn chưa bắt đầu.
