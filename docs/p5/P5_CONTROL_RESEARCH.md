# P5: nghiên cứu chuyển pha, tiếp xúc lực và Position OOD

Ngày: 2026-10-14. Đây là nghiên cứu và thực nghiệm chẩn đoán; chưa triển khai
controller mới trong lượt này. Các khuyến nghị dưới đây chưa phải kết quả đạt
IID ≥80%. Không thay đổi checkpoint hay logic success của môi trường.

Cập nhật 2026-10-15: H0 đã thêm contact mode tường minh và khôi phục mặc định
baseline; xem [P5_H0_RESULTS.md](./P5_H0_RESULTS.md). Số liệu dưới đây là nghiên
cứu ngày 2026-10-14; lệnh cuối tài liệu đã thêm mode để tái tạo với API hiện tại.

**1. Bằng chứng từ code và dữ liệu hiện tại**

Nguồn cohort cũ: `artifacts/p5_temporal_fallback_suite/baseline/`.
IID đạt 71/100; có 18 timeout transfer, 4 descend, 4 retreat, 2 approach và
1 episode kết thúc vì perception rejection. Như vậy nhóm 26 ca được hỏi là
18 transfer + 4 descend + 4 retreat; tổng timeout thực tế là 28. `descend` ở đây
là hạ để gắp chai. Hạ chai vào hộc là phase 5, tên `place`; cohort này không có
timeout place. Không thể kỳ vọng trigger ở place trực tiếp cứu 4 timeout descend.

Đọc toàn bộ trace thay vì chỉ sai số ở bước cuối cho thấy:

- Cả 18/18 ca transfer timeout IID từng có cả hai EE trong ngưỡng 25 mm.
  Sai số lớn hơn giữa hai tay tại thời điểm tốt nhất nằm trong khoảng 12,2–18,9 mm,
  tức đều từng đạt cả ngưỡng 20 mm. Ngưỡng cũ 12 mm bỏ lỡ cửa sổ này.
  Các cửa sổ dưới 25 mm kéo dài 8–12 mẫu điều khiển; sau đó BC trôi ra xa.
- Cả 2/2 transfer timeout Position cũng từng đạt 20 mm.
- Position đạt 5/20: 7 approach timeout, 5 episode perception rejection,
  1 descend timeout và 2 transfer timeout. Chỉ 7/20 episode tới transfer.
  Nếu chỉ sửa từ transfer trở đi và giữ nguyên diễn biến trước đó, trần của
  cohort này là 7/20, chưa thể cải thiện sâu generalization.
- Position hiện chỉ đổi vị trí chai trong ±4 cm XY; đây không phải bộ thay
  vị trí toàn bộ hộc/robot. Dữ liệu reset IID chỉ jitter ±1,5 cm.

**2. Audit trigger lực và đối chứng bổ sung**

Tái chạy seed 76000 với code hiện tại, dùng contact ground truth chỉ để audit:
trigger phát tại step 230, Fz world = 8,9258 N. Chai chỉ tiếp xúc hai ngón kẹp,
không có tiếp xúc chai–hộc. Median Fz trong transfer là 8,5773 N, đã vượt
ngưỡng tuyệt đối 8 N. Vì vậy trigger này là false positive đối với giả thuyết
"chai đã chạm đáy". Việc bù gravity trong actuator controller không làm wrist
force sensor tự mất tải trọng của bàn tay/chai.

MuJoCo định nghĩa force sensor là lực tương tác child–parent trong hệ tọa độ
site, có quy ước dấu và bao gồm các lực tác động lên hệ. Đây không phải cảm biến
chỉ xuất lực tiếp xúc đáy. [MuJoCo XML reference](https://mujoco.readthedocs.io/en/latest/XMLreference.html#sensor-force).
Ví dụ force control chính thức của Franka cũng hiệu chỉnh bias và gravity;
cách bù của ví dụ torque sensor cần chuyển đổi phù hợp trước khi dùng cho
wrist sensor của repo. [Franka force-control example](https://frankarobotics.github.io/libfranka/0.15.0/force_control_8cpp-example.html).

Chạy thêm hai cấu hình trên đúng 5 seed 76000–76004, giữ cùng BC/P5 checkpoint,
fallback bật, không mask/EMA. Để vô hiệu hóa trigger trong chẩn đoán, đặt ngưỡng
1.000.000 N và kiểm tra số trigger bằng 0; đây là workaround trước khi H0 thêm cờ off.

| Cấu hình | Success | Rescue so với baseline | Regression so với baseline |
| :--- | :--- | :--- | :--- |
| Transfer 12 mm, không trigger lực | 3/5 | — | — |
| Transfer 25 mm, không trigger lực | 2/5 | 1: seed 76003 | 2: seed 76000, 76004 |
| Transfer 25 mm + trigger raw 8 N | 0/5 | 0 | 3: seed 76000, 76001, 76004 |

Chưa thể quy nguyên nhân mọi regression cho nhả non: transfer-only cũng gây
regression. Audit vật lý xác nhận một false contact trigger; cần ghi thêm contact,
gripper và pose lúc release để quy nguyên nhân từng episode. Năm seed này là
tập development, không dùng 3/5 hoặc 2/5 để ước lượng tỷ lệ IID toàn hệ thống.

Artifacts: [research_summary.json](../../artifacts/p5_control_research/research_summary.json),
[contact_audit_76000.json](../../artifacts/p5_control_research/contact_audit_76000.json),
[baseline](../../artifacts/p5_control_research/baseline_76000.json),
[transfer-only](../../artifacts/p5_control_research/transfer_only_76000.json),
[transfer + raw force](../../artifacts/p5_control_research/transfer_raw_contact_76000.json).
Summary chứa hash source liên quan và audit theo từng seed transfer cũ.

**3. Phương pháp ưu tiên: hoàn tất transfer ổn định trước khi xuống**

Không ưu tiên tăng vận tốc: BC đã đi gần đích trong khoảng 50–60 bước, rồi
timeout ở 351 bước. Vấn đề được quan sát là bỏ lỡ chuyển pha và trạng thái đầu
pha sau không ổn định, không chỉ là tốc độ di chuyển thấp.

Đề xuất một nhánh điều khiển kết thúc transfer dựa trên EE và landmark camera:

1. Khi vào lân cận mục tiêu, giảm tốc và dùng Cartesian feedback có giới hạn
   vận tốc/gia tốc để hội tụ; giữ orientation và grip đã ổn định.
2. Chuyển sang place khi đạt vùng XY/Z hợp lệ, tốc độ EE thấp và cả hai tay
   ổn định qua một cửa sổ ngắn. Kiểm tra clearance của chai đối với miệng hộc.
3. Tách nhánh thành ablation với nới tolerance đơn thuần. Một hướng ít thay đổi
   hơn là giữ điều kiện 12 mm, chỉ kích hoạt hỗ trợ khi tiến độ chững lại trong
   lân cận 20–25 mm; không chuyển sớm ngay khi vừa chạm ngưỡng ngoài.
4. Ghi lại sai số, tốc độ và action trước/sau chuyển pha để kiểm tra BC có bị
   đưa vào state mới ngoài tập train hay không.

Các ngưỡng vùng/tốc độ cụ thể phải chọn trên development. Đây là đề xuất kỹ thuật
từ trace hiện có. Nếu bổ sung controller feedback, báo cáo rõ đây là BC kết hợp
controller, không quy toàn bộ cải thiện cho mô hình BC.

**4. Tiếp xúc có bù lực, rồi xác nhận nhả và rút tay**

Thay raw `abs(Fz) >= 8` bằng sai lệch giữa lực đo và lực mang tải dự kiến:
`delta_fz = signed_vertical(measured_world_force - expected_free_space_force)`.
Hiệu chuẩn dấu bằng thí nghiệm tiếp xúc; không mặc định phản lực đáy luôn làm
giá trị raw Fz tăng. Tải chai chuyển sang đáy có thể làm lực mang tải tại cổ tay giảm.

- Hiệu chuẩn bàn tay rỗng và mang chai trong free space, tính đến orientation,
  bias và tải trọng; hạn chế ảnh hưởng quán tính bằng motion profile chậm gần đáy.
  Có thể tái sử dụng ý tưởng unloaded/reference của `WrenchSlipMonitor`, nhưng
  cần detector placement riêng: unloading lúc đặt không phải slip.
- Kết hợp residual force với vận tốc xuống, vị trí tương đối trong hộc, độ tươi
  perception, lực ngang và moment. Va vào thành hộc phải dẫn đến giữ/re-align,
  không tự động nhả chai. Dùng debounce và hysteresis đã kiểm chứng với lực nền.
- Chuyển từ position/impedance motion sang điều khiển tiếp xúc: giữ XY và
  giới hạn lực theo Z. Đây là ứng dụng được đề xuất của
  [hybrid motion-force control, Lynch & Park](https://modernrobotics.northwestern.edu/nu-gm-book-resource/11-6-hybrid-motion-force-control/).
- Xác nhận support hoặc một pose thả hợp lệ trước khi mở kẹp; ra lệnh mở kẹp
  rõ ràng, kiểm tra aperture đo được, rồi rút thẳng theo Z tới clearance hợp lệ.
  Chỉ đổi phase token chưa bảo đảm BC thực sự mở kẹp/rút đúng hướng.
- Cho xử lý lực chạy ở mỗi tick điều khiển và đo độ trễ phản ứng. Hiện update
  manager bị ràng buộc bởi freshness của perception; render/inference đồng bộ
  cũng làm tăng thời gian phản ứng thực.

Ground truth contact chỉ được dùng để chấm false contact và phân tích nguyên
nhân. Baseline có thể thả từ độ cao nhỏ hợp lệ mà chưa chạm đáy; không đánh dấu
mọi release chưa có contact là lỗi. Với nhánh tuyên bố force contact, thiếu tiếp
xúc vật lý tại trigger mới là false positive cần đo riêng.

**5. Position OOD: sửa phân phối quan sát và dữ liệu phục hồi**

Collector hiện dùng `state_condition(obs, expert.phase)`, tức điều kiện hình học
oracle và phase của expert. DAgger cũ do đó chưa chứng minh bao phủ state do
P5 + visual phase manager tạo ra.

Đề xuất visual DAgger: rollout với landmark P5 và phase manager thực, lưu đúng
observation/action timing mà BC gặp, xin corrective labels từ teacher trên
những state đó và fine-tune bằng dữ liệu tổng hợp cũ + mới. Tập trung approach,
descend, terminal transfer, release và retreat, bao gồm các state sau contact.
Teacher có thể dùng oracle trong mô phỏng để tạo nhãn; observation của student
và runtime không nhận oracle. Đây là ứng dụng đề xuất của
[DAgger, Ross et al. 2011](https://proceedings.mlr.press/v15/ross11a.html).

Nếu corrective rollout khó thu thập, thêm nhiễu có kiểm soát vào demonstration
để thu động tác hồi phục, theo
[DART, Laskey et al. 2017](https://arxiv.org/abs/1703.09327).
Thử thêm đặc trưng tương đối như bottle–EE, handle–EE, drawer-goal–EE; giữ
proprioception để mô hình biết cấu hình cánh tay. Cần train checkpoint/schema
mới, không đổi input của checkpoint 67-D cũ tại inference. Việc dùng hệ tọa độ
theo landmark có cơ sở trong
[task-parameterized movement models, Calinon et al.](https://calinon.ch/paper3036.htm),
nhưng hiệu quả trên BC hiện tại vẫn phải đo.

Randomize chai/camera và chất lượng landmark trong data collection; thêm
occlusion và observation age thực đo. Dùng oracle-landmark ablation chỉ để
chẩn đoán phần lỗi thuộc perception hay policy. Nếu train mở rộng tới ±4 cm,
profile Position cũ trở thành stress/regression suite trong vùng train; cần
một bộ extrapolation mới, hợp lệ về workspace, để tiếp tục tuyên bố OOD.

Residual RL là hướng sau nếu feedback + visual DAgger chưa đủ; không cần
chuyển toàn bộ stack sang RL ngay. Ý tưởng học residual bên trên feedback
controller được khảo sát trong
[Johannink et al., Residual RL for Robot Control](https://arxiv.org/abs/1812.03201).

**6. Protocol trả lời ba mục tiêu**

Suite hiện tại gán cùng tolerance và contact threshold cho cả `baseline` và
`candidate`; khác biệt chính là legacy perception. Vì vậy lệnh paired suite
ở lượt trước không cô lập tác động hai thay đổi điều khiển. Cần manifest có
cấu hình controller riêng cho từng variant trước khi chạy benchmark kết luận.

| Variant | Perception | Transfer | Contact/release |
| :--- | :--- | :--- | :--- |
| B | Cố định cùng P5/fallback | Legacy 12 mm | Legacy, force trigger off |
| T | Như B | Terminal feedback + guards | Như B |
| F | Như B | Như B | Calibrated contact + verified release/retreat |
| TF | Như B | Như T | Như F |

Huấn luyện/relative features là vòng riêng sau khi chốt controller. Không đổi
checkpoint và controller đồng thời rồi quy mọi cải thiện cho tolerance.

Trình tự: development ablations → đóng băng cấu hình → replay regression
panel 100 IID seed 50000–50099 và 20 Position seed 53000–53019 → đánh giá trên
seed holdout mới. Các cohort cũ đã được phân tích nên không dùng để tuyên bố
khả năng generalization chưa qua lựa chọn. Baseline replay phải khớp kết quả
tham chiếu; nếu không, tìm drift trước khi tính paired rescue.

- IID: với cohort cũ, `success_new = 71 + rescues_all - regressions_all`.
  Đạt 80/100 cần cải thiện ròng ít nhất 9 episode. Báo riêng rescue trong 26
  timeout được hỏi, 2 approach timeout và 1 perception failure. Chuyển timeout
  từ transfer sang retreat vẫn là thất bại, không tính là rescue.
- Regressions: kiểm tra đầy đủ 71 success cũ, không chỉ chạy các seed thất bại.
  Với mỗi regression ghi thời điểm trigger/release, lực nền/residual, contact,
  aperture, pose/velocity và failure cuối cùng. Đặt mục tiêu development
  zero regression; số đo 0 trên mẫu không chứng minh rủi ro bằng 0 trong dân số.
- Position: báo net paired rescue và tỷ lệ tới từng pha, đặc biệt approach →
  grasp → transfer. Chỉ cứu 2 transfer failure tối đa tăng từ 5/20 lên 7/20
  trên cohort cũ; cải thiện hơn cần xử lý 13 episode hỏng trước transfer.
- Report Wilson 95% CI, paired rescue/regression, exact McNemar và phân bố
  thời gian/sai số từng pha. 80/100 có Wilson CI khoảng 71,1–86,7%; với N=100,
  cần ít nhất 88/100 để cận dưới Wilson 95% vượt 80%. Phân biệt mục tiêu
  observed ≥80% với tuyên bố success population ≥80%.

**7. Lệnh tái tạo ablation development**

Chạy từ repo root; output dùng thư mục mới vì evaluator từ chối overwrite.
Lệnh dưới dùng `off` thay cho ngưỡng rất cao của audit gốc; raw-force cần
`raw_debug` tường minh sau H0. Full replay có kiểm tra hash và đối chiếu seed
nằm trong `scripts.replay_p5_h0`.

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --policy weights/bc_kaggle_v2/best.pt \
  --episodes 5 --seed 76000 --transfer-tolerance .012 \
  --place-contact-mode off \
  --output artifacts/p5_control_research_repeat/baseline.json

MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --policy weights/bc_kaggle_v2/best.pt \
  --episodes 5 --seed 76000 --transfer-tolerance .025 \
  --place-contact-mode off \
  --output artifacts/p5_control_research_repeat/transfer_only.json

MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --policy weights/bc_kaggle_v2/best.pt \
  --episodes 5 --seed 76000 --transfer-tolerance .025 \
  --place-contact-mode raw_debug --place-contact-force-threshold-n 8 \
  --place-contact-debounce-steps 3 \
  --output artifacts/p5_control_research_repeat/transfer_raw_contact.json
```
