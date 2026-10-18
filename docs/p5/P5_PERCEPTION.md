# P5 — RGB-D perception và finite-language student

## Phạm vi và hợp đồng

P5 ước lượng **tâm 3D** của `bottle_site` và `handle_site` cho BC v2 từ ảnh
head 256×256, depth thật, ma trận nội tại `K`, camera-to-world `T` và câu lệnh.
Runtime không đọc simulator object state, segmentation geom ID, phase hay action
của expert. Các giá trị đó chỉ phục vụ tạo nhãn và chấm điểm. Đây là student
theo scene cố định với hai object và từ vựng đã biết, không phải open-world VLA.

`scripts/prepare_p5_labels.py` replay 12 expert episodes từ
`data/grounding_rgbd_v2_256.h5`, đối chiếu q, target, action, RGB, depth,
`K/T` từng frame trước khi render semantic và amodal mask. Cả 12 episode có
max replay error **0**. File teacher chứa 1.294 frame, split theo episode
8 train / 2 validation / 2 test, seed không trùng. Visibility được định nghĩa
là ít nhất 6 pixel hiện ra và tỷ lệ visible/amodal ≥0,25. Không dùng test để
chọn checkpoint hay hiệu chỉnh hình học.

Student gồm pixel MLP học semantic mask từ RGB, local RGB, measured depth và
điểm world đã unproject; connected component lớn nhất lọc nhiễu. Một MLP cho
mỗi object hồi quy độ lệch từ thống kê point cloud tới tâm object. Text encoder
embedding trung bình và classifier chọn object được huấn luyện trên finite
phrase set. Các câu paraphrase test là câu mới nhưng chỉ dùng **từ đã xuất hiện
trong train**; OOV được từ chối rõ ràng. Bottle có thêm hiệu chỉnh dựa trên
bán kính cylinder 2,5 cm đã biết trong scene: khi trục PCA gần ngang
(`|axis_z| < 0,4`), tâm z lấy từ median visible surface trừ 0,68 bán kính.
Nhánh và hệ số này được chọn từ validation, trước khi chạy test.

Checkpoint `weights/p5_student/best.pt` được chọn ở epoch center 399 theo số
target validation trong 1,5 cm, rồi mean accepted error; segmentation chọn
theo sampled validation IoU. Checkpoint ghi hash source/teacher và danh sách
episode train/validation. **Metric trong checkpoint là trước hiệu chỉnh
geometry**; report dưới đây đo runtime cuối cùng sau hiệu chỉnh. Teacher HDF5,
source HDF5 và weights là artifacts local bị gitignore.

## Kết quả offline

| Split/nhóm | Số target | Coverage | Mean 3D | P95 3D | Trong 1,5 cm |
|---|---:|---:|---:|---:|---:|
| Validation visible | 455 | 100% | 0,37 cm | 0,99 cm | 447/455 |
| Test visible | 423 | 100% | 0,29 cm | 0,83 cm | 418/423 |
| Test occluded | 5 | 100% | 1,35 cm | 2,08 cm | 3/5 |
| Test all | 428 | 100% | 0,31 cm | 0,87 cm | 421/428 |
| Frozen patch v3 test all | 428 | 33,64% | 10,23 cm accepted | 25,21 cm accepted | 3/428 |

**P5 visible mean ≤1,5 cm đạt trên test**; p95 cũng thấp hơn 1,5 cm ở split
này. Coverage được báo riêng, và mean/p95 chỉ tính accepted targets. Năm
occluded targets quá ít để suy rộng; các frame trong cùng episode có tương
quan thời gian, vì vậy 423 target không tương đương 423 thử nghiệm độc lập.
Test cùng scene, camera, object meshes và lighting với train, chỉ khác expert
episode/seed. Chưa đo OOD viewpoint, vật thể mới, texture mới hay calibration
noise. Tâm site là mục tiêu BC; đây không phải sai số grasp contact point.

Trên test, bỏ RGB còn 81/428 target accepted với mean 16,99 cm; bỏ depth hoặc
RGB-D khiến runtime từ chối 428/428 vì không có measured 3D cloud hợp lệ.
Đảo hai text query giữ coverage 100% nhưng mean tăng lên 23,86 cm. Tám
held-out paraphrases được phân loại đúng 8/8, giữ mean 3D 0,31 cm trên các
frame test; đây chỉ là finite-vocabulary paraphrase check, không chứng minh
khả năng hiểu ngôn ngữ mở. Ablation bỏ depth đo cả cơ chế từ chối an toàn,
không tách riêng đóng góp của depth cho network.

[Validation report](../../artifacts/p5_student/val_refined/report.json),
[test report và ablations](../../artifacts/p5_student/test/report.json),
[test error atlas](../../artifacts/p5_student/test/error_atlas.html),
[teacher replay report](../../data/grounding_p5_labels.report.json).

## CPU và closed-loop

CPU Dell, PyTorch CPU 2 threads, EGL head 256: hai query inference khi đọc
frame từ HDF5 p50 **25,65 ms**, p95 **28,66 ms** (không tính HDF5 I/O/render).
30 reset seeds 11000–11029, sau 3 warmup, head capture + inference p50
**66,69 ms**, p95 **76,30 ms**, trung bình **14,87 Hz**; 30/30 cặp landmarks
được chấp nhận. Reset-frame benchmark không tính physics, BC, queue hay ROS2,
vì vậy không phải deadline guarantee. [CPU report](../../artifacts/p5_student/cpu_throughput.json).

Frozen BC v2 dùng hai landmarks P5 và finite task manager, không nhận oracle
object/phase. Pilot seeds 10000–10004: **4/5** full drawer tasks thành công;
trial 10001 timeout ở transfer. Mở rộng trên seeds 12000–12019: **15/20**
thành công; hai timeout retreat, một descend, một transfer và một lần
perception từ chối vì support/text/workspace gate. Trong 1.906 lần định vị
được chấp nhận của loạt này, bottle error mean/p95 là 0,18/0,57 cm, handle
0,67/1,84 cm; các frame đến từ policy rollouts và metric dùng oracle chỉ để
chẩn đoán sau dự đoán. Capture+infer trong rollout p50/p95 70,37/78,94 ms.
Gate 3D đã qua nhưng **system success ≥90% chưa đạt**; cần sửa lỗi
task-manager/BC theo pha và xử lý occlusion/rejection có kiểm soát trước khi
coi pipeline hoàn chỉnh. [Pilot report](../../artifacts/p5_student/closed_loop.json),
[20-seed report](../../artifacts/p5_student/closed_loop_20.json).

## Tái hiện trên workspace có sẵn RGB-D v2 và BC v2

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.prepare_p5_labels
.venv/bin/python -m scripts.train_p5
.venv/bin/python -m scripts.eval_p5 --split val --output artifacts/p5_student/val_refined
.venv/bin/python -m scripts.eval_p5 --split test --output artifacts/p5_student/test
MUJOCO_GL=egl .venv/bin/python -m scripts.benchmark_p5_cpu
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --episodes 5 --seed 10000 \
  --output artifacts/p5_student/closed_loop.json
MUJOCO_GL=egl .venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/p5_student/best.pt --episodes 20 --seed 12000 \
  --output artifacts/p5_student/closed_loop_20.json
```

Các script tạo output mới và từ chối ghi đè đường dẫn đã tồn tại. Để tái tạo
RGB-D v2 từ scene, xem [pilot 256](../perception/GROUNDING_256_PILOT.md); lưu ý teacher
replay yêu cầu source hash và scene hash khớp. Nếu thay scene/mesh/camera,
phải collect lại, tạo lại teacher và train lại.

## Reliability follow-up — 14/10/2026

Xem [visual reliability](../operations/VISUAL_RELIABILITY.md) cho failure traces, observation
age, train-state shift, bounded rejection handling, async snapshot inference và
immutable IID/OOD evaluation plan. Các số P5 ở trên vẫn là frozen historical
results; runtime mới và async phải đọc report riêng. Không dùng thành công
định vị offline thay cho closed-loop system gate.

## Direct VLA successor pilot

[Direct VLA](../policies/DIRECT_VLA.md) là nhánh mới nhận trực tiếp RGB-D, instruction và
lịch sử robot để dự đoán action chunks; không dùng P5 landmarks/phase oracle
làm condition. P5 + BC ở tài liệu này vẫn là frozen baseline. Nhánh mới chỉ
đã CPU smoke, chưa full GPU training hoặc đạt closed-loop gate.
