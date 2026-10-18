# Grounding 256×256 pilot

Experiment thay kích thước render head camera 128 → 256, giữ BC v2 H1 và task
manager hiện tại. Bản 128 và dataset/checkpoint cũ được giữ nguyên. Camera wrist
chưa được thêm vào pipeline này.

## Phương pháp

Thu lại ảnh RGB/depth thật ở 256×256; không resize dataset 128. Cùng expert,
seeds 8000–8011, stride 5, scene và split 8 train / 2 validation / 2 test.
Audit xác nhận 12 episodes / 1.294 frames, tất cả success; targets, phase,
step, action, condition và robot proprioception giống hệt bản 128. Camera
extrinsics giống nhau; focal/principal pixel trong K tăng đúng 2 lần.

Grounding cùng CNN/word embeddings/cross-attention/depth-offset architecture;
encoder có grid 16×16 ở ảnh 256, so với 8×8 ở ảnh 128. Cùng 10 epochs, seed 42,
batch 32, AdamW lr 0.0005, 2 CPU threads. Heatmap Gaussian label sigma 6 px ở
256 giữ cùng góc nhìn với sigma 3 px ở 128. Vì label width cũng được scale,
đây là thử nghiệm render/training theo resolution, không phải chỉ thay một
giá trị img_size khi inference bằng weights cũ.

Best checkpoint chọn bằng mean 3D validation error; test không chọn epoch.
Depth chỉ dùng trong decode, chưa là CNN input. Không thêm depth-aware loss,
tracking hay confidence gate trong thử nghiệm resolution này.

Depth-visibility filter có thể khác giữa hai ảnh render. Báo cáo riêng test
cho từng model và báo cáo paired val/test chỉ lấy những target hợp lệ ở cả
hai bản, cùng episode/frame/target. Pixel error ở hai kích thước khác nhau
không trực tiếp so sánh được; ưu tiên XYZ error tính bằng mét.

Closed-loop dùng seeds 9000–9004, perception mỗi 5 action, frozen BC v2 và
cùng bốn setting của baseline: normal, drop RGB, swap queries, open-only.
Open-only dùng success criterion khác full task. 5 seeds mỗi setting chỉ là
diagnostic pilot, chưa đủ để kết luận đạt acceptance gate.

## Kết quả offline

Best 256 ở epoch 9. So sánh trên cùng 454 validation targets và 423 test
targets hợp lệ ở cả hai resolution:

| Metric | 128 | 256 |
|---|---:|---:|
| Val mean XYZ error | 20.90 cm | 13.15 cm |
| Test mean XYZ error | 20.44 cm | 15.40 cm |
| Test P95 XYZ error | 44.99 cm | 34.04 cm |
| Test bottle mean error | 12.11 cm | 12.20 cm |
| Test handle mean error | 28.98 cm | 18.69 cm |
| Test approach mean error | 22.49 cm | 15.53 cm |
| Test targets trong 1.5 cm | 2/423 | 6/423 |

Mean test giảm khoảng 24.6%, chủ yếu từ handle; bottle chưa cải thiện. Sai số
vẫn lớn hơn nhiều tolerance của approach. Đây là một lần train theo seed 42,
chưa có multi-training-seed uncertainty hoặc OOD benchmark.

Drop-RGB ở 256 tạo 423/423 invalid depth predictions; decoder chọn bề mặt nền
xa. Runtime chặn chúng trước khi điều khiển. Offline report vẫn ghi raw decoded
errors (~140m mean) cùng invalid count; không diễn giải chúng là tọa độ hợp lệ.
Swap queries có mean error 30.74 cm, 0/423 trong 1.5 cm. Các đối chứng này cho
thấy model phụ thuộc ảnh/query trong phạm vi test, không chứng minh hiểu lệnh
tổng quát hay đạt task success.

Report: [paired comparison](../../artifacts/vision_language_256_pilot/paired_grounding.json),
[native 256 test/ablations](../../artifacts/vision_language_256_pilot/test_ablations.json),
[camera audit](../../artifacts/camera_label_audit_256/audit.json),
[ảnh đầu approach](../../artifacts/camera_label_audit_256/snapshot_00.html).

## Kết quả closed-loop

| Setting | 128 success | 256 success |
|---|---:|---:|
| Normal | 0/5 | 0/5 |
| Drop RGB | 0/5 | 0/5 |
| Swap queries | 0/5 | 0/5 |
| Open-only | 0/5 | 0/5 |

256 có 15 approach timeouts và 5 drop-RGB trials dừng trước action vì invalid
depth. Normal capture+grounding p50/p95 49.12/55.83ms; BC 0.38/0.63ms, CPU
2 threads, EGL, không GUI pacing. Latency lấy từ các lần chạy riêng biệt;
không suy ra chênh lệch tuyệt đối giữa 128/256 từ số này.

Đã đối chiếu 40 trials archived/new: seed sets và BC hash khớp. Hash của
checkpoint 128, BC v2 và dataset 128 không thay đổi. 29 tests pass và các
lần collect/train/offline audit/closed-loop thật đều hoàn thành.

**256 cải thiện định vị nhưng chưa cải thiện task success.** Chưa đạt gate
VLA/vision closed-loop. Giữ cả hai checkpoint; không đổi global default hoặc
quảng bá bản 256 là policy hoàn thiện. Bước tiếp theo là kiểm tra manager với
oracle landmarks để tách lỗi phase, đồng thời sửa localization/depth decode
và thêm xử lý occlusion. Chưa cần train lại BC/CFM chỉ dựa vào thử nghiệm này.

[Bảng so sánh tổng hợp](../../artifacts/vision_language_256_pilot/resolution_comparison.md),
[provenance và failure counts](../../artifacts/vision_language_256_pilot/resolution_comparison.json),
[256 episode reports](../../artifacts/vision_language_256_pilot/closed_loop_comparison.json).

## Artifacts và lệnh tái hiện

```bash
MUJOCO_GL=egl .venv/bin/python -m scripts.collect_grounding_rgbd \
  --output data/grounding_rgbd_v2_256.h5 --episodes 12 --seed 8000 \
  --stride 5 --image-size 256

.venv/bin/python -m scripts.train_grounding_rgbd \
  --dataset data/grounding_rgbd_v2_256.h5 \
  --output weights/grounding_rgbd_v2_256_cpu_pilot \
  --epochs 10 --batch-size 32 --heatmap-sigma 6 --device cpu

.venv/bin/python -m scripts.eval_grounding_rgbd \
  --checkpoint weights/grounding_rgbd_v2_256_cpu_pilot/best.pt \
  --dataset data/grounding_rgbd_v2_256.h5 \
  --output artifacts/vision_language_256_pilot/test_ablations.json

.venv/bin/python -m scripts.compare_grounding_resolution

MUJOCO_GL=egl .venv/bin/python -m scripts.benchmark_visual_bridge \
  --grounding weights/grounding_rgbd_v2_256_cpu_pilot/best.pt \
  --episodes 5 --seed 9000 --output artifacts/vision_language_256_pilot

.venv/bin/python -m scripts.audit_camera_grounding \
  --grounding weights/grounding_rgbd_v2_256_cpu_pilot/best.pt \
  --dataset data/grounding_rgbd_v2_256.h5 \
  --output artifacts/camera_label_audit_256
```

Collector/trainer từ chối ghi đè dataset/checkpoint đã tồn tại. Khi chạy lại,
dùng tên output mới; giữ cùng seed/split để so sánh. Dataset/weights/ảnh/report
nằm local và bị gitignore.

Xem simulation và camera dashboard với checkpoint mới:

```bash
.venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/grounding_rgbd_v2_256_cpu_pilot/best.pt \
  --gui --loop --camera-debug --seed 9000 \
  --debug-output artifacts/camera_debug_256 \
  --output artifacts/visual_bridge_256_loop.json
```

Mở http://127.0.0.1:8765 trên Dell. Image size tự lấy từ checkpoint mới;
không cần sửa global env config. Ctrl+C hoặc đóng viewer dừng vòng lặp.
