# Vision-language CPU pilot — 10/10/2026

Đã chạy thêm [pilot 256×256](./GROUNDING_256_PILOT.md): paired test mean XYZ
20.44 → 15.40 cm; xem tài liệu đó để lấy checkpoint, closed-loop results và
lệnh GUI của bản mới. Báo cáo bên dưới giữ nguyên làm baseline 128.

P5 đã có đường chạy từ camera/câu lệnh tới BC v2 và đã train thử trên Dell.
**Chưa đạt gate định vị 3D hoặc closed-loop.** Giữ BC v2 làm baseline state/oracle;
không thay checkpoint của BC bằng mô hình pilot này.

## Luồng thực thi

Head camera RGB + một trong hai truy vấn chai/tay nắm → CNN/cross-attention
heatmap → lấy measured depth tại peak, cộng learned center offset → unproject
với camera calibration → hai landmarks 3D → task manager → frozen BC v2 H1.
Perception chạy mỗi 5 action (5 Hz theo simulation time); BC action 25 Hz.
Grounding học RGB và text; depth chỉ dùng trong bước giải mã, chưa là đầu vào CNN.

Policy/manager nhận robot proprioception và landmarks dự đoán. Không truyền
oracle object, oracle phase, expert action hay success flag vào quyết định action.
Evaluator dùng ground truth để chấm success và ghi lỗi định vị; calibration và
depth lấy từ camera simulator. Không có teacher/recovery fallback.

Hai câu lệnh task được hỗ trợ chính xác:

- `stow the bottle in the drawer then close it`
- `open the drawer`

Đổi câu lệnh task chọn luật của finite-task manager; open-only giữ tay trái và
chỉ mở drawer. Đổi grounding queries hoán đổi `grasp the bottle body` và
`grasp the drawer handle` trong mô hình định vị. Đây là hai đối chứng khác nhau.
Text encoder dùng hashed-word embeddings được train; không phải pretrained VLM.
Mô hình chưa joint-train vision-language-action và chưa hỗ trợ paraphrases.

## Dataset và training

`data/grounding_rgbd_v2.h5`: 12 expert episodes, seeds 8000–8011, tất cả success;
1.294 frames RGB-D 128×128, lấy mỗi 5 action. Split theo episode 8 train / 2 val /
2 test. Dataset này khác schema state/DAgger v2, không gộp trực tiếp vào BC data.

Định vị có 1.685 train targets, 454 validation targets và 423 test targets.
Visibility được lọc bằng depth agreement ≤8 cm tại pixel nhãn; đây là heuristic,
không phải segmentation-certified visibility. 17 train và 5 test targets bị bỏ.
Mẫu bị che khuất hoặc bị loại chưa được đánh giá trong metric offline.

CPU pilot: 10 epochs, 2 Torch threads, batch 32, AdamW lr 0.0005, seed 42.
Best chọn bằng validation mean 3D error, ở epoch 9. Checkpoint riêng:
`weights/grounding_rgbd_v2_cpu_pilot/best.pt`. Không dùng test để chọn checkpoint.

## Sai số trên test tách riêng

| Đầu vào | Mean 3D error | P95 3D error | Mean pixel error | Trong 1,5 cm |
|---|---:|---:|---:|---:|
| RGB + đúng query | 20,44 cm | 44,99 cm | 5,91 px | 2/423 |
| RGB toàn zero | 58,25 cm | 65,80 cm | 43,99 px | 0/423 |
| Hoán đổi queries | 37,40 cm | 66,73 cm | 16,50 px | 0/423 |

Bỏ RGB vẫn giữ depth/calibration, nên không phải bỏ toàn bộ vision.
Ảnh và query có ảnh hưởng tới đầu ra định vị trong bộ test này. Chưa suy ra
robot hiểu ngôn ngữ tổng quát hoặc các đầu vào giúp hoàn thành nhiệm vụ.

Với đầu vào bình thường, chai sai trung bình 12,11 cm, handle 28,98 cm;
camera-depth error 17,78 cm. Approach offline sai trung bình 22,49 cm.
Encoder giảm ảnh xuống grid 8×8 rồi nội suy heatmap; độ phân giải thấp và việc
lấy depth tại một pixel peak là các điểm cần kiểm tra tiếp. Lỗi pixel thấp chưa
bảo đảm peak nằm trên đúng bề mặt vật thể. Đây là hướng chẩn đoán, chưa phải
ablation chứng minh nguyên nhân riêng của từng thành phần.

Report chi tiết: [test ablations](../../artifacts/vision_language_pilot/test_ablations.json),
[training metrics](../../weights/grounding_rgbd_v2_cpu_pilot/metrics.jsonl).

## Closed-loop trên Dell

Pilot dùng cùng seeds 9000–9004 cho bốn setting: bình thường, bỏ RGB, swap
grounding queries và đổi task sang open-only. Full task chấm theo simulator
success; open-only chấm drawer opening >16 cm, không yêu cầu cất chai/đóng drawer.
Success criteria khác nhau nên không so sánh tỷ lệ open-only như cùng nhiệm vụ.

| Setting | Success | Capture + grounding p50/p95 | BC p50/p95 |
|---|---:|---:|---:|
| Bình thường | 0/5 | 48,05/57,68 ms | 0,37/0,77 ms |
| Bỏ RGB | 0/5 | 46,99/53,15 ms | 0,37/0,70 ms |
| Đổi queries | 0/5 | 49,20/58,30 ms | 0,38/0,67 ms |
| Chỉ mở drawer | 0/5 | 49,59/54,11 ms | 0,36/0,61 ms |

Cả 20 lượt đều timeout ở approach sau 351 action. Với setting bình thường,
landmarks sai đáng kể nên condition của frozen BC lệch khỏi oracle training
distribution. Chưa tách được ảnh hưởng riêng của landmark error và phase manager.
Success đều bằng zero nên chưa chứng minh image/text cải thiện task success;
chỉ offline ablation thể hiện sự phụ thuộc của kết quả định vị vào hai đầu vào.

Tests kiểm tra stripping oracle fields, condition 67 chiều dùng perceived
landmarks, ảnh hưởng của finite instruction tới chọn tay, camera/depth decoding,
lọc invalid depth và split leakage. Toàn bộ suite: 24 tests pass.

Xem [bảng kết quả/latency](../../artifacts/vision_language_pilot/comparison.md) và
[từng episode](../../artifacts/vision_language_pilot/closed_loop_comparison.json).
Report từng setting ghi perceived landmark error theo phase và camera refresh.
Latencies gồm capture+grounding và BC riêng; EGL không pace thời gian thực.
5 seeds là diagnostic pilot, chưa là acceptance benchmark 50+50 seeds.

## Chạy lại

Chạy từ root repository; dùng output mới để giữ kết quả cũ:

```bash
.venv/bin/python -m scripts.train_grounding_rgbd \
  --dataset data/grounding_rgbd_v2.h5 --device cpu --epochs 10 \
  --output weights/grounding_rgbd_v2_cpu_repeat

.venv/bin/python -m scripts.eval_grounding_rgbd \
  --checkpoint weights/grounding_rgbd_v2_cpu_pilot/best.pt \
  --dataset data/grounding_rgbd_v2.h5 --split test \
  --output artifacts/grounding_rgbd_test_repeat.json

MUJOCO_GL=egl .venv/bin/python -m scripts.benchmark_visual_bridge \
  --grounding weights/grounding_rgbd_v2_cpu_pilot/best.pt \
  --episodes 5 --seed 9000 --output artifacts/visual_bridge_repeat
```

Xem GUI từ terminal desktop:

```bash
.venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/grounding_rgbd_v2_cpu_pilot/best.pt \
  --episodes 1 --seed 9000 --gui --output artifacts/visual_bridge_gui.json
```

Để lặp vô hạn, thêm `--loop` vào lệnh GUI. Mỗi lượt kết thúc hoặc timeout sẽ
reset với seed tiếp theo. Đóng viewer hoặc nhấn `Ctrl+C` để dừng và lưu report;
lượt đang dở không tính vào success rate. Chế độ này giữ lịch sử giới hạn để
không tăng RAM liên tục. GUI vẫn là scene 3D, chưa có cửa sổ ảnh RGB/depth riêng.

## Bước tiếp theo

### Camera diagnostics đã triển khai

Thêm `--camera-debug` để xem RGB/depth và markers qua trình duyệt, cùng với
GUI MuJoCo đang chạy. Không cần cài OpenCV/Matplotlib:

```bash
.venv/bin/python -m scripts.eval_visual_policy \
  --grounding weights/grounding_rgbd_v2_cpu_pilot/best.pt \
  --gui --loop --seed 9000 --camera-debug \
  --debug-port 8765 --debug-output artifacts/camera_debug \
  --output artifacts/visual_bridge_debug_loop.json
```

Mở **http://127.0.0.1:8765** trên trình duyệt của Dell. Chai màu cam, handle
màu cyan; vòng tròn là tâm vật thể đúng chiếu lên ảnh, dấu X là peak dự đoán.
Nút “Tạm dừng” chỉ dừng cập nhật màn hình camera, không dừng simulation.
Đóng viewer hoặc Ctrl+C ở terminal để dừng simulation/server. Nếu port đang
được dùng, đổi `--debug-port 8766`. Camera input vẫn refresh mỗi 5 action;
monitor hiển thị frame đó, không chụp thêm hoặc thay condition/action.

Bảng hiển thị XYZ/pixel error, camera depth của tâm vật thể, surface depth tại
GT/pred pixel, predicted offset, depth sau offset và dấu hiệu che khuất/lệch nhãn.
GT markers có thể nằm ở phía sau bề mặt vì nhãn là tâm vật thể. Visibility chỉ
là depth-agreement heuristic ≤8cm, chưa có segmentation hay confidence detector.
Màn hình dùng depth grayscale cố định 0.2–3.0m; depth ngoài khoảng này bị clip
trong hình, giá trị số trong report vẫn giữ nguyên.

Monitor lưu `latest.html` tự chứa ảnh để mở offline, JSON thông số, PNG RGB/depth
có overlay và depth float `.npy`. Giữ tối đa 24 snapshots đầu theo episode/phase;
file latest được cập nhật định kỳ và flush khi đóng, nên vòng lặp không tạo
số lượng ảnh tăng vô hạn. `snapshots.json` liệt kê các ảnh đã lưu.

Audit riêng dataset, không cần viewer:

```bash
.venv/bin/python -m scripts.audit_camera_grounding \
  --output artifacts/camera_label_audit_full
```

Mặc định audit hai test episodes: quét visibility của tất cả frames và lưu frame
đầu mỗi phase cùng một frame visibility bị flag nếu còn trong `--max-frames`.
`--episode 000006` chọn một episode; `--max-frames 10` là giới hạn trên mỗi episode.
Đây là tập mẫu chẩn đoán, không dùng để chọn checkpoint hoặc thay test benchmark.

Đã chạy audit **20 frames** của hai test episodes và quét **428 targets** trên
214 frames: 5 handle labels bị depth-disagreement >8cm ở transfer, có khả năng
bị che khuất; ảnh của các trường hợp này đã được lưu để đối chiếu.
Ví dụ seed 8006, step 0: handle GT pixel `[90.49,47.46]`, pred `[88,40]`,
lệch 7.86 px. Depth tại GT 2.005m, tại pred 2.427m; lỗi XYZ 44.97cm.
Chọn sai vị trí ảnh dẫn tới lấy depth khác bề mặt trong ví dụ này.

Max roundtrip GT ~1.94e-7m, prediction reprojection ~1.08e-5px: phép toán
project/unproject nhất quán. Đây **không** phải chứng nhận calibration độc lập;
vẫn cần đối chiếu markers với ảnh và hình học scene.

Artifacts: [audit report](../../artifacts/camera_label_audit_full/audit.json),
[frame approach](../../artifacts/camera_label_audit_full/snapshot_00.html),
[frame transfer bị flag](../../artifacts/camera_label_audit_full/snapshot_05.html),
[RGB overlay](../../artifacts/camera_label_audit_full/snapshot_00_rgb.png).
Headless closed-loop với diagnostics seed 9000 vẫn 0/1 timeout approach, khớp
pilot trước. 27 tests pass; PNG encoder, snapshot flush, depth/geometry audit
và HTTP localhost đã được kiểm tra. JS của dashboard đã qua syntax check.

### Phần còn lại

Ưu tiên cải thiện định vị trước khi train policy mới: kiểm tra projected labels
và pixel/depth tại peak; thử heatmap có độ phân giải cao hơn và dùng depth trong
localization/3D loss. Giữ train/val/test episode disjoint. Sau khi sai số giảm,
thu thêm dữ liệu với camera/appearance/object variation và trạng thái learner,
kiểm tra khả năng phát hiện occlusion/ước lượng không tin cậy. Hiện runtime chỉ
chặn nonfinite/invalid depth, chưa có confidence gate hay xử lý occlusion.

Chỉ mở rộng closed-loop tới hai bộ 50 seeds khi pilot định vị đủ chính xác.
Frozen BC được train với oracle state; cần đánh giá hoặc fine-tune với perceived
landmarks để xử lý distribution shift. Screw fixture, recovery và ROS2 vẫn là
các bước sau; chưa coi pilot này là VLA hoàn chỉnh.
