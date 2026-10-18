# Quan sát BC v2 trong MuJoCo

Chạy từ root project trên Dell (terminal desktop, không headless):

```bash
.venv/bin/python -m scripts.eval_closed_loop \
  --checkpoint weights/bc_kaggle_v2/best.pt \
  --seeds 5000 5006 5026 --execute-horizon 1 --gui --speed 1 \
  --output artifacts/bc_v2_visual_review.json
```

`--speed 1` pace mỗi action theo 25 Hz để xem ở tốc độ thực nếu máy đủ nhanh.
`--speed 0.5` xem chậm gấp đôi. Speed chỉ thay đổi thời gian chờ ở GUI, không
thay timestep/action/controller. Headless benchmark vẫn không pace. Terminal
hiện seed và phase khi chuyển bước. `--seeds` ghi đè episodes/seed thông thường.
Đóng viewer dừng cả phiên xem; episode đang dở không tính vào completed trials.

| Seed | Kết quả benchmark trước khi thêm GUI pacing | Cần quan sát |
|---|---|---|
| 5000 | Success | Hai tay approach, grasp chai/handle, mở drawer, transfer/place, thả chai, retreat, đóng drawer |
| 5006 | Timeout approach | Tay nào đứng lệch hoặc dao động; gripper có đóng trước lúc tới mục tiêu không |
| 5026 | Timeout retreat | Chai có được thả ổn định không; tay nào chưa về vị trí retreat để chuyển sang close |

Ghi nhận các điểm sau cho mỗi lượt: grasp có giữ chai bằng contact; chai có
rơi/va vào thành drawer; cách chai nằm sau release; có dao động kéo dài; phase
cuối cùng. Success predicate hiện chấp nhận chai nằm nghiêng trong drawer,
không yêu cầu đứng thẳng. Nếu demo cho thấy việc này, đó là giới hạn task đã
định nghĩa, không phải chứng minh upright placement.

Policy vẫn nhận privileged state và oracle phase manager, không thực thi
expert actions để cứu failures, không bật recovery trong lệnh trên. GUI run
không dùng để kết luận latency benchmark; rendering/pacing làm đổi wall time.
Kết quả benchmark chính thức vẫn nằm ở `bc_kaggle_v2_same.json` và
`bc_kaggle_v2_fresh.json`. Human visual review không thay thế numeric benchmark.
