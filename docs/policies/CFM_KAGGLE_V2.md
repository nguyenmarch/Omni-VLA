# CFM trên dataset v2 và so sánh BC v2

BC v2 được giữ nguyên tại `weights/bc_kaggle_v2/best.pt`: 46/50 same seeds,
42/50 fresh seeds. Người dùng đã train CFM v2 trên Kaggle; outputs được import và đánh giá đủ 300 CPU trials. Không dùng
weights CFM CPU cũ làm kết quả của experiment này.

## Files để upload/import

1. `artifacts/kaggle_source.zip`: source mới, chứa notebook/config/evaluator CFM.
2. `data/drawer_dagger_v2.h5`: dataset đã dùng cho BC v2; không thay đổi dữ liệu.
3. `notebooks/kaggle_train_cfm_v2.ipynb`: import thành notebook mới trên Kaggle.

Attach một source version và dataset v2, bật GPU + Internet. Dataset validator
phải ra 57 episodes, 30,902 steps, split 49/4/4. Chạy lần lượt setup, validate,
train, export và package. Notebook chỉ train CFM, không train lại BC. Dùng
session mới/output path mới; nếu cell train báo output đã tồn tại, không chạy
lại vào cùng folder để tránh trộn metrics của hai lượt. Save Version với outputs,
tải `cfm_dagger_v2_outputs.zip` về root project trên Dell.

## Experiment

| Thành phần | BC v2 (frozen) | CFM v2 |
|---|---|---|
| Dataset / normalization | Dataset v2; train-only stats | Cùng dataset và train-only stats |
| State / action | 67 / 14 | 67 / 14 |
| Phase sampling | Balanced, min_count 100 | Balanced, min_count 100 |
| Train seed / epochs | 42 / 150 | 42 / 150 |
| Hidden / action chunk | 128 / 1 | 256 / 16 |
| Execute horizon | 1 | 1 |
| Sampler | Direct BC prediction | Euler 4 / 6 / 8 |

Config: `configs/train_cfm_dagger_v2.yaml`. CFM dùng independent Gaussian/data
pairing, có optional direct noise/condition skip; không phải OT-CFM. Validation
chọn checkpoint bằng sampled chunk MSE với 6 Euler steps và fixed noise; không
chọn bằng 300 test rollouts. Loss BC và CFM không tương đương mục tiêu train.
So sánh này kiểm tra hai cấu hình deploy; khác model size/chunk size nên không
quy mọi khác biệt cho phương pháp CFM. Nếu cần ablation riêng, phải match
capacity và chunk size sau.

## Đánh giá khi nhận ZIP về Dell

```bash
.venv/bin/python -m scripts.evaluate_cfm_candidate \
  --archive cfm_dagger_v2_outputs.zip
```

Lệnh thực hiện:

- Validate checkpoint CFM/dataset hash, metadata; import vào `weights/cfm_kaggle_v2/`.
- Kiểm tra ONNX/PyTorch Euler parity trên cùng inputs/noise ở 4/6/8 steps.
- Với mỗi setting: 50 seeds 5000–5049 và 50 seeds 7000–7049: tổng 300 episodes.
- Torch CPU, 2 threads, execute horizon 1; không GUI, không recovery hoặc expert
  action fallback. Privileged state và oracle phase context vẫn tồn tại.
- CFM noise lấy từ `numpy_default_rng(episode_seed + action_step)`, giúp matching
  seed/action-step có cùng starting noise giữa settings và Torch/ONNX. Trajectories
  có thể khác nhau sau action đầu; fixed noise không ép outcomes giống nhau.
- Đo success, Wilson CI, failure phase, sampler latency p50/p95 và real-time factor.
  Latency này là state-policy sampler, không phải toàn bộ VLA/controller pipeline.

Outputs trong `artifacts/cfm_kaggle_v2/`: sáu `euler_{4,6,8}_{same,fresh}.json`,
`comparison.json`, `comparison.md`. Report so với BC v2 cùng seeds; BC files
không bị thay thế. CLI có thể chạy lại với cùng archive để tiếp tục sau gián đoạn:
chỉ tái sử dụng completed reports khớp checkpoint và settings, không overwrite
reports khác. Tốc độ sampler BC tham chiếu từ benchmark trước trên cùng máy;
không coi đây là benchmark hardware đồng thời kiểm soát mọi background load.

Observed target là 80% trên cả hai sets cho từng Euler setting. Nếu không đạt
hoặc kém BC, giữ BC v2 làm baseline; phân tích phase failures trước khi tăng
training hoặc tuyên bố CFM tốt hơn. Hai bộ seeds đã công bố và đánh giá trên BC;
chưa chứng minh OOD, vision-language hoặc nhiều training seeds.

## Kiểm tra local

CFM 3-epoch CPU smoke trên dataset v2 chỉ xác minh train/export/import/eval.
Six rollout smoke (1 episode per set/setting) không phải benchmark 300 episodes;
CLI ghi `benchmark_complete=false` và không tính gate hoặc delta vs BC cho smoke.
GPU checkpoint đã có ở `weights/cfm_kaggle_v2/best.pt`; benchmark đầy đủ đã hoàn tất.

## Kết quả GPU và benchmark CPU

Checkpoint tốt nhất ở epoch 133/150, validation sampled chunk MSE 0.021087.
Dataset/config khớp experiment. [Import metadata](../../artifacts/cfm_kaggle_v2_import.json).
ONNX/PyTorch Euler parity max abs error 1.19e-06 trên inputs/noise đã kiểm tra.

| Policy | Set | Success | Sampler p50/p95 ms | Delta vs BC (pp) |
|---|---|---|---|---|
| BC v2 | same | 46/50 | 0.465/0.839 | — |
| BC v2 | fresh | 42/50 | 0.476/0.899 | — |
| CFM Euler 4 | same | 25/50 | 1.435/2.518 | -42.0 |
| CFM Euler 4 | fresh | 25/50 | 1.474/2.590 | -34.0 |
| CFM Euler 6 | same | 26/50 | 1.973/3.303 | -40.0 |
| CFM Euler 6 | fresh | 23/50 | 1.977/3.316 | -38.0 |
| CFM Euler 8 | same | 30/50 | 2.618/4.245 | -32.0 |
| CFM Euler 8 | fresh | 26/50 | 2.606/4.182 | -32.0 |

Seed same = 5000–5049; fresh = 7000–7049. Cả 4/6/8 Euler đều chưa đạt
observed target 80% trên cả hai sets. 8 Euler tốt nhất của CFM hiện tại (60%/52%),
nhưng còn thấp hơn BC v2 (92%/84%); tăng số bước không khắc phục hoàn toàn lỗi.
Latency sampler đủ nhịp action 25 Hz trên máy này; không chứng minh end-to-end
VLA latency hoặc hard real-time. Failures vẫn tập trung approach; xem từng JSON.

Giữ BC v2 làm policy chính. Đây là kết quả một checkpoint CFM chunk16/hidden256
vs BC chunk1/hidden128, không kết luận CFM nói chung kém BC. Nếu tiếp tục tuning,
ưu tiên ablation CFM H=1 và kiểm tra first-action error theo phase để tách ảnh
hưởng chunk/policy khỏi sampler, rồi dùng protocol seed giữ nguyên. Chưa thực
hiện ablation này. Reports đầy đủ: [comparison](../../artifacts/cfm_kaggle_v2/comparison.json).
