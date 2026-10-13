# CFM H=1: chẩn đoán theo phase và thử nghiệm chunk

## Chẩn đoán trước khi train thêm

Dataset v2 giữ nguyên. So sánh first-action trên 2,080 validation states của
4 episodes, 3 noise draws/state cho CFM, fixed seed và noise prefixes chung
cho 4/6/8 Euler; BC deterministic. Dùng action sau denormalize/clip [-1,1].
Đây là action error trên fixed states, không phải success rate.

| Phase | BC xyz MAE | CFM H16 Euler8 xyz MAE |
|---|---|---|
| approach | 0.0103 | 0.0244 |
| descend | 0.0082 | 0.0225 |
| grasp | 0.0199 | 0.0439 |
| place | 0.0039 | 0.0165 |
| retreat | 0.0040 | 0.0163 |

MAE dùng normalized action units, không phải mét. Trong các phase trên,
validation không thấy đảo dấu gripper; CFM vẫn có sai số/biến thiên theo noise.
[Chi tiết validation](../artifacts/cfm_h16_validation_diagnostics.json).

Trace CFM H16 Euler8 trên 6 seeds 5000, 5006, 5011, 5026, 7000, 7006 cho
3 successes/3 failures, và success/step count khớp benchmark đã có. Các seed
được chọn để chẩn đoán, không dùng pilot này để ước lượng population success.
Fail 5000 ở descend: cuối episode left/right distance khoảng 19.1/11.8 cm;
fail 5011/5026 ở approach: khoảng 38.9/23.1 cm. Trong trace này, approach xyz
MAE 0.747, grip MAE 0.686; descend xyz MAE 0.548, grip MAE 0.697. Các state
khác validation và timeout làm failures chiếm nhiều bước, nên không so trực tiếp
hai con số như cùng phân phối. Quan sát phù hợp với policy đi lệch rồi sai
mạnh trên states ngoài quỹ đạo expert; chưa chứng minh root cause là chunk16.
[Chi tiết rollout](../artifacts/cfm_h16_phase_diagnostics.json).

## Thử nghiệm H1

`configs/train_cfm_h1_v2.yaml` chỉ đổi `model.chunk_size: 16 -> 1` so với
`train_cfm_dagger_v2.yaml`. Giữ hidden256, direct_skip, train seed42, 150 epochs,
lr0.0002, batch128, phase sampling/cap100, data/splits và execute horizon1.
Không thêm phase heads, không sửa controller, success predicates hoặc action
schema. Vì shape input/output đổi, parameter count thay đổi theo chunk; vẫn
không phải capacity-matched comparison với BC hidden128/phase heads.
Train from scratch; checkpoint H16 không tương thích để resume H1.

Ba files để dùng trên Kaggle:

1. `artifacts/kaggle_source.zip`: ZIP source mới.
2. `data/drawer_dagger_v2.h5`: dataset cũ, không cần recollect.
3. `notebooks/kaggle_train_cfm_h1_v2.ipynb`: notebook mới chỉ train CFM H1.

Attach đúng một source version, bật GPU + Internet. Chạy tất cả cells, Save
Version với outputs, tải `cfm_h1_v2_outputs.zip` về root project trên Dell.
Notebook dùng output folder mới/session mới để không trộn logs của các runs.

Khi có outputs:

```bash
.venv/bin/python -m scripts.evaluate_cfm_candidate \
  --archive cfm_h1_v2_outputs.zip --expected-chunk-size 1 \
  --destination weights/cfm_h1_kaggle_v2 \
  --output-dir artifacts/cfm_h1_kaggle_v2 \
  --reference-comparison artifacts/cfm_kaggle_v2/comparison.json
```

CLI từ chối chunk sai, kiểm tra hashes/parity, chạy 300 CPU rollouts với cùng
hai bộ seeds và 4/6/8 Euler. Reports riêng: `artifacts/cfm_h1_kaggle_v2/`;
comparison giữ BC v2 và CFM H16 làm references, có delta success vs từng mốc.
BC/H16 weights và reports không bị thay thế. Sampler CPU 2 threads, execute
horizon1, fixed seed/action-step noise, không teacher action fallback/recovery.
State/oracle phase context vẫn có; chưa phải VLA. Same/fresh seed sets đã được
sử dụng trong experiment cũ; không quảng bá thành untouched/OOD test.

Sau khi import H1, chạy thêm fixed-state diagnostics trên cả 3 checkpoints:

```bash
.venv/bin/python -m scripts.diagnose_policy_actions \
  --checkpoints weights/bc_kaggle_v2/best.pt weights/cfm_kaggle_v2/best.pt \
                weights/cfm_h1_kaggle_v2/best.pt \
  --output artifacts/cfm_h1_validation_diagnostics.json
```

Decision: so success, failure phase, first-action error, noise variability và
sampler p50/p95. H1 chưa được chứng minh tốt hơn H16; giữ BC v2 làm policy chính
cho tới khi có benchmark đủ. Không điều chỉnh target/timeout để làm đẹp số.

## Verification

20 tests pass; H1 CPU smoke 3 epochs, export và six rollout smoke xác minh
pipeline với chunk1 và reference H16. Smoke không tính gate/delta như benchmark
50 episodes/set. GPU H1 outputs đã import; full 300-episode evaluation hoàn tất, xem bên dưới.

## Kết quả H1 trên Kaggle và CPU Dell

GPU train đủ 150 epochs; checkpoint chọn epoch 144, validation MSE 0.001867.
Chunk1, dataset hash và config khớp. ONNX/PyTorch Euler parity <4e-7 trên
inputs/noise đã kiểm tra. BC v2 và CFM H16 weights giữ nguyên.

| Policy | Set | Success | Sampler p50/p95 ms | Delta vs BC (pp) |
|---|---|---|---|---|
| BC v2 | same | 46/50 | 0.465/0.839 | — |
| BC v2 | fresh | 42/50 | 0.476/0.899 | — |
| Reference CFM Euler 4 | same | 25/50 | 1.435/2.518 | -42.0 |
| Reference CFM Euler 4 | fresh | 25/50 | 1.474/2.590 | -34.0 |
| Reference CFM Euler 6 | same | 26/50 | 1.973/3.303 | -40.0 |
| Reference CFM Euler 6 | fresh | 23/50 | 1.977/3.316 | -38.0 |
| Reference CFM Euler 8 | same | 30/50 | 2.618/4.245 | -32.0 |
| Reference CFM Euler 8 | fresh | 26/50 | 2.606/4.182 | -32.0 |
| CFM Euler 4 | same | 35/50 | 1.211/1.970 | -22.0 |
| CFM Euler 4 | fresh | 39/50 | 1.284/2.103 | -6.0 |
| CFM Euler 6 | same | 37/50 | 1.693/2.543 | -18.0 |
| CFM Euler 6 | fresh | 32/50 | 1.693/2.513 | -20.0 |
| CFM Euler 8 | same | 30/50 | 2.137/3.193 | -32.0 |
| CFM Euler 8 | fresh | 22/50 | 2.484/3.764 | -40.0 |

Same seeds 5000–5049; fresh seeds 7000–7049. H1 cải thiện so với H16 ở
4/6 Euler; 8 Euler không cải thiện nhất quán (same giữ 60%, fresh giảm 52→44%).
Không setting nào đạt observed 80% trên cả hai sets. H1 4 Euler có tổng success
cao nhất trong các settings đã thử: 74/100, so BC 88/100; chưa đủ thay BC.
Không coi lựa chọn từ benchmark đã nhìn là generalization trên test mới.

Failure phases H1: 4 Euler same/fresh approach 12/7, place 2/3, retreat 1/1;
6 Euler approach 13/10, place 0/6, descend 0/1, retreat 0/1;
8 Euler approach 13/10, place 6/16, descend 0/1, retreat 1/1.
Số lượt lớn nhất ở approach/place, không chỉ là timeout của một phase.

[First-action diagnostics](../artifacts/cfm_h1_validation_diagnostics.json)
đánh giá cả BC/H16/H1 trên 2,080 fixed validation states với 3 noise draws.
Approach xyz MAE: BC 0.01034; H1 4/6/8 = 0.01772/0.01817/0.01688. H1
cải thiện so H16 nhưng còn cao hơn BC. Place xyz MAE H1 4/6/8 =
0.00828/0.00700/0.00678: error giảm trên fixed states nhưng closed-loop
place failures tăng ở 8 Euler. MAE là normalized action units. Điều này cho
thấy validation first-action error không đủ dự đoán robustness; chưa chứng
minh root cause hoặc ưu thế/phụ thuộc chunk với nhiều training seeds.

Giữ BC v2 làm baseline cho vision-language integration. Chưa thay controller,
action schema, success threshold hoặc nối vision/language trong bước này.
[Report đầy đủ](../artifacts/cfm_h1_kaggle_v2/comparison.json).
