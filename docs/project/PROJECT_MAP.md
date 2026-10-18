# Project map và cleanup plan

Cập nhật: **2026-10-18**.

## Cấu trúc hiện tại

```text
configs/       YAML/JSON config, seed sets và controller variants
data/          Dataset local, thường bị Git ignore
artifacts/     Immutable run plans, reports, traces và diagnostics
weights/       Checkpoint/model outputs local, bị Git ignore
models/        Policy, perception và impedance model
simulation/    MuJoCo scene, environment, expert, vision/control runtime
scripts/       CLI cho collect, train, export, audit và evaluation
utils/         Dataset/checkpoint/camera/protocol/diagnostic helpers
tests/         Correctness và integration checks
notebooks/     Kaggle/cloud training notebooks
docs/          Architecture, roadmap, handoff và evidence reports
```

Repo đã tách code theo miền chính và dùng module CLI (`python -m scripts...`).
Không chuyển package hoặc đổi import path trong đợt cleanup này: benchmark plans
đóng dấu source hashes, tài liệu dùng nhiều relative links, và đổi path có thể
làm hỏng khả năng tái lập. Một migration package lớn cần bước riêng có import
compatibility và cập nhật manifest rõ ràng.

## Giữ lại và tái tạo

| Path | Phân loại | Chính sách |
|---|---|---|
| `data/`, `weights/`, `artifacts/` | Dữ liệu/bằng chứng | Giữ lại; không dọn tự động |
| `*.zip`, `omni_vla_aloha_weights_3ep.pt` | Handoff/checkpoint cục bộ | Giữ; xác định retention trước khi xóa |
| `.venv/` | Local Python environment | Giữ; tạo lại bằng lock/config khi cần |
| `build/`, `dist/`, `*.egg-info/` | Build output | Có thể xóa, được setuptools tạo lại |
| `.pytest_cache/`, source `__pycache__/` | Cache | Có thể xóa, sinh lại khi chạy Python/pytest |
| `ros2_ws/` | Scaffold rỗng | Đã xóa; tạo lại khi milestone ROS2 bắt đầu |
| `.git/`, `.github/`, source packages | Source/control | Không dọn bằng cleanup script |

Đợt cleanup 2026-10-18 đã xóa `build/`, `omni_vla.egg-info/`,
`.pytest_cache/`, `__pycache__/` thuộc source và `ros2_ws/` rỗng. Không xóa
cache trong `.venv/`; không xóa data, model weights, artifacts, ZIP hoặc checkpoint.

## Checklist tái cấu trúc

- [x] Phân biệt source, local data, generated artifacts và build/cache.
- [x] Dọn build/package/test cache có thể tái tạo.
- [x] Tạo docs index và trạng thái report; liên kết H5 làm nguồn mới nhất.
- [x] Xóa ROS2 workspace rỗng; trì hoãn tới milestone tích hợp.
- [ ] Thêm contributor guide và quy ước code/config.
- [ ] Tạo artifact catalog (run ID → plan, report, seed, hashes, retention).
- [ ] Chuẩn hóa versioning cho observation/action/config/checkpoint schemas.
- [ ] Tách perception, control, policy và evaluation thành package ổn định sau
  khi có compatibility plan; không đổi path benchmark đang dùng giữa chừng.
- [ ] CI bao phủ smoke install/import, lint/type checks và test subsets theo miền.
- [ ] Quyết định cách lưu/archive artifacts lớn sau khi kiểm kê giá trị khoa học
  và khả năng tái tạo từng run.

## Thư mục cấp gốc

Không thêm code/data tùy tiện vào root. File root nên giới hạn ở project entry
points (`README.md`, `pyproject.toml`, lock/config build), license và CI config.
Reports thuộc `docs/` hoặc `artifacts/`; seed/config thuộc `configs/`; scripts
mới phải có CLI module và owner/domain rõ ràng.
