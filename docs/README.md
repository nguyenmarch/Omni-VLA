# Tài liệu Omni‑VLA

Trang này là mục lục tài liệu và trạng thái cập nhật. Khi số liệu khác nhau,
ưu tiên report có manifest/artifact và ngày mới hơn; README gốc là quickstart.

## Bắt đầu

| Tài liệu | Trạng thái | Dùng để |
|---|---|---|
| [README dự án](../README.md) | Hiện hành, cần cập nhật định kỳ | Tổng quan, cài đặt, lệnh chạy |
| [Architecture](./project/ARCHITECTURE.md) | Thiết kế nền tảng | Ranh giới module và contracts |
| [Build plan](./project/BUILD_PLAN.md) | Roadmap; không phải toàn bộ là trạng thái đã đạt | Milestones và acceptance gates |
| [Implementation status](./project/IMPLEMENTATION_STATUS.md) | Snapshot 14/10/2026; lịch sử triển khai | Bằng chứng các pipeline ban đầu |
| [Project map](./project/PROJECT_MAP.md) | Mới, nguồn chuẩn cho cấu trúc repo | Thư mục, dữ liệu, cleanup và refactor checklist |

## Trạng thái dự án hiện tại

| Tài liệu | Trạng thái | Phạm vi |
|---|---|---|
| [P5 control handoff](./p5/P5_CONTROL_HANDOFF.md) | Đang thực hiện; H5 failed gate/incomplete | Kế hoạch controller, replay và holdout |
| [P5 H0](./p5/P5_H0_RESULTS.md) | Hoàn tất | Baseline/contact mode |
| [P5 H1](./p5/P5_H1_RESULTS.md) | Hoàn tất | Paired harness và telemetry |
| [P5 H2](./p5/P5_H2_RESULTS.md) | Hoàn tất development | Terminal transfer |
| [P5 H3](./p5/P5_H3_RESULTS.md) | Hoàn tất development | Calibrated placement observer |
| [P5 H4](./p5/P5_H4_RESULTS.md) | Hoàn tất development | Verified release và retreat |
| [P5 H5](./p5/P5_H5_RESULTS.md) | Development, replay và diagnostic holdout xong; TF failed gate; H5 chưa nghiệm thu | Kết quả paired B/T/F/TF mới nhất |
| [P5 perception](./p5/P5_PERCEPTION.md) | Offline localization gate đạt; system gate mở | RGB‑D student, accuracy và loop |
| [Visual reliability](./operations/VISUAL_RELIABILITY.md) | Chẩn đoán/historical experiments | Async, timing, IID/OOD protocol |
| [Reproducibility](./operations/REPRODUCIBILITY.md) | Hướng dẫn vận hành | CPU lock, CI, bundle artifacts |

## Training và policy

| Tài liệu | Trạng thái | Phạm vi |
|---|---|---|
| [BC corrective round 2](./policies/BC_CORRECTIVE_ROUND_2.md) | Đã chạy | Dataset DAgger và BC v2 |
| [Direct VLA](./policies/DIRECT_VLA.md) | Pilot | Policy trực tiếp RGB/text/history |
| [Direct VLA results](./policies/DIRECT_VLA_KAGGLE_RESULTS.md) | Report kết quả | Candidate Kaggle và giới hạn |
| [CFM Kaggle v2](./policies/CFM_KAGGLE_V2.md) | Đã chạy; chưa vượt BC gate | So sánh Euler 4/6/8 |
| [CFM H1 v2](./policies/CFM_H1_V2.md) | Đã chạy; chưa vượt BC gate | Chunk-1 ablation |
| [BC visual review](./policies/BC_VISUAL_REVIEW.md) | Hướng dẫn đánh giá | Quan sát rollout trực quan |

## Perception, grounding và reliability

| Tài liệu | Trạng thái | Phạm vi |
|---|---|---|
| [Vision-language pilot](./perception/VISION_LANGUAGE_PILOT.md) | Pilot/history | RGB‑D grounding và finite language |
| [Grounding 256 pilot](./perception/GROUNDING_256_PILOT.md) | Historical experiment | Thử độ phân giải 256 |
| [Patch grounding v3](./perception/PATCH_GROUNDING_V3.md) | Baseline cũ, chưa đạt gate | Patch labels và audit |
| [P5 temporal mask](./p5/P5_TEMPORAL_MASK.md) | Historical ablations | Mask và temporal fallback |
| [Advanced control design](./project/ADVANCED_CONTROL_DESIGN.md) | Thiết kế mục tiêu | Roadmap A1–A6 |

## Quy ước bảo trì tài liệu

- **Current**: phản ánh code/config và artifact gần nhất.
- **In progress / failed gate**: còn acceptance work; phải trỏ đến report gần nhất.
- **Historical**: giữ nguyên kết quả và giới hạn của lần chạy cũ; không ghi đè số.
- Mỗi benchmark mới cần plan/manifest, seed list, checkpoint/config hashes,
  paired outcome và limitations. Không đưa số development thành holdout.
- Khi xóa/đổi tên file, tìm và cập nhật mọi liên kết Markdown trước khi chuyển.
- Không lưu checkpoint/dataset/large run output dưới `docs/`; dùng `data/`,
  `weights/`, `artifacts/` và ghi đường dẫn/hash trong report.

## Tài liệu còn thiếu

- [x] H5 regression replay và kết luận TF candidate failed gate.
- [x] Hoàn tất diagnostic holdout TF bất biến; kết quả không đảo failed replay gate.
- [ ] Sửa controller trên development rồi tạo candidate mới; cần holdout mới sau replay gates.
- [ ] Contributor guide: style, dependency update, command/test entrypoints.
- [ ] Data/artifact catalog có owner, provenance, retention và cách tái tạo.
- [ ] API/contracts reference cho observation, action, frames và clock semantics.
