# Vision-language CPU pilot

Matched seeds; frozen BC v2; learned RGB localization + measured simulator depth/calibration; no oracle object/phase policy inputs.
Drop-RGB retains depth/calibration. Query swapping differs from changing the finite task instruction. Open-only has a different success criterion.

| Setting | Success | Capture + perception p50/p95 ms | BC p50/p95 ms |
|---|---|---|---|
| normal | 0/5 | 49.12/55.83 | 0.38/0.63 |
| drop RGB | 0/5 | n/a/n/a | n/a/n/a |
| swap grounding queries | 0/5 | 51.23/64.69 | 0.44/0.74 |
| open-only instruction | 0/5 | 54.57/65.10 | 0.48/0.80 |
