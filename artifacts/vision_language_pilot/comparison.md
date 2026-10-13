# Vision-language CPU pilot

Matched seeds; frozen BC v2; learned RGB localization + measured simulator depth/calibration; no oracle object/phase policy inputs.
Drop-RGB retains depth/calibration. Query swapping differs from changing the finite task instruction. Open-only has a different success criterion.

| Setting | Success | Capture + perception p50/p95 ms | BC p50/p95 ms |
|---|---|---|---|
| normal | 0/5 | 48.05/57.68 | 0.37/0.77 |
| drop RGB | 0/5 | 46.99/53.15 | 0.37/0.70 |
| swap grounding queries | 0/5 | 49.20/58.30 | 0.38/0.67 |
| open-only instruction | 0/5 | 49.59/54.11 | 0.36/0.61 |
