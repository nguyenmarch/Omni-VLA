# Grounding 128 vs 256

| Test metric (423 common targets) | 128 | 256 |
|---|---:|---:|
| Mean XYZ cm | 20.44 | 15.40 |
| P95 XYZ cm | 44.99 | 34.04 |
| Mean camera depth error cm | 17.78 | 14.05 |

| Resolution | Setting | Success | Capture + grounding p50/p95 ms |
|---|---|---:|---:|
| 128 | normal | 0/5 | 48.05/57.68 |
| 128 | drop RGB | 0/5 | 46.99/53.15 |
| 128 | swap queries | 0/5 | 49.20/58.30 |
| 128 | open-only | 0/5 | 49.59/54.11 |
| 256 | normal | 0/5 | 49.12/55.83 |
| 256 | drop RGB | 0/5 | n/a |
| 256 | swap queries | 0/5 | 51.23/64.69 |
| 256 | open-only | 0/5 | 54.57/65.10 |

256: 15 approach timeouts; 5 drop-RGB failures stopped before action due to invalid depth. No teacher/recovery fallback.

Archived 128 pilot vs newly trained 256, same seeds, BC and phase manager. Timings are from separate runs, not a controlled back-to-back latency experiment. Resolution training scales label sigma 3 to 6.
