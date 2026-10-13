# CFM vs BC v2 CPU benchmark

Torch CPU, 2 threads; state/oracle context; no teacher/recovery; execute horizon 1 for BC and CFM; BC chunk 1, CFM chunk 16

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
