# CFM vs BC v2 CPU benchmark

Torch CPU, 2 threads; state/oracle context; no teacher/recovery; execute horizon 1 for BC and CFM; BC chunk 1, CFM chunk 1

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
