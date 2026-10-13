# Reproducible baseline freeze

This Git snapshot contains source code, configs, notebooks, tests, small JSON
reports, and [the hash manifest](REPRO_BASELINE.json). HDF5 datasets, PyTorch
checkpoints, ONNX files, and output ZIPs remain local because they are large.
The manifest identifies their exact bytes; a fresh checkout needs those files
copied to the listed paths before benchmark verification or rerunning inference.

## Verify a checkout

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[train,deploy,test]'
.venv/bin/python -m pytest -q
.venv/bin/python -m pip check
.venv/bin/python -m scripts.freeze_baseline
```

The last command hashes the inputs and reports, checks checkpoint/dataset lineage,
checks episode splits and seed separation, and compares them with the frozen
manifest. It does **not** rerun physics or training. If a new result is accepted
as the next baseline, run `python -m scripts.freeze_baseline --write` and review
the manifest diff with its reports before committing it. A hash match proves
artifact identity, not that a different computer reproduces a stochastic
rollout bit for bit.

## Frozen experiments and seed policy

| Experiment | Training data / checkpoint | Evaluation seeds | Recorded result |
|---|---|---|---|
| Scripted drawer expert | MuJoCo scene and scripted FSM | 3000–3099 | 99/100 |
| BC v2 | `drawer_dagger_v2.h5` / `bc_kaggle_v2/best.pt` | 5000–5049; 7000–7049 | 46/50; 42/50 |
| CFM H16 | Same drawer data / `cfm_kaggle_v2/best.pt` | Same two sets, Euler 4/6/8 | 25–30/50; 23–26/50 |
| CFM H1 | Same drawer data / `cfm_h1_kaggle_v2/best.pt` | Same two sets, Euler 4/6/8 | 30–37/50; 22–39/50 |
| Patch grounding v3 | `grounding_patch_v3_256.h5` / `selected.pt` | Separate validation/test episodes; visual loop 9000–9004 | Test 144/428 accepted, 3/428 within 1.5 cm; loop 0/5 |

The 5000–5049 set was reused from earlier BC evaluations, so treat it as a
development comparison. The 7000–7049 set was fresh for the BC v2 comparison,
but subsequent CFM comparisons have now reused it. Neither set is a pristine
future holdout. All drawer training episode seeds are distinct from these two
evaluation ranges. Grounding uses episode-level train/validation/test splits;
the manifest records their exact episode names and seeds.

The BC/CFM numbers use privileged object state and expert phase context; the
visual loop uses predicted landmarks and a finite-language task manager.
Sampler timings in state-policy reports omit camera capture and perception.
The patch v3 checkpoint is an engineering pilot selected among retained CPU
candidates, not proof of the best model across its ten training epochs.

Reports are under `artifacts/` and referenced by path and SHA-256 in the
manifest. The full seed-by-seed BC reports and CFM Euler reports are part of
this source snapshot. Future tuning should reserve a new seed range before
inspecting its outcomes and report a confidence interval over enough rollouts.
