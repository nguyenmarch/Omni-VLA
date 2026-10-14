# Reproduce the CPU environment and restore artifacts

The exact local Python 3.12 CPU environment is pinned in
`requirements-cpu.lock`, including Torch CPU and transitive dependencies.
`pyproject.toml` retains supported ranges for normal installation. Treat a lock
upgrade as a new experiment and record its benchmark impact.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-cpu.lock
.venv/bin/python -m pip install --no-deps --no-build-isolation -e .
.venv/bin/python -m pip check
MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

`.github/workflows/tests.yml` performs those checks on Ubuntu 24.04/Python 3.12
with CPU OSMesa (`libosmesa6`) to avoid depending on a GPU/EGL device. CI tests create small temporary fixtures and do not require
local training artifacts. The workflow is prepared locally; remote GitHub CI
execution requires pushing the branch. The lock records this machine's installed
versions; downloading all pins in a clean networked environment has not been
verified in this session.

Runtime YAML configs are included in the wheel, alongside robot XML/meshes.
A wheel build and a state-only environment reset from an extracted wheel under
`/tmp`, outside the repository, were checked successfully.

## Local artifact handoff

`docs/ARTIFACT_BUNDLE.json` records hashes/sizes for 42 frozen-baseline and P5
artifacts, source hash and the archive checksum. The prepared archive is
`artifacts/bundles/omni_vla_artifacts_20261014.zip` (ignored by Git).
Transfer this archive alongside a checkout of the reviewed source and manifest.
It includes HDF5 source/teacher data, BC/CFM/P5/patch checkpoints and historical
reports needed to restore the recorded baseline. Fresh reliability-suite outputs
are separate from this historical bundle.

```bash
# Create a new bundle after deliberate artifact changes; outputs are exclusive.
.venv/bin/python -m scripts.artifact_bundle pack \
  --archive artifacts/bundles/new_baseline.zip --manifest docs/NEW_BUNDLE.json

# Verify before restoring; take the checksum from the trusted manifest.
.venv/bin/python -m scripts.artifact_bundle verify \
  --archive /path/omni_vla_artifacts_20261014.zip --sha256 CHECKSUM
.venv/bin/python -m scripts.artifact_bundle restore \
  --archive /path/omni_vla_artifacts_20261014.zip --sha256 CHECKSUM \
  --destination /path/to/checkout
.venv/bin/python -m scripts.freeze_baseline
```

Restoration verifies every hash/size before writing, rejects archive traversal,
symlinks escaping the checkout and conflicting destination files, and is
idempotent for identical files. It does not overwrite differing artifacts.
Checksums establish identity, not deterministic contact rollouts on every CPU.
The trusted manifest/source checkout must come through the same reviewed handoff;
a checksum supplied by an unknown archive is not provenance.
