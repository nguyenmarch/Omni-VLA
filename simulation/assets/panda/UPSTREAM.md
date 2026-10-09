# Panda collision assets

Source: https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda

Downloaded 2026-10-09. `upstream.xml` preserves the original model; `collision.xml`
is modified to omit decorative visual meshes, keeping kinematics, inertias,
collision meshes and fingertips. `simulation/scene.py` prefixes two instances,
adds end-effector/wrist sites, and replaces arm position actuators with bounded
joint torque motors. Arm collisions remain active and visible in the runtime.

The upstream asset license is included in `LICENSE` (Apache-2.0). This directory
contains upstream assets and modifications; the license applies to those assets.
