"""Seeded evaluation perturbations; no changes to the frozen scene assets."""
import mujoco
import numpy as np


class DomainRandomizer:
    def __init__(self, env):
        self.env = env
        self.original = {key: getattr(env.model, key).copy() for key in
                         ('cam_pos', 'cam_quat', 'light_diffuse', 'light_ambient', 'geom_friction', 'geom_rgba')}

    def apply(self, seed, profile='iid'):
        if profile not in ('iid', 'camera', 'lighting', 'friction', 'position', 'combined'):
            raise ValueError('Unknown evaluation profile')
        env = self.env
        for key, value in self.original.items():
            getattr(env.model, key)[:] = value
        rng = np.random.default_rng(seed + 982451653)
        params = {'profile': profile, 'seed': seed}
        if profile in ('camera', 'combined'):
            delta = rng.uniform(-.02, .02, 3)
            env.model.cam_pos[env.model.camera('head_cam').id] += delta
            params['head_translation_m'] = delta.tolist()
        if profile in ('lighting', 'combined'):
            scale = float(rng.uniform(.65, 1.35))
            env.model.light_diffuse[:] = np.clip(self.original['light_diffuse']*scale, 0, 1)
            env.model.light_ambient[:] = np.clip(self.original['light_ambient']*scale, 0, 1)
            params['light_scale'] = scale
        if profile in ('friction', 'combined'):
            scale = float(rng.uniform(.7, 1.3))
            env.model.geom_friction[:, 0] *= scale
            params['friction_scale'] = scale
        if profile in ('position', 'combined'):
            # Replace the usual +/-1.5 cm reset jitter, never teleport during a task.
            delta = rng.uniform(-.04, .04, 2)
            env.data.qpos[env.bottle_adr:env.bottle_adr+2] = [.45, -.18] + delta
            params['bottle_xy_jitter_m'] = delta.tolist()
        mujoco.mj_forward(env.model, env.data)
        return params
