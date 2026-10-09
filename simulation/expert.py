"""Privileged state-based task manager and scripted demonstrator.

No object teleports or grasp welds: finger friction must move the objects.
The same phase manager supplies task context for the state-policy baseline.
"""
import numpy as np
from models.impedance_controller import rotation_error

PHASES = ('approach', 'descend', 'grasp', 'pull_lift', 'transfer', 'place', 'release', 'retreat', 'close', 'finish')

class ScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.phase = 0
        self.age = 0
        self.bottle = env.site_pos('bottle_site')
        self.handle = env.site_pos('handle_site')
        self.rotation = np.diag([1., -1., -1.])
        self.failed = False
        self.failure_reason = ''

    def goals(self):
        bottle, handle = self.bottle, self.handle
        left_high = np.r_[bottle[:2], .72]
        left_grasp = bottle + [0, 0, .025]
        right_high = handle + [0, 0, .18]
        right_grasp = handle.copy()
        pulled = handle + [-.205, 0, 0]
        center = self.env.data.body('drawer').xpos.copy()
        place = center + [0, 0, .12]
        phase = self.phase
        left, right, lg, rg = left_high, right_high, 1., 1.
        if phase >= 1:
            left, right = left_grasp, right_grasp
        if phase >= 2:
            lg = rg = -1.
        if phase >= 3:
            left, right = left_high, pulled
        if phase >= 4:
            left = np.r_[center[:2], .72]
        if phase >= 5:
            left = place
        if phase >= 6:
            lg = 1.
        if phase >= 7:
            left = np.r_[center[:2], .78]
        if phase >= 8:
            right = handle + [.005, 0, 0]
        if phase >= 9:
            rg = 1.
            right = right_high
        return left, right, lg, rg

    def update(self):
        self.age += 1
        left, right, _, _ = self.goals()
        lp, rp = self.env.site_pos('left_ee'), self.env.site_pos('right_ee')
        reached = np.linalg.norm(lp-left) < .012 and np.linalg.norm(rp-right) < .012
        info = self.env.info()
        if self.phase in [2, 6]:
            ready = self.age >= 20
        elif self.phase == 5:
            ready = np.linalg.norm(lp-left) < .025 and np.linalg.norm(rp-right) < .02
        elif self.phase == 3:
            ready = info['drawer_open'] and lp[2] > .69
        elif self.phase == 8:
            ready = info['drawer_position'] < .03
        elif self.phase == 9:
            ready = info['success']
        else:
            ready = reached
        if ready and self.phase < len(PHASES)-1:
            self.phase += 1
            self.age = 0
        if self.age > 350:
            self.failed = True
            self.failure_reason = f'timeout_{PHASES[self.phase]}'

    def action(self):
        left, right, lg, rg = self.goals()
        result = []
        for side, goal, grip in [('left', left, lg), ('right', right, rg)]:
            position = self.env.site_pos(f'{side}_ee')
            rotation = self.env.data.site(f'{side}_ee').xmat.reshape(3, 3)
            result.extend(np.clip((goal-position)/.015, -1, 1))
            result.extend(np.clip(rotation_error(self.rotation, rotation)/.08, -1, 1))
            result.append(grip)
        return np.array(result, np.float32)


def state_condition(obs, phase):
    """Privileged-state baseline schema v1; no claim of vision-language learning."""
    onehot = np.eye(len(PHASES), dtype=np.float32)[phase]
    return np.concatenate([obs[k] for k in ('q', 'dq', 'gripper', 'wrench', 'ee', 'object')] + [onehot]).astype(np.float32)

COND_DIM = 67
