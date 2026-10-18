"""Bounded terminal Cartesian servo using proprioception and perceived landmarks."""
import numpy as np

from models.impedance_controller import rotation_error

TRANSFER_DEFAULTS = {
    'transfer_mode': 'legacy', 'transfer_entry_radius_m': .03,
    'transfer_max_speed_m_s': .15, 'transfer_max_acceleration_m_s2': .75,
    'transfer_position_gain': .7, 'transfer_velocity_damping_s': .03,
    'transfer_ready_tolerance_m': .012, 'transfer_ready_speed_m_s': .025,
    'transfer_stable_steps': 3, 'transfer_settle_budget_steps': 150,
    'transfer_no_progress_steps': 40, 'transfer_max_source_age_s': .4,
    'transfer_max_angular_speed_rad_s': .5, 'transfer_max_angular_acceleration_rad_s2': 2.,
    'transfer_max_orientation_error_rad': .12, 'transfer_max_goal_jump_m': .03,
}


def limit_norm(value, limit):
    value = np.asarray(value, dtype=float)
    norm = np.linalg.norm(value, axis=-1, keepdims=True)
    return value*np.minimum(1., limit/np.maximum(norm, 1e-12))


def validate_transfer_options(options):
    if options['transfer_mode'] not in ('legacy', 'terminal'):
        raise ValueError('transfer_mode must be legacy or terminal')
    for key, default in TRANSFER_DEFAULTS.items():
        if key == 'transfer_mode':
            continue
        value = options[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        if isinstance(default, int) and value != int(value):
            raise ValueError(f'{key} must be an integer')
    if options['transfer_ready_tolerance_m'] >= options['transfer_entry_radius_m']:
        raise ValueError('Transfer readiness tolerance must be smaller than entry radius')
    if options['transfer_stable_steps'] >= options['transfer_settle_budget_steps']:
        raise ValueError('Transfer settle budget must exceed the stable window')


class TerminalTransfer:
    def __init__(self, options=None):
        self.options = {**TRANSFER_DEFAULTS, **(options or {})}
        if set(self.options) != set(TRANSFER_DEFAULTS):
            raise ValueError('Unknown terminal transfer option')
        validate_transfer_options(self.options)
        self.reset()

    def reset(self):
        self.state = 'bc_transfer'
        self.age = self.stable = self.no_progress = 0
        self.best_error = float('inf')
        self.reference_rotations = self.reference_goals = None
        self.reference_positions = self.position_mask = self.feedforward = None
        self.command_velocity = np.zeros((2, 3))
        self.command_angular_velocity = np.zeros((2, 3))
        self.failure_reason = ''
        self.guards = {}

    def value(self, name):
        return self.options['transfer_'+name]

    def fresh(self, usable, source_age):
        return bool(usable and source_age is not None and np.isfinite(source_age)
                    and 0 <= source_age <= self.value('max_source_age_s'))

    def proprioception(self, robot):
        position = np.asarray(robot['ee'], dtype=float).reshape(2, 3)
        velocity = np.asarray(robot.get('ee_velocity', np.full((2, 3), np.nan)), dtype=float)
        rotations = np.asarray(robot.get('ee_rotations', np.full((2, 3, 3), np.nan)), dtype=float)
        if (velocity.shape != (2, 3) or rotations.shape != (2, 3, 3)
                or not np.isfinite(np.r_[position.ravel(), velocity.ravel(), rotations.ravel()]).all()):
            raise ValueError('Terminal transfer requires finite EE pose, rotations and velocity')
        return position, velocity, rotations

    def command(self, action, robot, goals, dt, usable, source_age):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Terminal transfer requires positive dt')
        if self.value('max_speed_m_s')*dt > .015 or self.value('max_angular_speed_rad_s')*dt > .08:
            raise ValueError('Terminal speed limits exceed the environment action delta at this dt')
        position, velocity, rotations = self.proprioception(robot)
        goals = np.asarray(goals, dtype=float)
        if self.state == 'bc_transfer' and self.fresh(usable, source_age):
            if np.max(np.linalg.norm(goals-position, axis=1)) <= self.value('entry_radius_m'):
                self.state = 'settle'
                self.reference_rotations, self.reference_goals = rotations.copy(), goals.copy()
                self.reference_positions = position.copy()
                self.feedforward = np.asarray(action, dtype=float).reshape(2, 7).copy()
                # Keep already-good components at their loaded grasp pose. A
                # handle-center goal is not an exact gripper-center goal.
                residual = goals-position
                self.position_mask = np.zeros((2, 3), dtype=bool)
                for arm in range(2):
                    while np.linalg.norm(np.where(self.position_mask[arm], 0., residual[arm])) > self.value('ready_tolerance_m')*.8:
                        remaining = np.where(self.position_mask[arm], -1., abs(residual[arm]))
                        self.position_mask[arm, int(np.argmax(remaining))] = True
                self.command_velocity = limit_norm(velocity, self.value('max_speed_m_s'))
        if self.state == 'bc_transfer':
            return np.asarray(action).copy()
        result = np.zeros(14, dtype=np.float32)
        result[[6, 13]] = -1.  # Retain the two grasps; a measured-width hold would unload the fingers.
        consistent_goal = np.max(np.linalg.norm(goals-self.reference_goals, axis=1)) <= self.value('max_goal_jump_m')
        if not self.fresh(usable, source_age) or not consistent_goal or self.failure_reason:
            self.stable = 0
            self.command_velocity[:] = 0.
            self.command_angular_velocity[:] = 0.
            return result  # Emergency hold may decelerate faster than the normal slew limit.
        servo_goals = np.where(self.position_mask, goals, self.reference_positions)
        preload = np.where(self.position_mask, 0., self.feedforward[:, :3]*.015)
        delta = preload+self.value('position_gain')*(servo_goals-position)-self.value('velocity_damping_s')*velocity
        desired = limit_norm(delta/dt, self.value('max_speed_m_s'))
        change = limit_norm(desired-self.command_velocity, self.value('max_acceleration_m_s2')*dt)
        self.command_velocity = limit_norm(self.command_velocity+change, self.value('max_speed_m_s'))
        angular_error = np.stack([rotation_error(target, current)
                                  for target, current in zip(self.reference_rotations, rotations)])
        angular_delta = self.feedforward[:, 3:6]*.08+.5*angular_error
        desired_angular = limit_norm(angular_delta/dt, self.value('max_angular_speed_rad_s'))
        change = limit_norm(desired_angular-self.command_angular_velocity,
                            self.value('max_angular_acceleration_rad_s2')*dt)
        self.command_angular_velocity += change
        for arm in range(2):
            result[7*arm:7*arm+3] = self.command_velocity[arm]*dt/.015
            result[7*arm+3:7*arm+6] = self.command_angular_velocity[arm]*dt/.08
        return np.clip(result, -1., 1.)

    def observe(self, robot, goals, points, opened, dt, usable, source_age):
        if self.state == 'bc_transfer':
            return False
        self.age += 1
        position, velocity, rotations = self.proprioception(robot)
        goals, points = np.asarray(goals, dtype=float), np.asarray(points, dtype=float)
        error = float(np.max(np.linalg.norm(goals-position, axis=1)))
        orientation_error = max(np.linalg.norm(rotation_error(target, current))
                                for target, current in zip(self.reference_rotations, rotations))
        center = points[1]+[.19, 0., -.09]
        bottle = points[0]
        # Known bottle/fixture dimensions; every position here is perceived or proprioceptive.
        clearance = (opened and abs(bottle[0]-center[0]) < .085 and abs(bottle[1]-center[1]) < .05
                     and bottle[2]-.06 >= center[2]+.09+.015 and position[0, 2] >= .69)
        self.guards = {
            'fresh_source': self.fresh(usable, source_age),
            'goal_consistent': bool(np.max(np.linalg.norm(goals-self.reference_goals, axis=1)) <= self.value('max_goal_jump_m')),
            'position_ready': error <= self.value('ready_tolerance_m'),
            'speed_ready': bool(np.max(np.linalg.norm(velocity, axis=1)) <= self.value('ready_speed_m_s')),
            'orientation_ready': bool(orientation_error <= self.value('max_orientation_error_rad')),
            'clearance_ready': bool(clearance),
        }
        self.stable = self.stable+1 if all(self.guards.values()) else 0
        if error < self.best_error-.0005:
            self.best_error, self.no_progress = error, 0
        else:
            self.no_progress += 1
        if self.stable >= self.value('stable_steps'):
            self.state = 'ready_to_place'
            return True
        if self.age >= self.value('settle_budget_steps'):
            self.failure_reason = 'visual_transfer_settle_timeout'
        elif self.no_progress >= self.value('no_progress_steps') and not self.guards['position_ready']:
            self.failure_reason = 'visual_transfer_settle_no_progress'
        return False

    def telemetry(self):
        return {'substate': self.state, 'settle_age': self.age, 'stable_steps': self.stable,
                'no_progress_steps': self.no_progress, 'best_error_m': self.best_error if np.isfinite(self.best_error) else None,
                'command_velocity_m_s': self.command_velocity.tolist(),
                'command_angular_velocity_rad_s': self.command_angular_velocity.tolist(),
                'position_servo_axes': self.position_mask.tolist() if self.position_mask is not None else None,
                'loaded_pose_m': self.reference_positions.tolist() if self.reference_positions is not None else None,
                'guards': self.guards.copy(), 'failure_reason': self.failure_reason}
