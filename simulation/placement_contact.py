"""Local wrist-load observer. Proprioception only; support is not release permission.

The scene's wrist sensor measures the child/parent interaction, including the
hand and payload. In the upright grasp used here bottom support *reduces* world
Fz. Empty and loaded free-space references absorb tool gravity and local bias.
They are valid only near the calibrated orientation, grip and low dynamics.
No simulator contacts, object positions, teacher actions or vision enter this
observer. Geometry/release verification belongs to the task controller (H4).
"""
from collections import deque

import numpy as np

from models.impedance_controller import rotation_error


PLACEMENT_DEFAULTS = {
    'placement_filter_frames': 5, 'placement_calibration_frames': 12,
    'placement_debounce_steps': 3,
    'placement_calibration_speed_m_s': .16,
    'placement_calibration_acceleration_m_s2': .6,
    'placement_contact_speed_m_s': .02,
    'placement_contact_acceleration_m_s2': .8,
    'placement_min_payload_n': .4, 'placement_max_payload_n': 3.,
    'placement_support_on_fraction': .6, 'placement_support_off_fraction': .35,
    'placement_min_support_n': .5,
    'placement_lateral_limit_n': 1., 'placement_moment_limit_nm': .12,
    'placement_collision_clear_steps': 5,
    'placement_max_orientation_error_rad': .2,
    'placement_max_grip_change_m': .006,
    'placement_max_force_std_n': .4,
}


def validate_placement_options(options):
    for key, default in PLACEMENT_DEFAULTS.items():
        value = options[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        if isinstance(default, int) and int(value) != value:
            raise ValueError(f'{key} must be an integer')
    if options['placement_filter_frames'] % 2 != 1:
        raise ValueError('placement_filter_frames must be odd')
    if not 0 < options['placement_support_off_fraction'] < options['placement_support_on_fraction'] < 1:
        raise ValueError('Placement support hysteresis must satisfy 0 < off < on < 1')
    if options['placement_min_payload_n'] >= options['placement_max_payload_n']:
        raise ValueError('Placement payload interval is empty')


class PlacementContactObserver:
    def __init__(self, options=None):
        self.options = {**PLACEMENT_DEFAULTS, **(options or {})}
        if set(self.options) != set(PLACEMENT_DEFAULTS):
            raise ValueError('Unknown placement observer option')
        validate_placement_options(self.options)
        self.reset_episode()

    def value(self, key):
        return self.options['placement_'+key]

    def reset_episode(self):
        self.filtered = deque(maxlen=int(self.value('filter_frames')))
        self.empty_samples = deque(maxlen=int(self.value('calibration_frames')))
        self.loaded_samples = deque(maxlen=int(self.value('calibration_frames')))
        self.empty = self.empty_rotation = self.loaded = self.loaded_rotation = self.loaded_width = None
        self.previous_velocity = None
        self.previous_stage = None
        self.payload_n = None
        self.streak = 0
        self.collision_clear_streak = 0
        self.support = self.collision_latched = False
        self.result = self._unknown('uncalibrated')

    def _unknown(self, status):
        return {'calibration_status': status, 'support': False, 'support_confidence': None,
                'collision': False, 'hold_requested': status == 'sensor_invalid',
                'collision_clear_streak': 0,
                'wrench_world': None, 'filtered_wrench_world': None,
                'expected_free_wrench_world': None, 'residual_world': None,
                'empty_wrench_world': None, 'payload_n': None, 'load_loss_n': None,
                'raw_load_loss_n': None, 'force_std_n': None, 'force_stable': False,
                'support_streak': 0, 'speed_m_s': None, 'acceleration_m_s2': None}

    def update(self, robot, dt):
        """One update per sensor tick, independent of camera freshness.

        Required fields: wrench[12], wrist_rotations[2,3,3], ee[6],
        ee_velocity[2,3], gripper[4], placement_stage (finite-task phase 0..9).
        Only stages 0/1 with open fingers calibrate empty; 3/4 at EE Z>.69
        calibrate loaded. References freeze at place (5). Invalid/missing
        sensor input clears references and cannot carry a support streak over.
        """
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Placement observer requires positive finite dt')
        try:
            wrench = np.asarray(robot['wrench'], dtype=float)
            rotations = np.asarray(robot['wrist_rotations'], dtype=float)
            ee = np.asarray(robot['ee'], dtype=float)
            velocity = np.asarray(robot['ee_velocity'], dtype=float)
            gripper = np.asarray(robot['gripper'], dtype=float)
            stage = robot['placement_stage']
            if (wrench.shape != (12,) or rotations.shape != (2, 3, 3) or ee.shape != (6,)
                    or velocity.shape != (2, 3) or gripper.shape != (4,)
                    or type(stage) is not int or stage not in range(10)
                    or not np.isfinite(np.r_[wrench, rotations.ravel(), ee, velocity.ravel(), gripper]).all()
                    or np.max(abs(wrench.reshape(2, 6)[:, :3])) > 200
                    or np.max(abs(wrench.reshape(2, 6)[:, 3:])) > 20
                    or not np.allclose(rotations@rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4)
                    or not np.allclose(np.linalg.det(rotations), 1., atol=1e-4)):
                raise ValueError('Invalid placement sensor sample')
        except (KeyError, TypeError, ValueError):
            self.reset_episode()
            self.result = self._unknown('sensor_invalid')
            return self.result.copy()
        world = np.r_[rotations[0]@wrench[:3], rotations[0]@wrench[3:6]]
        speed = float(np.linalg.norm(velocity[0]))
        acceleration = (float(np.linalg.norm(velocity[0]-self.previous_velocity)/dt)
                        if self.previous_velocity is not None else None)
        self.previous_velocity = velocity[0].copy()
        width = float(gripper[:2].mean())
        if stage != self.previous_stage:
            self.filtered.clear()
            self.streak = 0
            self.support = False
            # A side impact must not become support when friction later unloads
            # the wrist. Re-arm on a free-space stage or restored nominal load.
            if stage in (0, 1, 3, 4):
                self.collision_latched = False
                self.collision_clear_streak = 0
            self.previous_stage = stage
        self.filtered.append(world)
        filtered = np.median(self.filtered, axis=0)
        steady = (acceleration is not None and speed <= self.value('calibration_speed_m_s')
                  and acceleration <= self.value('calibration_acceleration_m_s2'))
        if stage in (0, 1) and width >= .032 and steady and world[2] > 1.:
            self.empty_samples.append(world.copy())
            if len(self.empty_samples) == self.empty_samples.maxlen:
                self.empty = np.median(self.empty_samples, axis=0)
                self.empty_rotation = rotations[0].copy()
        if stage in (3, 4) and ee[2] > .69 and .003 < width < .032 and steady and self.empty is not None:
            load = world[2]-self.empty[2]
            if (self.value('min_payload_n') <= load <= self.value('max_payload_n')
                    and np.linalg.norm(rotation_error(self.empty_rotation, rotations[0]))
                        <= self.value('max_orientation_error_rad')
                    and np.linalg.norm(world[:2]-self.empty[:2]) < self.value('lateral_limit_n')
                    and np.linalg.norm(world[3:]-self.empty[3:]) < self.value('moment_limit_nm')):
                self.loaded_samples.append((world.copy(), rotations[0].copy(), width))
                if len(self.loaded_samples) == self.loaded_samples.maxlen:
                    forces = np.stack([sample[0] for sample in self.loaded_samples])
                    # Reject a noisy calibration instead of interpreting motion as payload.
                    span = max(np.linalg.norm(rotation_error(sample[1], rotations[0]))
                               for sample in self.loaded_samples)
                    if np.max(np.std(forces[:, :3], axis=0)) < .2 and span <= self.value('max_orientation_error_rad'):
                        self.loaded = np.median(forces, axis=0)
                        self.loaded_rotation = rotations[0].copy()
                        self.loaded_width = float(np.median([sample[2] for sample in self.loaded_samples]))
                        self.payload_n = float(self.loaded[2]-self.empty[2])
        ready = self.loaded is not None
        orientation_ok = bool(ready and np.linalg.norm(rotation_error(self.loaded_rotation, rotations[0]))
                              <= self.value('max_orientation_error_rad'))
        grip_ok = bool(ready and .003 < width < .032
                       and abs(width-self.loaded_width) <= self.value('max_grip_change_m'))
        status = ('uncalibrated_empty' if self.empty is None else 'uncalibrated_payload' if not ready
                  else 'orientation_out_of_range' if not orientation_ok
                  else 'grasp_changed' if not grip_ok else 'calibrated')
        residual = filtered-self.loaded if ready else None
        raw_residual = world-self.loaded if ready else None
        if stage == 5 and ready:
            lateral = max(np.linalg.norm(raw_residual[:2]), np.linalg.norm(residual[:2]))
            moment = max(np.linalg.norm(raw_residual[3:]), np.linalg.norm(residual[3:]))
            if lateral > self.value('lateral_limit_n') or moment > self.value('moment_limit_nm'):
                self.collision_latched = True
                self.collision_clear_streak = 0
            else:
                # Re-arm only when nominal carrying load has returned. Side
                # friction that keeps unloading the wrist cannot clear a veto.
                restored = (status == 'calibrated' and abs(residual[2]) < self.value('min_support_n')*.5
                            and lateral < self.value('lateral_limit_n')*.5
                            and moment < self.value('moment_limit_nm')*.5)
                self.collision_clear_streak = self.collision_clear_streak+1 if restored else 0
                if self.collision_clear_streak >= self.value('collision_clear_steps'):
                    self.collision_latched = False
        loss = float(-residual[2]) if ready else None
        raw_loss = float(-raw_residual[2]) if ready else None
        force_std = float(np.max(np.std(np.stack(self.filtered)[:, :3], axis=0)))
        force_stable = len(self.filtered) == self.filtered.maxlen and force_std <= self.value('max_force_std_n')
        threshold = (max(self.value('min_support_n'), self.payload_n*self.value(
            'support_off_fraction' if self.support else 'support_on_fraction')) if ready else None)
        slow = (acceleration is not None and speed <= self.value('contact_speed_m_s')
                and acceleration <= self.value('contact_acceleration_m_s2'))
        suspect = (stage == 5 and status == 'calibrated' and slow and not self.collision_latched
                   and force_stable and loss >= threshold and raw_loss >= threshold)
        self.streak = self.streak+1 if suspect else 0
        self.support = self.streak >= self.value('debounce_steps')
        confidence = float(np.clip(loss/self.payload_n, 0., 1.)) if status == 'calibrated' else None
        self.result = {
            'calibration_status': status, 'support': bool(self.support),
            'support_confidence': (confidence if suspect else 0.) if stage == 5 and status == 'calibrated' else None,
            'collision': self.collision_latched,
            'collision_clear_streak': self.collision_clear_streak,
            'hold_requested': bool(stage == 5 and (self.support or self.collision_latched or status != 'calibrated')),
            'wrench_world': world.tolist(), 'filtered_wrench_world': filtered.tolist(),
            'expected_free_wrench_world': self.loaded.tolist() if ready else None,
            'residual_world': residual.tolist() if ready else None,
            'empty_wrench_world': self.empty.tolist() if self.empty is not None else None,
            'payload_n': self.payload_n, 'load_loss_n': loss, 'support_streak': self.streak,
            'raw_load_loss_n': raw_loss, 'force_std_n': force_std, 'force_stable': bool(force_stable),
            'speed_m_s': speed, 'acceleration_m_s2': acceleration,
        }
        return self.result.copy()
