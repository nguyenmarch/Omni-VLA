"""Observable release/open verification and bounded vertical retreat (H4).

Only perceived bottle/handle landmarks, robot proprioception, BC preload and
H3 sensor evidence enter this FSM. Known bottle/fixture dimensions are local
scene assumptions. Simulator contact/success/object state must stay in scoring.
"""
import numpy as np

from models.impedance_controller import rotation_error
from simulation.transfer import limit_norm


RELEASE_DEFAULTS = {
    'release_max_source_age_s': .24, 'release_max_drop_m': .035,
    'release_stable_steps': 3, 'release_ready_speed_m_s': .025,
    'release_open_threshold_m': .032, 'release_open_budget_steps': 60,
    'release_verify_budget_steps': 50, 'release_geometry_budget_steps': 25,
    'release_collision_hold_steps': 10, 'release_correction_step_m': .008,
    'release_max_corrections': 2, 'release_correction_budget_steps': 50,
    'release_retreat_height_m': .78, 'release_retreat_tolerance_m': .012,
    'release_retreat_budget_steps': 200, 'release_no_progress_steps': 50,
    'release_max_speed_m_s': .15, 'release_max_acceleration_m_s2': .75,
    'release_max_angular_speed_rad_s': .5, 'release_max_angular_acceleration_rad_s2': 2.,
    'release_max_right_drift_m': .02, 'release_max_goal_jump_m': .03,
    'release_collision_rearm_steps':5,'release_collision_load_tolerance_n':.25,
}


def validate_release_options(options):
    for key, default in RELEASE_DEFAULTS.items():
        value = options[key]
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive')
        if isinstance(default, int) and int(value) != value:
            raise ValueError(f'{key} must be an integer')
    if not .032 <= options['release_open_threshold_m'] < .04:
        raise ValueError('Release open threshold must be in [32,40) mm')
    if not .72 <= options['release_retreat_height_m'] <= .85:
        raise ValueError('Release retreat height outside the local safe envelope')
    if options['release_retreat_tolerance_m'] > .02 or options['release_max_drop_m'] > .04:
        raise ValueError('Release clearance/drop tolerances exceed scene bounds')


class VerifiedPlacement:
    def __init__(self, options=None):
        self.options = {**RELEASE_DEFAULTS, **(options or {})}
        if set(self.options) != set(RELEASE_DEFAULTS):
            raise ValueError('Unknown verified placement option')
        validate_release_options(self.options)
        self.reset()

    def value(self, key):
        return self.options['release_'+key]

    def reset(self):
        self.state = 'support_or_valid_pose'
        self.elapsed = self.state_age = self.stale_age = self.stable = self.open_streak = 0
        self.corrections = self.no_progress = self.unknown_age = 0
        self.best_error = float('inf')
        self.anchor = self.rotations = self.preload = self.center = None
        self.correction_goal = None
        self.last_bc = np.zeros((2, 7))
        self.velocity = np.zeros((2, 3))
        self.angular_velocity = np.zeros((2, 3))
        self.open_since = None
        self.open_commanded = self.open_verified = False
        self.authorization = self.failure_reason = ''
        self.guards = {}
        self.collision_rearmed=False
        self.rearm_streak=self.rearms=0

    def fresh(self, usable, age):
        return bool(usable and age is not None and np.isfinite(age)
                    and 0 <= age <= self.value('max_source_age_s')+1e-9)

    def robot_state(self, robot):
        position = np.asarray(robot['ee'], dtype=float).reshape(2, 3)
        velocity = np.asarray(robot.get('ee_velocity'), dtype=float)
        rotations = np.asarray(robot.get('ee_rotations'), dtype=float)
        widths = np.asarray(robot['gripper'], dtype=float).reshape(2, 2)
        if (velocity.shape != (2, 3) or rotations.shape != (2, 3, 3)
                or not np.isfinite(np.r_[position.ravel(), velocity.ravel(), rotations.ravel(), widths.ravel()]).all()):
            raise ValueError('Verified placement requires finite robot pose, rotations, velocity and aperture')
        return position, velocity, rotations, widths

    def geometry(self, points, position, opening, usable, age):
        points = np.asarray(points, dtype=float)
        if points.shape != (2, 3) or not np.isfinite(points).all():
            return False, False, None
        center = points[1]+[.19, 0., -.09]
        drop = float(points[0, 2]-.06-(center[2]+.012))
        footprint = abs(points[0, 0]-center[0]) < .085 and abs(points[0, 1]-center[1]) < .05
        consistent = self.center is None or np.linalg.norm(center-self.center) <= self.value('max_goal_jump_m')
        valid = bool(self.fresh(usable, age) and consistent and opening > .16 and footprint
                     and -.008 <= drop <= self.value('max_drop_m'))
        self.guards = {'fresh_geometry': self.fresh(usable, age), 'goal_consistent': bool(consistent),
                       'drawer_open': bool(opening > .16), 'footprint_inside': bool(footprint),
                       'drop_m': drop, 'valid_pose': valid}
        return valid, bool(valid and drop <= .01), center

    def needs_policy(self, evidence):
        return (self.state == 'support_or_valid_pose' and not self.failure_reason
                and evidence.get('calibration_status') == 'calibrated'
                and not self.collision_active(evidence) and not evidence.get('support'))

    def collision_components(self,evidence):
        if evidence.get('wrench_world') is None or evidence.get('expected_free_wrench_world') is None:
            return None
        raw=np.asarray(evidence['wrench_world'])-np.asarray(evidence['expected_free_wrench_world'])
        filtered=np.asarray(evidence['residual_world'])
        return (max(np.linalg.norm(raw[:2]),np.linalg.norm(filtered[:2])),
                max(np.linalg.norm(raw[3:]),np.linalg.norm(filtered[3:])))

    def collision_active(self,evidence):
        if not evidence.get('collision'):return False
        components=self.collision_components(evidence)
        current=(components is None or components[0]>evidence.get('lateral_limit_n',1.)
                 or components[1]>evidence.get('moment_limit_nm',.12))
        return bool(not self.collision_rearmed or current)

    def capture(self, position, velocity, rotations, center):
        self.anchor, self.rotations, self.center = position.copy(), rotations.copy(), center.copy()
        self.preload = self.last_bc.copy()
        self.velocity = limit_norm(velocity, self.value('max_speed_m_s'))
        self.angular_velocity[:] = 0.

    def switch(self, state):
        self.state = state
        self.state_age = self.stable = 0

    def servo(self, target, robot, dt, left_open, evidence):
        position, velocity, rotations, widths = self.robot_state(robot)
        desired_delta = .7*(target-position)-.03*velocity
        # The right handle is a loaded grasp, not its landmark center. Preserve
        # the captured BC preload; the hand pose itself remains the reference.
        desired_delta[1] += self.preload[1, :3]*.015
        if not left_open and evidence.get('payload_n') is not None:
            desired_delta[0, 2] += float(evidence['payload_n'])/700.
        desired = limit_norm(desired_delta/dt, self.value('max_speed_m_s'))
        self.velocity = limit_norm(self.velocity+limit_norm(desired-self.velocity,
                                   self.value('max_acceleration_m_s2')*dt), self.value('max_speed_m_s'))
        rotation_delta = .5*np.stack([rotation_error(goal, current)
                                     for goal, current in zip(self.rotations, rotations)])
        rotation_delta[1] += self.preload[1, 3:6]*.08
        angular = limit_norm(rotation_delta/dt, self.value('max_angular_speed_rad_s'))
        self.angular_velocity += limit_norm(angular-self.angular_velocity,
                                             self.value('max_angular_acceleration_rad_s2')*dt)
        action = np.zeros((2, 7), dtype=np.float32)
        action[:, :3] = self.velocity*dt/.015
        action[:, 3:6] = self.angular_velocity*dt/.08
        action[:, 6] = [1. if left_open else -1., -1.]
        return np.clip(action.ravel(), -1., 1.)

    def close_allowed(self, robot):
        position, _, _, widths = self.robot_state(robot)
        return bool(self.open_verified and self.state=='retreat_verified'
                    and position[0, 2] >= self.value('retreat_height_m')-self.value('retreat_tolerance_m')
                    and np.all(widths[0] > self.value('open_threshold_m')))

    def command(self, bc, robot, phase, points, opening, dt, usable, source_age, evidence):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Verified placement requires positive dt')
        if self.value('max_speed_m_s')*dt > .015 or self.value('max_angular_speed_rad_s')*dt > .08:
            raise ValueError('Release speed exceeds environment action delta')
        position, velocity, rotations, widths = self.robot_state(robot)
        valid, _, center = self.geometry(points, position, opening, usable, source_age)
        bc = np.asarray(bc, dtype=float).reshape(2, 7)
        if phase == 5 and self.needs_policy(evidence) and self.fresh(usable, source_age):
            self.last_bc = bc.copy()
            result = bc.copy()
            result[:, 6] = -1.  # BC cannot open before authorization.
            return result.ravel().astype(np.float32)
        if self.anchor is None:
            self.capture(position, velocity, rotations, center if center is not None else np.zeros(3))
        left_open = self.open_commanded
        if phase==6 and not self.authorization:
            self.failure_reason='visual_release_unauthorized'
        if phase>=7 and not self.open_verified:
            self.failure_reason='visual_release_unverified_retreat'
        if phase>=7 and np.any(widths[0]<=self.value('open_threshold_m')):
            self.failure_reason='visual_retreat_gripper_closed'
        if evidence.get('calibration_status')=='sensor_invalid':
            self.failure_reason='visual_placement_sensor_invalid'
        if (phase == 6 and not left_open and valid and not self.failure_reason
                and evidence.get('calibration_status')=='calibrated' and not self.collision_active(evidence)
                and .001<widths[1].mean()<.032):
            self.open_commanded = left_open = True
            self.switch('verify_open')
        if self.failure_reason or not self.fresh(usable, source_age):
            self.velocity[:] = self.angular_velocity[:] = 0.
            result = np.zeros((2, 7), dtype=np.float32)
            result[:, 6] = [1. if left_open else -1., -1.]
            return result.ravel()  # Emergency hold can decelerate faster than the normal slew.
        target = self.anchor.copy()
        if phase >= 7:
            target[0, 2] = self.value('retreat_height_m')
        elif self.state == 'collision_correction':
            target = self.correction_goal.copy()
        result=self.servo(target, robot, dt, left_open, evidence)
        if phase==8:
            if self.close_allowed(robot):
                result[7:]=bc[1]
                result[13]=-1.
            else:
                result[7:13]=0.
                result[13]=-1.
        return result

    def observe(self, robot, phase, points, opening, goals, dt, usable, source_age, evidence):
        self.elapsed += dt
        self.state_age += 1
        position, velocity, rotations, widths = self.robot_state(robot)
        valid, near_floor, center = self.geometry(points, position, opening, usable, source_age)
        self.stale_age = 0 if self.fresh(usable, source_age) else self.stale_age+1
        self.unknown_age = self.unknown_age+1 if phase==5 and evidence.get('calibration_status')!='calibrated' else 0
        if evidence.get('calibration_status') == 'sensor_invalid':
            self.failure_reason = 'visual_placement_sensor_invalid'
        if self.stale_age >= self.value('geometry_budget_steps'):
            self.failure_reason = 'visual_placement_perception_timeout'
        if self.unknown_age>=self.value('verify_budget_steps'):
            self.failure_reason='visual_placement_calibration_timeout'
        if self.failure_reason:
            return None, ''
        if phase == 5:
            if self.collision_active(evidence) and self.state == 'support_or_valid_pose':
                self.collision_rearmed=False;self.rearm_streak=0
                self.capture(position, velocity, rotations, center)
                self.switch('collision_hold')
            if self.state in ('collision_hold', 'collision_correction'):
                components=self.collision_components(evidence)
                quiet=(components is not None and evidence.get('calibration_status')=='calibrated'
                       and self.fresh(usable,source_age) and self.guards['footprint_inside'] and opening>.16
                       and evidence.get('force_stable')
                       and evidence.get('load_loss_n') is not None and evidence.get('raw_load_loss_n') is not None
                       and max(abs(evidence['load_loss_n']),abs(evidence['raw_load_loss_n']))<=self.value('collision_load_tolerance_n')
                       and components[0]<.8*evidence.get('lateral_limit_n',1.)
                       and components[1]<.8*evidence.get('moment_limit_nm',.12)
                       and np.max(np.linalg.norm(velocity,axis=1))<=self.value('ready_speed_m_s'))
                self.rearm_streak=self.rearm_streak+1 if quiet else 0
                if ((not evidence.get('collision') and evidence.get('calibration_status') == 'calibrated')
                        or self.rearm_streak>=self.value('collision_rearm_steps')):
                    if evidence.get('collision'):self.collision_rearmed=True;self.rearms+=1
                    self.anchor = self.center = None
                    self.switch('support_or_valid_pose')
                elif self.state_age >= (self.value('collision_hold_steps') if self.state == 'collision_hold'
                                        else self.value('correction_budget_steps')):
                    if self.corrections >= self.value('max_corrections'):
                        self.failure_reason = 'visual_placement_collision_timeout'
                    else:
                        self.corrections += 1
                        self.correction_goal = self.anchor.copy()
                        self.correction_goal[0, 2] += self.value('correction_step_m')*self.corrections
                        self.switch('collision_correction')
                return None, ''
            calibrated = evidence.get('calibration_status') == 'calibrated'
            slow = np.max(np.linalg.norm(velocity, axis=1)) <= self.value('ready_speed_m_s')
            retained = .003 < widths[0].mean() < .032 and .001 < widths[1].mean() < .032
            near_goal = np.linalg.norm(position[0]-goals[0]) < .035 and np.linalg.norm(position[1]-goals[1]) < .025
            grasp_geometry=(.015<position[0,2]-np.asarray(points)[0,2]<.085
                            and np.linalg.norm(position[0,:2]-np.asarray(points)[0,:2])<.04)
            force_ready = calibrated and evidence.get('support', False) and near_floor and grasp_geometry
            pose_ready = calibrated and valid and retained and near_goal
            self.guards.update(calibrated=calibrated, retained_grasp=bool(retained), low_speed=bool(slow),
                               force_ready=bool(force_ready), pose_ready=bool(pose_ready))
            self.stable = self.stable+1 if valid and slow and retained and grasp_geometry and (force_ready or pose_ready) else 0
            if self.stable >= self.value('stable_steps'):
                self.authorization = 'calibrated_support' if force_ready else 'verified_valid_pose'
                self.capture(position, velocity, rotations, center)
                self.switch('release')
                return 6, self.authorization
        elif phase == 6:
            opened = bool(np.all(widths[0] > self.value('open_threshold_m')))
            self.open_streak = self.open_streak+1 if opened and self.open_commanded else 0
            if opened and self.open_since is None and self.open_commanded:
                self.open_since = self.elapsed
            if not opened:
                self.open_since = None
            new_pixels = self.open_since is not None and self.elapsed-self.open_since > (source_age or 0.)+dt*.5
            payload = evidence.get('payload_n')
            unloaded = (payload is not None and evidence.get('raw_load_loss_n') is not None
                        and evidence.get('load_loss_n') is not None and evidence.get('force_stable', False)
                        and min(evidence['raw_load_loss_n'], evidence['load_loss_n']) >= max(.4, .5*payload))
            separated = position[0, 2]-np.asarray(points)[0, 2] >= .055
            self.guards.update(aperture_open=opened, new_pixels_after_open=bool(new_pixels),
                               unloaded=bool(unloaded), separated=bool(separated))
            if self.open_streak >= self.value('stable_steps') and valid and new_pixels and unloaded and separated:
                self.open_verified = True
                self.best_error = abs(self.value('retreat_height_m')-position[0, 2])
                self.no_progress = 0
                self.switch('retreat_clear')
                return 7, 'gripper_open_and_unloaded_verified'
            if self.state_age >= self.value('open_budget_steps') and not opened:
                self.failure_reason = 'visual_release_open_timeout'
            elif self.state_age >= self.value('verify_budget_steps') and opened:
                self.failure_reason = 'visual_release_verification_timeout'
        elif phase == 7:
            if np.any(widths[0]<=self.value('open_threshold_m')):
                self.failure_reason='visual_retreat_gripper_closed'
                return None,''
            error = abs(self.value('retreat_height_m')-position[0, 2])
            if error < self.best_error-.0005:
                self.best_error, self.no_progress = error, 0
            else:
                self.no_progress += 1
            right_held = (np.linalg.norm(position[1]-self.anchor[1]) < self.value('max_right_drift_m')
                          and .001 < widths[1].mean() < .032)
            clear = (self.open_verified and np.all(widths[0] > self.value('open_threshold_m'))
                     and error < self.value('retreat_tolerance_m')
                     and position[0, 2] > center[2]+.09+.20
                     and np.linalg.norm(position[0, :2]-self.anchor[0, :2]) < .02
                     and np.linalg.norm(velocity[0]) <= self.value('ready_speed_m_s') and right_held)
            self.guards.update(retreat_clear=bool(clear), right_grasp_held=bool(right_held))
            self.stable = self.stable+1 if clear and valid else 0
            if self.stable >= self.value('stable_steps'):
                self.switch('retreat_verified')
                return 8, 'retreat_clear_verified'
            if self.state_age >= self.value('retreat_budget_steps'):
                self.failure_reason = 'visual_retreat_clear_timeout'
            elif self.no_progress >= self.value('no_progress_steps'):
                self.failure_reason = 'visual_retreat_no_progress'
        elif phase==8 and not self.close_allowed(robot):
            # Hold closure and let the bounded left servo regain clearance.
            if self.state_age>=self.value('retreat_budget_steps'):
                self.failure_reason='visual_close_clearance_lost'
        return None, ''

    def telemetry(self):
        return {'substate': self.state, 'state_age': self.state_age, 'geometry_stale_ticks': self.stale_age,
                'stable_steps': self.stable, 'open_streak': self.open_streak,
                'open_commanded': self.open_commanded, 'open_verified': self.open_verified,
                'authorization': self.authorization, 'corrections': self.corrections,
                'collision_rearmed':self.collision_rearmed,'rearm_streak':self.rearm_streak,'rearms':self.rearms,
                'anchor_m': self.anchor.tolist() if self.anchor is not None else None,
                'command_velocity_m_s': self.velocity.tolist(),
                'command_angular_velocity_rad_s': self.angular_velocity.tolist(),
                'no_progress_steps': self.no_progress, 'guards': self.guards.copy(),
                'failure_reason': self.failure_reason}
