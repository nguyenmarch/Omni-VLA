"""Privileged telemetry only. No control decision may consume these observations."""
from collections import defaultdict

import mujoco
import numpy as np


def scoring_snapshot(env):
    contacts = {}
    for contact_id, contact in enumerate(env.data.contact):
        if contact.efc_address < 0:
            continue
        ids = (int(contact.geom1), int(contact.geom2))
        key = tuple(sorted(ids))
        wrench = np.zeros(6)
        mujoco.mj_contactForce(env.model, env.data, contact_id, wrench)
        normal = float(abs(wrench[0]))
        if normal <= 1e-6:
            continue
        row = contacts.setdefault(key, {
            'geoms': [env.model.geom(g).name or f'geom_{g}' for g in key],
            'bodies': [env.model.body(int(env.model.geom_bodyid[g])).name for g in key],
            'contact_points': 0, 'normal_force_sum_n': 0.})
        row['contact_points'] += 1
        row['normal_force_sum_n'] += normal
    rows = list(contacts.values())
    bottle = env.site_pos('bottle_site')
    drawer = env.data.body('drawer').xpos.copy()
    relative = bottle-drawer
    inside = bool(abs(relative[0]) < .11 and abs(relative[1]) < .075 and .445 < bottle[2] < .57)
    return {'contacts': rows, 'bottle_position_m': bottle.tolist(), 'drawer_center_m': drawer.tolist(),
            'bottle_relative_to_drawer_m': relative.tolist(), 'inside_env_placement_envelope': inside,
            'bottle_drawer_contact': any(set(row['bodies']) == {'bottle', 'drawer'} for row in rows),
            'bottle_floor_contact': any(set(row['geoms']) == {'bottle_geom', 'drawer_floor'} for row in rows),
            'physical_placed': bool(env.placed),
            'physical_success': bool(env.opened and env.placed and inside
                                     and env.data.qpos[env.finger_qadr[:2]].mean() > .032
                                     and env.data.qpos[env.drawer_adr] < .035)}


class EpisodeAudit:
    """Score contact claims, commands, actual opening and latency after decisions."""
    def __init__(self):
        self.events = []
        self.floor_onset = self.release_command = self.opened = self.held = None
        self.phase_steps = defaultdict(int)
        self.deadline_misses = self.ticks = 0

    def record(self, phase, before, after, action, gripper_before, gripper_after,
               step, sim_before, sim_after, wall_before, wall_after, transition=None, held=False):
        self.phase_steps[phase] += 1
        if phase == 5 and self.floor_onset is None and before['bottle_floor_contact']:
            self.floor_onset = (sim_before, wall_before)
        if phase == 5 and self.floor_onset is None and after['bottle_floor_contact']:
            self.floor_onset = (sim_after, wall_after)
        if transition and transition['reason'] in ('wrench_contact','calibrated_support'):
            self.events.append({'kind': 'contact_trigger', 'step': step,
                                'sim_time_s': sim_after, 'wall_time_s': wall_after,
                                'false_contact': not after['bottle_floor_contact'],
                                'scoring_only': after})
        command = phase is not None and phase >= 5 and float(action[6]) > .6
        if command and self.release_command is None:
            self.release_command = (sim_before, wall_before)
            self.events.append({'kind': 'release_command', 'step': step, 'sim_time_s': sim_before,
                                'wall_time_s': wall_before,
                                'outside_env_placement_envelope': not before['inside_env_placement_envelope'],
                                'scoring_only': before})
        if (phase is not None and phase >= 5 and self.opened is None
                and np.mean(gripper_before[:2]) <= .032 < np.mean(gripper_after[:2])):
            self.opened = (sim_after, wall_after)
            self.events.append({'kind': 'actual_gripper_open', 'step': step, 'sim_time_s': sim_after,
                                'wall_time_s': wall_after,
                                'outside_env_placement_envelope': not after['inside_env_placement_envelope'],
                                'scoring_only': after})
        if held and self.floor_onset is not None and self.held is None:
            self.held = (sim_before, wall_before)

    def timing(self, elapsed, dt):
        self.ticks += 1
        self.deadline_misses += int(elapsed > dt)

    def summary(self, dt):
        def latency(target):
            if self.floor_onset is None or target is None or target[0] < self.floor_onset[0]:
                return None
            return {'sim_s': target[0]-self.floor_onset[0], 'wall_s': target[1]-self.floor_onset[1]}
        return {'phase_steps': {str(k): v for k, v in self.phase_steps.items()},
                'phase_duration_sim_s': {str(k): v*dt for k, v in self.phase_steps.items()},
                'contact_trigger_events': sum(e['kind'] == 'contact_trigger' for e in self.events),
                'false_contact_triggers': sum(e.get('false_contact', False) for e in self.events),
                'release_commands_outside_env_envelope': sum(
                    e['kind'] == 'release_command' and e['outside_env_placement_envelope'] for e in self.events),
                'actual_openings_outside_env_envelope': sum(
                    e['kind'] == 'actual_gripper_open' and e['outside_env_placement_envelope'] for e in self.events),
                'floor_contact_to_hold': latency(self.held),
                'floor_contact_to_release_command': latency(self.release_command),
                'floor_contact_to_actual_open': latency(self.opened),
                'control_ticks': self.ticks, 'deadline_misses': self.deadline_misses,
                'events': self.events,
                'definition': 'False force contact means no active bottle-floor contact at the claim; '
                              'release envelope uses env inside bounds and permits a small valid drop. '
                              'Null latency means an onset or subsequent response was not observed.'}
