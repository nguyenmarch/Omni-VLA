"""Train-state envelopes and post-prediction diagnostics, never action inputs."""
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np
from simulation.expert import PHASES


class TrainingEnvelope:
    def __init__(self, path):
        self.envelopes = {}
        with h5py.File(path) as file:
            train = json.loads(file.attrs['splits'])['train']
            conditions = np.concatenate([file[f'episodes/{name}/condition'][:] for name in train])
        for phase in range(10):
            values = conditions[conditions[:, -10:].argmax(1) == phase, :57]
            if len(values):
                self.envelopes[phase] = np.quantile(values, [.01, .99], axis=0)

    def measure(self, condition, phase):
        if phase not in self.envelopes:
            return None
        low, high = self.envelopes[phase]
        scale = np.maximum(high-low, .05)
        excess = np.maximum(np.maximum(low-condition[:57], condition[:57]-high), 0)/scale
        return {'outside_train_envelope_fraction': float((excess > 0).mean()),
                'max_normalized_excess': float(excess.max())}


def trace_row(manager, robot, action, seed, step, sim_age, wall_age, envelope=None,
              before_override=None, ee_velocity=None, metadata=None, placement=None):
    condition = manager.condition(robot) if manager is not None else None
    goals = np.stack(manager.goals()) if manager is not None else None
    phase = manager.phase if manager is not None else None
    ee = np.asarray(robot['ee']).reshape(2, 3)
    error = goals-ee if goals is not None else None
    raw = np.asarray(robot['wrench']).reshape(2, 6)
    rotations = np.asarray(robot.get('wrist_rotations', np.tile(np.eye(3), (2, 1, 1))))
    world = np.concatenate([np.einsum('bij,bj->bi', rotations, raw[:, :3]),
                            np.einsum('bij,bj->bi', rotations, raw[:, 3:])], axis=1)
    mode = getattr(manager, 'place_contact_mode', 'off')
    metadata = metadata or {}
    transfer = getattr(manager, 'terminal_transfer', None)
    return {'schema': 'omni-visual-trace-v2', 'seed': seed, 'step': step, 'phase': phase,
            'substate': getattr(manager, 'substate', PHASES[phase] if phase is not None else 'await_perception'),
            'transfer': transfer.telemetry() if transfer is not None else None,
            'release': manager.verified_placement.telemetry() if manager is not None and
                manager.verified_placement is not None else None,
            'perceived_landmarks_m': manager.points.tolist() if manager is not None else None,
            'phase_age': manager.age if manager is not None else None,
            'position_error_m': np.linalg.norm(error, axis=1).tolist() if error is not None else None,
            'goal_position_m': goals.tolist() if goals is not None else None,
            'goal_error_xy_m': np.linalg.norm(error[:, :2], axis=1).tolist() if error is not None else None,
            'goal_error_z_m': abs(error[:, 2]).tolist() if error is not None else None,
            'ee_position_m': ee.tolist(), 'ee_rotation_world': robot.get('ee_rotations', None),
            'ee_velocity_m_s': np.asarray(ee_velocity).tolist() if ee_velocity is not None else None,
            'wrench_site': raw.tolist(), 'wrench_world': world.tolist(),
            'wrench_residual_world': placement.get('residual_world') if placement else None,
            'calibration_status': placement['calibration_status'] if placement else
                'disabled' if mode == 'off' else 'unavailable_H3',
            'placement_observer': placement,
            'contact_mode': mode, 'contact_streak': getattr(manager, 'place_contact_streak', 0),
            'gripper_aperture_m': np.asarray(robot['gripper']).reshape(2, 2).mean(1).tolist(),
            'gripper_fingers_m': np.asarray(robot['gripper']).reshape(2,2).tolist(),
            'action_before_override': np.asarray(before_override).tolist() if before_override is not None else None,
            'observation_sim_age_s': sim_age, 'observation_wall_age_s': wall_age,
            'perception_source_sim_age_s': sim_age+metadata.get('fallback_age_s', 0.) if sim_age is not None else None,
            'perception_source_wall_age_s': None if metadata.get('fallback_used', False) else wall_age,
            'fallback_used': bool(metadata.get('fallback_used', False)),
            'opening_m': manager.opening if manager is not None else None,
            'action': np.asarray(action).tolist() if action is not None else None,
            'distribution': envelope.measure(condition, phase) if envelope and manager is not None else None}


def summarize_trace(path):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if row.get('executed', True) is False or row['phase'] is None:
            continue
        groups[(row['seed'], row['phase'])].append(row)
    result = []
    for (seed, phase), rows in groups.items():
        distributions = [r['distribution'] for r in rows if r['distribution'] is not None]
        result.append({'seed': seed, 'phase': phase, 'steps': len(rows),
                       'final_goal_error_m': rows[-1]['position_error_m'],
                       'p95_observation_sim_age_s': float(np.percentile([r['observation_sim_age_s'] for r in rows if r['observation_sim_age_s'] is not None], 95))
                                                    if any(r['observation_sim_age_s'] is not None for r in rows) else None,
                       'mean_outside_train_envelope_fraction': float(np.mean([d['outside_train_envelope_fraction'] for d in distributions])) if distributions else None})
    return result
