"""Train-state envelopes and post-prediction diagnostics, never action inputs."""
import json
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np


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


def trace_row(manager, robot, action, seed, step, sim_age, wall_age, envelope=None):
    condition = manager.condition(robot)
    goals = np.stack(manager.goals())
    return {'seed': seed, 'step': step, 'phase': manager.phase, 'phase_age': manager.age,
            'position_error_m': np.linalg.norm(np.asarray(robot['ee']).reshape(2, 3)-goals, axis=1).tolist(),
            'observation_sim_age_s': sim_age, 'observation_wall_age_s': wall_age,
            'opening_m': manager.opening, 'action': np.asarray(action).tolist(),
            'distribution': envelope.measure(condition, manager.phase) if envelope else None}


def summarize_trace(path):
    groups = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        groups[(row['seed'], row['phase'])].append(row)
    result = []
    for (seed, phase), rows in groups.items():
        distributions = [r['distribution'] for r in rows if r['distribution'] is not None]
        result.append({'seed': seed, 'phase': phase, 'steps': len(rows),
                       'final_goal_error_m': rows[-1]['position_error_m'],
                       'p95_observation_sim_age_s': float(np.percentile([r['observation_sim_age_s'] for r in rows], 95)),
                       'mean_outside_train_envelope_fraction': float(np.mean([d['outside_train_envelope_fraction'] for d in distributions])) if distributions else None})
    return result
