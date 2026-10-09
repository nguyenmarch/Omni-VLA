"""Certify a state-only dataset against the current scene by replaying every action.

This does not certify RGB-D compatibility or unseen trajectories. It permits an
explicit, recorded merge after a scene revision only when all recorded states
reproduce within tolerance. No dataset metadata is modified.
"""
import argparse
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from simulation.bimanual_env import ROOT
from simulation.expert import state_condition
from utils.config import make_environment
from utils.checkpoint import file_hash


def audit(dataset, output, tolerance=1e-5):
    env = make_environment()
    maxima = {}; rows = []
    try:
        with h5py.File(dataset) as h:
            for name, g in h['episodes'].items():
                if any(k in g['obs'] for k in ('rgb', 'depth', 'head_rgb', 'wrist_rgb')):
                    raise ValueError('Audit accepts state-only data')
                # Any image/camera data disqualifies state-only compatibility.
                invalid = []
                g['obs'].visititems(lambda key, value: invalid.append(key) if isinstance(value, h5py.Dataset) and value.ndim > 2 else None)
                if invalid: raise ValueError(f'Image arrays found: {invalid}')
                obs, _ = env.reset(seed=int(g.attrs['seed']))
                episode_max = 0.
                for step in range(len(g['action'])):
                    values = {'condition': state_condition(obs, int(g['phase'][step])),
                              **{f'obs/{k}': v for k,v in obs.items()}}
                    for key, value in values.items():
                        expected = g[key][step]
                        error = float(np.max(np.abs(np.asarray(value) - expected)))
                        maxima[key] = max(maxima.get(key, 0.), error)
                        episode_max = max(episode_max, error)
                        if not np.isfinite(error) or error > tolerance:
                            raise ValueError(f'Replay diverges {name}/{step}/{key}: {error}')
                    action = g['executed_action'][step] if 'executed_action' in g else g['action'][step]
                    obs, _, _, _, _ = env.step(action)
                rows.append({'episode': name, 'steps': len(g['action']), 'max_error': episode_max})
                print(json.dumps(rows[-1]), flush=True)
            legacy = str(h.attrs['scene_sha256'])
        result = {'passed': True, 'dataset_sha256': file_hash(dataset),
                  'legacy_scene_sha256': legacy,
                  'current_scene_sha256': hashlib.sha256((ROOT/'simulation/assets/bottle_drawer_scene.xml').read_bytes()).hexdigest(),
                  'scope': 'all recorded state-only trajectories; no image compatibility claim',
                  'tolerance': tolerance, 'max_errors': maxima, 'episodes': rows}
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2)+'\n')
        return result
    finally:
        env.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); audit(a.dataset, a.output)
