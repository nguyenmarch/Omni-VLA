"""Collect/audit H3 development probes with real friction grasps and contacts.

The collector is privileged (ScriptedExpert + fixture goals); the observer sees
only the serialized `robot` sub-dictionary. Oracle contacts are sampled AFTER
the observer and stored under scoring_only. Nothing here measures task SR.
"""
import argparse
import json
from pathlib import Path
import platform
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from models.impedance_controller import rotation_error
from simulation.expert import ScriptedExpert
from simulation.placement_contact import PlacementContactObserver, PLACEMENT_DEFAULTS
from simulation.vision import robot_observation
from utils.checkpoint import file_hash
from utils.config import make_environment, ROOT
from utils.reporting import write_json
from utils.visual_protocol import object_hash
from utils.visual_scoring import scoring_snapshot


def observer_sample(env, obs, previous_ee, stage):
    robot = robot_observation(obs)
    ee = robot['ee'].reshape(2, 3)
    robot['ee_velocity'] = np.zeros((2, 3)) if previous_ee is None else (ee-previous_ee)/env.dt
    robot['wrist_rotations'] = np.stack([env.data.site(side+'_wrist').xmat.reshape(3, 3).copy()
                                        for side in ('left', 'right')])
    robot['placement_stage'] = int(stage)
    return robot


def labels(score):
    finger_contact = any('bottle' in row['bodies'] and any('left_' in body and 'finger' in body
                         for body in row['bodies']) for row in score['contacts'])
    # A bottle already dropped onto the floor is not supported in the grasp.
    supported = score['bottle_floor_contact'] and finger_contact
    side = any('bottle' in row['bodies'] and 'drawer' in row['bodies']
               and 'drawer_floor' not in row['geoms'] for row in score['contacts'])
    return {'floor_contact': score['bottle_floor_contact'], 'supported_in_grasp': bool(supported),
            'side_contact': bool(side), 'finger_contact': bool(finger_contact)}


def collect_trial(seed, profile, fixture, options, stream):
    env = make_environment(xml_path=fixture if profile == 'bottom' else None)
    observer = PlacementContactObserver(options)
    started = time.perf_counter()
    rows = []
    previous_ee = None
    try:
        obs, _ = env.reset(seed=seed)
        expert = ScriptedExpert(env)

        def record(stage, segment, action=None):
            nonlocal previous_ee
            robot = observer_sample(env, obs, previous_ee, stage)
            previous_ee = robot['ee'].reshape(2, 3).copy()
            before = time.perf_counter()
            result = observer.update(robot, env.dt)
            duration = time.perf_counter()-before
            # Privileged labels are read only after the observable prediction.
            score = scoring_snapshot(env)
            row = {'seed': seed, 'profile': profile, 'segment': segment,
                   'sim_time_s': float(env.data.time), 'wall_time_s': time.perf_counter()-started,
                   'observer_update_ms': duration*1000,
                   'robot': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in robot.items()},
                   'action_executed': action.tolist() if action is not None else None,
                   'prediction': result, 'labels_scoring_only': labels(score), 'scoring_only': score}
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            rows.append(row)

        record(0, 'empty')
        for _ in range(1000):
            stage = expert.phase
            action = expert.action()
            obs, *_ = env.step(action)
            record(stage, 'empty' if stage < 2 else 'grasp' if stage == 2 else 'carry', action)
            expert.update()
            if expert.phase >= 5:
                break
            if expert.failed:
                return rows, 'collector_'+expert.failure_reason
        else:
            return rows, 'collector_transfer_not_reached'
        center = env.data.body('drawer').xpos.copy()  # Privileged collection goal only.
        target = np.r_[center[:2], .72]
        right = env.site_pos('right_ee').copy()
        for tick in range(190):
            if tick == 50:
                target = env.site_pos('left_ee').copy()
            if tick >= 50 and profile == 'bottom':
                target[2] = max(.54, target[2]-.0015)
            if tick >= 50 and profile == 'side':
                target[1] = min(center[1]+.13, target[1]+.001)
                target[2] = max(.54, target[2]-.0015)
            action = np.zeros(14, dtype=np.float32)
            action[[6, 13]] = -1.
            for arm, (side, goal) in enumerate((('left', target), ('right', right))):
                action[7*arm:7*arm+3] = np.clip(.7*(goal-env.site_pos(side+'_ee'))/.015, -.3, .3)
                rotation = env.data.site(side+'_ee').xmat.reshape(3, 3)
                action[7*arm+3:7*arm+6] = np.clip(rotation_error(expert.rotation, rotation)/.08, -.3, .3)
            obs, *_ = env.step(action)
            record(4 if tick < 50 else 5, 'carry' if tick < 50 else 'probe', action)
        return rows, None
    finally:
        env.close()


def trial_summary(rows, collection_failure=None):
    probe = [r for r in rows if r['segment'] == 'probe']
    support = [r for r in probe if r['prediction']['support']]
    positives = [r for r in probe if r['labels_scoring_only']['supported_in_grasp']]
    true_support = [r for r in support if r['labels_scoring_only']['supported_in_grasp']]
    false_support = [r for r in support if not r['labels_scoring_only']['supported_in_grasp']]
    first = positives[0] if positives else None
    detected = next((r for r in true_support if r['sim_time_s'] >= first['sim_time_s']), None) if first else None
    latency = ({'sim_s': detected['sim_time_s']-first['sim_time_s'],
                'wall_s': detected['wall_time_s']-first['wall_time_s']} if detected else None)
    return {'seed': rows[0]['seed'], 'profile': rows[0]['profile'], 'collection_failure': collection_failure,
            'ticks': len(rows), 'probe_ticks': len(probe),
            'calibrated_probe_ticks': sum(r['prediction']['calibration_status'] == 'calibrated' for r in probe),
            'supported_in_grasp_ticks': len(positives),
            'side_contact_ticks': sum(r['labels_scoring_only']['side_contact'] for r in probe),
            'support_ticks': len(support), 'false_support_ticks': len(false_support),
            'detected_bottom': bool(detected), 'missed_bottom': bool(positives and not detected),
            'collision_veto_ticks': sum(r['prediction']['collision'] for r in probe),
            'first_support_latency': latency,
            'calibrated_payload_n': next((r['prediction']['payload_n'] for r in reversed(probe)
                                          if r['prediction']['payload_n'] is not None), None)}


def collect(args):
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(set(args.seeds)) != len(args.seeds) or any(s < 0 for s in args.seeds):
        raise ValueError('Unique nonnegative development seeds required')
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    if config.get('schema') != 'omni-placement-calibration-v1':
        raise ValueError('Unsupported placement calibration config')
    options = PlacementContactObserver(config['observer_options']).options
    output.mkdir(parents=True)
    original = ROOT/'simulation/assets/bottle_drawer_scene.xml'
    tree = ET.parse(original)
    tree.getroot().find('compiler').set('meshdir', str(original.parent/'panda/assets'))
    tree.getroot().find(".//geom[@name='drawer_floor']").set('pos', '0 0 .12')
    fixture = output/'raised_floor_fixture.xml'
    tree.write(fixture)
    from scripts.evaluate_visual_suite import source_hash
    manifest = {'schema': 'omni-placement-calibration-run-v1', 'purpose': 'development',
                'seeds': args.seeds, 'profiles': ['free', 'bottom', 'side'],
                'observer_options': options, 'observer_options_sha256': object_hash(options),
                'config': str(config_path), 'config_sha256': file_hash(config_path),
                'source_sha256': source_hash(), 'scene_sha256': file_hash(original),
                'fixture_sha256': file_hash(fixture), 'fixture_floor_offset_m': .12,
                'fixture_scope': 'Only bottom probes use a floor raised 12 cm at reset, compiled from XML. '
                                 'No object teleport, weld, force injection or contact labels enter observer.',
                'python': platform.python_version(), 'mujoco': mujoco.__version__, 'numpy': np.__version__}
    write_json(output/'manifest.json', manifest)
    trials = []
    with (output/'samples.jsonl').open('x') as stream:
        for seed in args.seeds:
            for profile in manifest['profiles']:
                rows, failure = collect_trial(seed, profile, str(fixture), options, stream)
                result = trial_summary(rows, failure)
                trials.append(result)
                print(json.dumps(result), flush=True)
    result = aggregate(trials)
    result.update(manifest_sha256=file_hash(output/'manifest.json'), samples_sha256=file_hash(output/'samples.jsonl'))
    write_json(output/'report.json', result)
    return result


def aggregate(trials):
    false = sum(t['false_support_ticks'] for t in trials)
    bottom = [t for t in trials if t['profile'] == 'bottom']
    side = [t for t in trials if t['profile'] == 'side']
    return {'trials': trials, 'false_support_ticks': false,
            'detected_bottom_trials': sum(t['detected_bottom'] for t in bottom),
            'missed_bottom_trials': sum(t['missed_bottom'] for t in bottom),
            'development_gate_passed': bool(trials and not false
                and all(t['collection_failure'] is None for t in trials)
                and all(t['calibrated_probe_ticks'] for t in trials)
                and bottom and all(t['detected_bottom'] for t in bottom)
                and side and all(t['side_contact_ticks'] and t['collision_veto_ticks'] for t in side)),
            'scope': 'Controlled development fixture probes, not IID/OOD task success or wall-clock real-time validation.'}


def audit(args):
    root = Path(args.dataset)
    manifest = json.loads((root/'manifest.json').read_text())
    original = json.loads((root/'report.json').read_text())
    if (file_hash(root/'samples.jsonl') != original['samples_sha256']
            or file_hash(root/'manifest.json') != original['manifest_sha256']):
        raise ValueError('Calibration data/manifest checksum changed')
    options = manifest['observer_options']
    assert object_hash(options) == manifest['observer_options_sha256']
    groups = {}
    observers = {}
    for line in (root/'samples.jsonl').open():
        row = json.loads(line)
        key = (row['seed'], row['profile'])
        observer = observers.setdefault(key, PlacementContactObserver(options))
        prediction = observer.update(row['robot'], .04)
        if prediction != row['prediction']:
            raise ValueError(f'Observer replay changed at {key}, {row["sim_time_s"]}')
        groups.setdefault(key, []).append(row)
    trials = [trial_summary(rows, prior['collection_failure'])
              for rows, prior in zip(groups.values(), original['trials'], strict=True)]
    result = aggregate(trials)
    if result['trials'] != original['trials']:
        raise ValueError('Calibration audit aggregate changed')
    print(json.dumps({'replayed_trials': len(trials), 'identical_predictions': True,
                      'development_gate_passed': result['development_gate_passed']}))
    return result


def replay_known(args):
    """Replay H0 actions in physics; compare H3 at the exact old false claim."""
    from simulation.randomization import DomainRandomizer
    from scripts.evaluate_visual_suite import source_hash
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    reference = json.loads(Path(args.reference_report).read_text())
    episode = next(e for e in reference['episodes'] if e['seed'] == args.seed)
    claim = next(t for t in episode['phase_transitions'] if t['reason'] == 'wrench_contact')
    actions = [json.loads(line) for line in Path(args.trace).open()]
    actions = [r for r in actions if r['seed'] == args.seed and r['step'] < claim['step']]
    if [r['step'] for r in actions] != list(range(claim['step'])):
        raise ValueError('Known-case replay requires a complete ordered action trace')
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    observer = PlacementContactObserver(config['observer_options'])
    root.mkdir(parents=True)
    manifest = {'purpose': 'development_known_false_positive', 'seed': args.seed,
                'trace': args.trace, 'trace_sha256': file_hash(args.trace),
                'reference_report': args.reference_report, 'reference_sha256': file_hash(args.reference_report),
                'source_sha256': source_hash(), 'observer_options': observer.options,
                'config_sha256': file_hash(config_path), 'claim_step': claim['step']}
    write_json(root/'manifest.json', manifest)
    env = make_environment()
    rows = []
    try:
        obs, _ = env.reset(seed=args.seed)
        DomainRandomizer(env).apply(args.seed, 'iid')
        obs = env._get_observation()
        begun = time.perf_counter()
        with (root/'samples.jsonl').open('x') as stream:
            for row in actions:
                previous = obs['ee'].reshape(2, 3).copy()
                obs, *_ = env.step(row['action'])
                robot = observer_sample(env, obs, previous, row['phase'] if row['phase'] is not None else 0)
                prediction = observer.update(robot, env.dt)
                score = scoring_snapshot(env)
                sample = {'step': row['step']+1, 'sim_time_s': float(env.data.time),
                          'wall_time_s': time.perf_counter()-begun,
                          'robot': {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in robot.items()},
                          'prediction': prediction, 'labels_scoring_only': labels(score), 'scoring_only': score}
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
                rows.append(sample)
    finally:
        env.close()
    final = rows[-1]
    fz = final['prediction']['wrench_world'][2]
    result = {'seed': args.seed, 'step': final['step'], 'raw_fz_world_n': fz,
              'reference_raw_fz_world_n': claim['place_contact_fz_n'],
              'raw_force_reproduced': bool(np.isclose(fz, claim['place_contact_fz_n'], atol=1e-5, rtol=0)),
              'calibration_status': final['prediction']['calibration_status'],
              'h3_support': final['prediction']['support'],
              'h3_load_loss_n': final['prediction']['load_loss_n'],
              'bottle_drawer_contact_scoring_only': final['scoring_only']['bottle_drawer_contact'],
              'false_support_ticks_before_claim': sum(r['prediction']['support']
                    and not r['labels_scoring_only']['supported_in_grasp'] for r in rows),
              'samples_sha256': file_hash(root/'samples.jsonl'),
              'manifest_sha256': file_hash(root/'manifest.json')}
    result['development_gate_passed'] = bool(result['raw_force_reproduced']
        and result['calibration_status'] == 'calibrated' and not result['h3_support']
        and not result['bottle_drawer_contact_scoring_only'] and not result['false_support_ticks_before_claim'])
    write_json(root/'report.json', result)
    print(json.dumps(result), flush=True)
    return result


def replay_actions(args):
    """Audit a complete saved visual trajectory with sensors recomputed by physics.

    This is a development replay. Actions are frozen, not resampled by BC;
    wall times here measure the replay, not the original visual runtime.
    """
    from simulation.randomization import DomainRandomizer
    from scripts.evaluate_visual_suite import source_hash
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(root)
    reference = json.loads(Path(args.reference_report).read_text())
    episode = next(e for e in reference['episodes'] if e['seed'] == args.seed)
    actions = [json.loads(line) for line in Path(args.trace).open()]
    actions = [r for r in actions if r['seed'] == args.seed and r.get('executed', True)]
    if [r['step'] for r in actions] != list(range(episode['steps'])):
        raise ValueError('Complete ordered executed action trace required')
    config_path = Path(args.config)
    config = json.loads(config_path.read_text())
    observer = PlacementContactObserver(config['observer_options'])
    root.mkdir(parents=True)
    write_json(root/'manifest.json', {'purpose': 'development_frozen_action_replay', 'seed': args.seed,
               'trace': args.trace, 'trace_sha256': file_hash(args.trace),
               'reference_report': args.reference_report, 'reference_sha256': file_hash(args.reference_report),
               'source_sha256': source_hash(), 'observer_options': observer.options,
               'config_sha256': file_hash(config_path)})
    env = make_environment()
    max_ee_error = false_supports = true_supports = 0
    samples = []
    try:
        obs, _ = env.reset(seed=args.seed)
        DomainRandomizer(env).apply(args.seed, 'iid')
        obs = env._get_observation()
        with (root/'samples.jsonl').open('x') as stream:
            for row in actions:
                before = obs['ee'].reshape(2, 3).copy()
                obs, *_ = env.step(row['action'])
                robot = observer_sample(env, obs, before, row['phase'] if row['phase'] is not None else 0)
                prediction = observer.update(robot, env.dt)
                score = scoring_snapshot(env)
                labeled = labels(score)
                false_supports += int(prediction['support'] and not labeled['supported_in_grasp'])
                true_supports += int(prediction['support'] and labeled['supported_in_grasp'])
                error = float(np.max(abs(obs['ee'].reshape(2, 3)-row['ee_position_after_step_m'])))
                max_ee_error = max(max_ee_error, error)
                sample = {'seed': args.seed, 'step': row['step'], 'robot': {
                    k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in robot.items()},
                    'prediction': prediction, 'labels_scoring_only': labeled, 'scoring_only': score}
                stream.write(json.dumps(sample, allow_nan=False)+'\n')
                samples.append(sample)
        final = env.info()
    finally:
        env.close()
    result = {'seed': args.seed, 'ticks': len(actions), 'max_ee_replay_error_m': max_ee_error,
              'physical_success': final['success'], 'reference_success': episode['success'],
              'false_support_ticks': false_supports, 'true_support_ticks': true_supports,
              'samples_sha256': file_hash(root/'samples.jsonl'),
              'manifest_sha256': file_hash(root/'manifest.json')}
    result['development_gate_passed'] = bool(max_ee_error <= 1e-7
        and final['success'] == episode['success'] and not false_supports)
    write_json(root/'report.json', result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    c = sub.add_parser('collect')
    c.add_argument('--output', required=True)
    c.add_argument('--config', default='configs/p5_placement_observer.json')
    c.add_argument('--seeds', type=int, nargs='+', default=[76000, 76001, 76002])
    a = sub.add_parser('audit')
    a.add_argument('--dataset', required=True)
    r = sub.add_parser('replay-known')
    r.add_argument('--output', required=True)
    r.add_argument('--config', default='configs/p5_placement_observer.json')
    r.add_argument('--seed', type=int, default=76000)
    # H1 recorded the initial await-perception holds missing from H0's trace.
    r.add_argument('--trace', default='artifacts/p5_h1_20261015/Raw_contact_debug/iid.jsonl')
    r.add_argument('--reference-report', default='artifacts/p5_h0_20261015/raw_debug.json')
    v = sub.add_parser('replay-actions')
    v.add_argument('--output', required=True)
    v.add_argument('--trace', required=True)
    v.add_argument('--reference-report', required=True)
    v.add_argument('--seed', type=int, required=True)
    v.add_argument('--config', default='configs/p5_placement_observer.json')
    args = p.parse_args()
    {'collect': collect, 'audit': audit, 'replay-known': replay_known,
     'replay-actions': replay_actions}[args.command](args)


if __name__ == '__main__':
    main()
