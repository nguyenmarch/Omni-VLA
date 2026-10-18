"""Replay the H0 baseline and the known raw-force false positive; never tune on it."""
import argparse
from collections import Counter
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from scripts import eval_visual_policy as visual
from scripts.evaluate_visual_suite import source_hash
from utils.checkpoint import file_hash
from utils.reporting import write_json


def compare_episodes(report, reference, seeds):
    actual = {row['seed']: row for row in report['episodes']}
    expected = {row['seed']: row for row in reference['episodes']}
    if list(actual) != seeds:
        raise ValueError('Replay seed order or count differs from the baseline contract')
    # Contact telemetry deliberately changes with force off; physical outcomes
    # and tick counts must still reproduce the historical high-threshold run.
    fields = ('success', 'steps', 'phase', 'failure_reason')
    return [{'seed': seed, 'field': field, 'expected': expected[seed][field],
             'actual': actual[seed][field]}
            for seed in seeds for field in fields if actual[seed][field] != expected[seed][field]]


def raw_contact_audit(args):
    """Inspect simulator contacts only after the manager has made its decision."""
    original_make = visual.make_environment
    original_update = visual.VisualTaskManager.update
    environments, events = [], []

    def make_environment(*positional, **keywords):
        env = original_make(*positional, **keywords)
        environments.append(env)
        return env

    def update(manager, robot):
        previous = manager.phase
        original_update(manager, robot)
        if previous != 5 or manager.phase != 6 or manager.last_transition_reason != 'wrench_contact':
            return
        env = environments[-1]
        pairs = Counter()
        for contact in env.data.contact:
            geoms = [int(contact.geom1), int(contact.geom2)]
            bodies = tuple(env.model.body(int(env.model.geom_bodyid[g])).name for g in geoms)
            pairs[bodies] += 1
        events.append({'step': env.step_counter, 'reason': manager.last_transition_reason,
                       'fz_world_n': manager.place_contact_fz_n,
                       'bottle_drawer_contact': any(set(pair) == {'bottle', 'drawer'} for pair in pairs),
                       'contact_body_pairs_scoring_only': [
                           {'bodies': list(pair), 'count': count} for pair, count in sorted(pairs.items())],
                       'bottle_site_scoring_only': env.site_pos('bottle_site').tolist(),
                       'ee': np.asarray(robot['ee']).tolist(),
                       'gripper': np.asarray(robot['gripper']).tolist()})

    with patch.object(visual, 'make_environment', make_environment), \
            patch.object(visual.VisualTaskManager, 'update', update):
        report = visual.evaluate(args)
    return report, events


def replay(config_path, output):
    config_path, output = Path(config_path), Path(output)
    contract = json.loads(config_path.read_text())
    if contract['schema'] != 'omni-p5-h0-baseline-v1':
        raise ValueError('Unsupported H0 baseline config')
    seeds = contract['development']['seeds']
    arguments = contract['arguments']
    baseline_args = visual.parser().parse_args(arguments + [
        '--seeds', *map(str, seeds), '--output', str(output/'baseline.json'),
        '--trace', str(output/'baseline.jsonl')])
    if baseline_args.place_contact_mode != 'off' or baseline_args.transfer_tolerance != .012:
        raise ValueError('H0 baseline must use force off and 12 mm transfer')
    if (baseline_args.legacy_perception or baseline_args.enable_arm_mask
            or baseline_args.enable_temporal_smoothing or baseline_args.async_perception):
        raise ValueError('H0 replay requires synchronous P5 fallback without mask/EMA')
    for key in ('policy', 'grounding'):
        if file_hash(getattr(baseline_args, key)) != contract[key+'_sha256']:
            raise ValueError(f'H0 {key} checkpoint hash mismatch')
    reference_paths = [contract['development']['reference_report'], contract['raw_debug']['reference_report'],
                       *contract['reference_regression_reports'].values()]
    references = {path: json.loads(Path(path).read_text()) for path in reference_paths}
    for path, report in references.items():
        if any(report[key+'_sha256'] != contract[key+'_sha256'] for key in ('policy', 'grounding')):
            raise ValueError(f'Reference checkpoint mismatch: {path}')
    versions = {}
    for package in ('numpy', 'torch', 'mujoco', 'gymnasium', 'h5py', 'PyYAML', 'PyOpenGL'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    manifest = {'scope': 'H0 development replay, not a fresh evaluation or controller ablation suite',
                'baseline_contract': str(config_path), 'contract_sha256': file_hash(config_path),
                'source_sha256': source_hash(), 'python': sys.version, 'platform': platform.platform(),
                'versions': versions, 'MUJOCO_GL': os.environ.get('MUJOCO_GL'),
                'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                'git_status': subprocess.check_output(['git', 'status', '--short'], text=True),
                'reference_sha256': {path: file_hash(path) for path in reference_paths},
                'datasets_sha256': {path: file_hash(path) for path in
                                   ('data/drawer_dagger_v2.h5', 'data/grounding_rgbd_v2_256.h5')},
                'baseline_args': vars(baseline_args), 'raw_debug': contract['raw_debug']}
    # Refuse an existing folder, including partially completed runs.
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'manifest.json', manifest)
    baseline = visual.evaluate(baseline_args)
    raw = contract['raw_debug']
    raw_args = visual.parser().parse_args(arguments + [
        '--seeds', str(raw['seed']), '--place-contact-mode', 'raw_debug',
        '--transfer-tolerance', str(raw['transfer_tolerance_m']),
        '--output', str(output/'raw_debug.json'), '--trace', str(output/'raw_debug.jsonl')])
    raw_report, events = raw_contact_audit(raw_args)
    write_json(output/'contact_audit.json', {'scope': 'scoring-only oracle contact audit', 'events': events})
    baseline_drift = compare_episodes(baseline, references[contract['development']['reference_report']], seeds)
    raw_drift = compare_episodes(raw_report, references[raw['reference_report']], [raw['seed']])
    mask = [row['success'] for row in baseline['episodes']]
    expected_perception = contract['perception']
    perception_matches = all(report.get(key) == value
                             for report in (baseline, raw_report)
                             for key, value in expected_perception.items())
    false_positive_reproduced = (len(events) == 1 and events[0]['step'] == raw['expected_trigger_step']
                                 and np.isclose(events[0]['fz_world_n'], raw['expected_fz_n'], atol=1e-4, rtol=0)
                                 and not events[0]['bottle_drawer_contact'])
    checks = {'success_mask_matches': mask == contract['development']['success_mask'],
              'baseline_ticks_and_outcomes_match': not baseline_drift,
              'baseline_force_off': baseline['place_contact_mode'] == 'off'
                                    and all(row['place_contact_triggers'] == 0 for row in baseline['episodes']),
              'perception_matches': perception_matches,
              'raw_debug_ticks_and_outcomes_match': not raw_drift,
              'known_false_positive_reproduced': bool(false_positive_reproduced),
              'source_unchanged': source_hash() == manifest['source_sha256']}
    summary = {'h0_passed': all(checks.values()), 'checks': checks, 'success_mask': mask,
               'expected_success_mask': contract['development']['success_mask'],
               'baseline_successes': baseline['successes'], 'baseline_attempted': baseline['attempted'],
               'baseline_drift': baseline_drift, 'raw_debug_drift': raw_drift,
               'raw_debug_contact_events': len(events),
               'note': 'raw_debug intentionally reproduces a known incorrect contact trigger; calibrated awaits H3'}
    write_json(output/'summary.json', summary)
    print(json.dumps(summary, indent=2))
    if not summary['h0_passed']:
        raise RuntimeError(f'H0 replay gate failed; see {output}/summary.json')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/p5_control_baseline.json')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    replay(args.config, args.output)


if __name__ == '__main__':
    main()
