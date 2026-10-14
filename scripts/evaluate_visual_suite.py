"""Prepare an immutable seed/input plan, then run a resumable visual IID/OOD suite."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import h5py

from scripts.eval_visual_policy import evaluate, parser as visual_parser
from utils.checkpoint import file_hash
from utils.reporting import write_json, wilson

ROOT = Path(__file__).resolve().parents[1]
PROFILES = ('iid', 'camera', 'lighting', 'friction', 'position', 'combined')


def source_hash():
    digest = hashlib.sha256()
    for directory in ('models', 'simulation', 'utils', 'scripts', 'configs'):
        for path in sorted((ROOT/directory).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                digest.update(str(path.relative_to(ROOT)).encode())
                digest.update(path.read_bytes())
    for name in ('pyproject.toml', 'requirements-cpu.lock'):
        digest.update(name.encode())
        digest.update((ROOT/name).read_bytes())
    return digest.hexdigest()


def validate_seeds(sets, datasets, historical=()):
    flat = [seed for seeds in sets.values() for seed in seeds]
    if len(flat) != len(set(flat)):
        raise ValueError('Overlapping evaluation seed ranges')
    used = set(historical)
    for dataset in datasets:
        with h5py.File(dataset) as file:
            used.update(int(group.attrs['seed']) for group in file['episodes'].values())
    if used.intersection(flat):
        raise ValueError('Evaluation seed overlaps training or prior evaluation')


def prepare(args):
    sets = {profile: list(range(args.seed+index*1000,
                               args.seed+index*1000+(args.iid_episodes if profile == 'iid' else args.ood_episodes)))
            for index, profile in enumerate(PROFILES)}
    used = set()
    for path in (ROOT/'artifacts').rglob('*.json'):
        try:
            report = json.loads(path.read_text())
            if isinstance(report, dict):
                used.update(row['seed'] for row in report.get('episodes', []) if isinstance(row, dict) and 'seed' in row)
                if report.get('schema') == 'omni-visual-eval-plan-v1':
                    used.update(seed for seeds in report['sets'].values() for seed in seeds)
                if report.get('schema') == 'omni-state-eval-plan-v1':
                    used.update(report['seeds'])
        except (ValueError, TypeError, KeyError):
            continue
    datasets = ['data/drawer_dagger_v2.h5', 'data/grounding_rgbd_v2_256.h5']
    validate_seeds(sets, datasets, used)
    plan = {'schema': 'omni-visual-eval-plan-v1', 'sets': sets,
            'policy': args.policy, 'grounding': args.grounding,
            'policy_sha256': file_hash(args.policy), 'grounding_sha256': file_hash(args.grounding),
            'source_sha256': source_hash(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'datasets_sha256': {path: file_hash(path) for path in datasets},
            'options': {'async_perception': args.async_perception, 'transition_tolerance': args.transition_tolerance,
                        'max_rejections': 3, 'perception_every': 5},
            'selection_policy': 'Prepared before evaluation; no tuning on these seeds. One checkpoint/training seed. Report profiles separately.',
            'gates': {'iid_observed_target': .9, 'iid_population_lower_bound_target': .8}}
    Path(args.output).mkdir(parents=True, exist_ok=False)
    write_json(Path(args.output)/'plan.json', plan)
    return plan


def validate_report(plan, profile, report, plan_sha256):
    seeds = plan['sets'][profile]
    if (report['seeds_requested'] != seeds or report['attempted'] != len(seeds)
            or [row['seed'] for row in report['episodes']] != seeds
            or report['successes'] != sum(bool(row['success']) for row in report['episodes'])
            or report['profile'] != profile
            or report['policy_sha256'] != plan['policy_sha256']
            or report['grounding_sha256'] != plan['grounding_sha256']
            or report['suite_plan_sha256'] != plan_sha256):
        raise ValueError('Existing report incompatible or incomplete')


def run(output):
    folder = Path(output)
    plan = json.loads((folder/'plan.json').read_text())
    if source_hash() != plan['source_sha256']:
        raise ValueError('Source changed after plan preparation; prepare a new plan before evaluation')
    for key in ('policy', 'grounding'):
        if file_hash(plan[key]) != plan[key+'_sha256']:
            raise ValueError('Checkpoint changed after plan preparation')
    for path, expected in plan['datasets_sha256'].items():
        if file_hash(path) != expected:
            raise ValueError('Dataset changed after plan preparation')
    summary = []
    for profile, seeds in plan['sets'].items():
        path = folder/(profile+'.json')
        if path.exists():
            report = json.loads(path.read_text())
            validate_report(plan, profile, report, file_hash(folder/'plan.json'))
        else:
            args = visual_parser().parse_args(['--grounding', plan['grounding'], '--policy', plan['policy'],
                                              '--profile', profile, '--output', str(path),
                                              '--trace', str(folder/(profile+'.jsonl')),
                                              '--diagnostic-dataset', 'data/drawer_dagger_v2.h5'])
            args.seeds = seeds
            for key, value in plan['options'].items():
                setattr(args, key, value)
            args.suite_plan_sha256 = file_hash(folder/'plan.json')
            report = evaluate(args)
        summary.append({'profile': profile, 'successes': report['successes'], 'attempted': report['attempted'],
                        'success_rate': report['success_rate'], 'wilson_95': wilson(report['successes'], report['attempted']),
                        'failures': {reason: sum(row['failure_reason'] == reason for row in report['episodes'])
                                     for reason in sorted({row['failure_reason'] for row in report['episodes'] if row['failure_reason']})},
                        'report': str(path)})
    iid = summary[0]
    result = {'plan_sha256': file_hash(folder/'plan.json'), 'profiles': summary,
              'iid_observed_90_gate': iid['success_rate'] >= .9,
              'iid_lower_95_bound_80_gate': iid['wilson_95'][0] >= .8,
              'limitations': 'Single training seed; IID/OOD profiles are not pooled. Perturbations do not establish new-object/open-world generalization.'}
    if (folder/'summary.json').exists():
        if json.loads((folder/'summary.json').read_text()) != result:
            raise ValueError('Existing summary differs from verified reports')
    else:
        write_json(folder/'summary.json', result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--output', required=True)
    p.add_argument('--seed', type=int, default=30000)
    p.add_argument('--iid-episodes', type=int, default=100)
    p.add_argument('--ood-episodes', type=int, default=20)
    p.add_argument('--policy', default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--grounding', default='weights/p5_student/best.pt')
    p.add_argument('--async-perception', action='store_true')
    p.add_argument('--transition-tolerance', type=float, default=.012)
    args = p.parse_args()
    if min(args.iid_episodes, args.ood_episodes) < 1 or max(args.iid_episodes, args.ood_episodes) > 1000:
        p.error('Episode counts must be in [1,1000]')
    if args.action == 'prepare':
        prepare(args)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
