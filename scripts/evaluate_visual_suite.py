"""Prepare an immutable seed/input plan, then run a resumable visual IID/OOD suite."""
import argparse
import hashlib
import json
import math
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
    profiles = tuple(args.profiles or PROFILES)
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles)-set(PROFILES):
        raise ValueError('Invalid or duplicate suite profiles')
    sets = {profile: list(range(args.seed+index*1000,
                               args.seed+index*1000+(args.iid_episodes if profile == 'iid' else args.ood_episodes)))
            for index, profile in enumerate(profiles)}
    used = set()
    for path in (ROOT/'artifacts').rglob('*.json'):
        try:
            report = json.loads(path.read_text())
            if isinstance(report, dict):
                used.update(row['seed'] for row in report.get('episodes', []) if isinstance(row, dict) and 'seed' in row)
                if report.get('schema') in ('omni-visual-eval-plan-v1','omni-visual-eval-plan-v2'):
                    used.update(seed for seeds in report['sets'].values() for seed in seeds)
                if report.get('schema') == 'omni-state-eval-plan-v1':
                    used.update(report['seeds'])
        except (ValueError, TypeError, KeyError):
            continue
    datasets = ['data/drawer_dagger_v2.h5', 'data/grounding_rgbd_v2_256.h5']
    validate_seeds(sets, datasets, used)
    plan = {'schema': 'omni-visual-eval-plan-v2', 'sets': sets,
            'policy': args.policy, 'grounding': args.grounding,
            'policy_sha256': file_hash(args.policy), 'grounding_sha256': file_hash(args.grounding),
            'source_sha256': source_hash(),
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'datasets_sha256': {path: file_hash(path) for path in datasets},
            'options': {'async_perception': args.async_perception, 'transition_tolerance': args.transition_tolerance,
                        'max_rejections': 3, 'perception_every': 5,
                        'arm_mask_candidate': args.arm_mask_candidate,
                        'temporal_smoothing_candidate': args.temporal_smoothing_candidate,
                        'fallback_max_age_s_candidate': args.fallback_max_age_s_candidate,
                        'transfer_tolerance_m': args.transfer_tolerance,
                        'place_contact_force_threshold_n': args.place_contact_force_threshold_n,
                        'place_contact_debounce_steps': args.place_contact_debounce_steps},
            'variants': ['baseline', 'candidate'] if args.paired else ['candidate'],
            'selection_policy': 'Prepared before evaluation; baseline and candidate use identical seeds, checkpoints, task, and physics. No tuning on these seeds. One checkpoint/training seed. Report profiles separately.',
            'gates': {'iid_observed_target': .8, 'iid_population_lower_bound_target': .8}}
    Path(args.output).mkdir(parents=True, exist_ok=False)
    write_json(Path(args.output)/'plan.json', plan)
    return plan


def validate_report(plan, profile, report, plan_sha256, legacy):
    seeds = plan['sets'][profile]
    if (report['seeds_requested'] != seeds or report['attempted'] != len(seeds)
            or [row['seed'] for row in report['episodes']] != seeds
            or report['successes'] != sum(bool(row['success']) for row in report['episodes'])
            or report['profile'] != profile
            or report.get('legacy_perception', False) != legacy
            or report.get('arm_mask_enabled', False) != (not legacy and plan['options']['arm_mask_candidate'])
            or report.get('temporal_smoothing_enabled', False) != (not legacy and plan['options']['temporal_smoothing_candidate'])
            or report.get('fallback_max_age_s') != (0.4 if legacy else plan['options']['fallback_max_age_s_candidate'])
            or report.get('transfer_tolerance_m') != plan['options']['transfer_tolerance_m']
            or report.get('place_contact_force_threshold_n') != plan['options']['place_contact_force_threshold_n']
            or report.get('place_contact_debounce_steps') != plan['options']['place_contact_debounce_steps']
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
    summaries = {}
    raw = {}
    for variant in plan['variants']:
        legacy = variant == 'baseline'
        summaries[variant] = []
        raw[variant] = {}
        variant_dir = folder/variant
        variant_dir.mkdir(exist_ok=True)
        for profile, seeds in plan['sets'].items():
            path = variant_dir/(profile+'.json')
            if path.exists():
                report = json.loads(path.read_text())
                validate_report(plan, profile, report, file_hash(folder/'plan.json'), legacy)
            else:
                variant_flags = (['--legacy-perception'] if legacy else
                                 (['--enable-arm-mask'] if plan['options']['arm_mask_candidate'] else []) +
                                 (['--enable-temporal-smoothing'] if plan['options']['temporal_smoothing_candidate'] else []))
                if not legacy:
                    variant_flags += ['--fallback-max-age-s', str(plan['options']['fallback_max_age_s_candidate'])]
                args = visual_parser().parse_args(['--grounding', plan['grounding'], '--policy', plan['policy'],
                                                  '--profile', profile, '--output', str(path),
                                                  '--trace', str(variant_dir/(profile+'.jsonl')),
                                                  '--diagnostic-dataset', 'data/drawer_dagger_v2.h5'] +
                                                  ['--transfer-tolerance', str(plan['options']['transfer_tolerance_m']),
                                                   '--place-contact-force-threshold-n', str(plan['options']['place_contact_force_threshold_n']),
                                                   '--place-contact-debounce-steps', str(plan['options']['place_contact_debounce_steps'])] +
                                                  variant_flags)
                args.seeds = seeds
                for key, value in plan['options'].items():
                    setattr(args, key, value)
                args.suite_plan_sha256 = file_hash(folder/'plan.json')
                report = evaluate(args)
            summaries[variant].append({'profile': profile, 'successes': report['successes'], 'attempted': report['attempted'],
                            'success_rate': report['success_rate'], 'wilson_95': wilson(report['successes'], report['attempted']),
                            'fallback_predictions': report.get('fallback_predictions', 0),
                            'place_contact_triggers': sum(row.get('place_contact_triggers', 0) for row in report['episodes']),
                            'fallback_episodes': report.get('fallback_episodes', 0),
                            'perception_rejections': report.get('perception_rejections', 0),
                            'invalid_reasons': report.get('perception_invalid_reasons', {}),
                            'failures': {reason: sum(row['failure_reason'] == reason for row in report['episodes'])
                                         for reason in sorted({row['failure_reason'] for row in report['episodes'] if row['failure_reason']})},
                            'report': str(path)})
            raw[variant][profile] = report
    paired = {}
    if {'baseline','candidate'} <= set(raw):
        for profile, baseline in raw['baseline'].items():
            candidate = raw['candidate'][profile]
            b = {row['seed']: row for row in baseline['episodes']}
            c = {row['seed']: row for row in candidate['episodes']}
            if list(b) != list(c):
                raise ValueError(f'Paired episode seeds differ for {profile}')
            rescued = sum(not b[s]['success'] and c[s]['success'] for s in b)
            regressed = sum(b[s]['success'] and not c[s]['success'] for s in b)
            n = rescued+regressed
            p_value = min(1., 2*sum(math.comb(n,k) for k in range(min(rescued,regressed)+1))/(2**n)) if n else 1.
            paired[profile] = {'rescued_episodes': rescued, 'regressed_episodes': regressed,
                               'net_success_delta': candidate['successes']-baseline['successes'],
                               'mcnemar_exact_p_two_sided': p_value}
    iid_candidate = next(row for row in summaries['candidate'] if row['profile']=='iid')
    result = {'plan_sha256': file_hash(folder/'plan.json'), 'variants': summaries, 'paired': paired,
              'iid_observed_80_gate': iid_candidate['success_rate'] >= .8,
              'iid_lower_95_bound_80_gate': iid_candidate['wilson_95'][0] >= .8,
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
    p.add_argument('--profiles', nargs='+', choices=PROFILES, help='Profile subset; defaults to all registered profiles')
    p.add_argument('--policy', default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--grounding', default='weights/p5_student/best.pt')
    p.add_argument('--async-perception', action='store_true')
    p.add_argument('--single-candidate', dest='paired', action='store_false', help='Skip paired legacy baseline (paired comparison is default)')
    p.set_defaults(paired=True)
    p.add_argument('--enable-arm-mask-candidate', dest='arm_mask_candidate', action='store_true',
                   help='Development ablation: compare temporal fallback with projected arm masking')
    p.set_defaults(arm_mask_candidate=False)
    p.add_argument('--enable-temporal-smoothing-candidate', dest='temporal_smoothing_candidate', action='store_true',
                   help='Development ablation: compare fallback with EMA smoothing')
    p.set_defaults(temporal_smoothing_candidate=False)
    p.add_argument('--fallback-max-age-s-candidate', type=float, default=.4,
                   help='Candidate landmark fallback age limit in simulated seconds')
    p.add_argument('--transition-tolerance', type=float, default=.012)
    p.add_argument('--transfer-tolerance', type=float, default=.025)
    p.add_argument('--place-contact-force-threshold-n', type=float, default=8.)
    p.add_argument('--place-contact-debounce-steps', type=int, default=3)
    args = p.parse_args()
    if min(args.iid_episodes, args.ood_episodes) < 1 or max(args.iid_episodes, args.ood_episodes) > 1000:
        p.error('Episode counts must be in [1,1000]')
    if (not np.isfinite(args.transfer_tolerance) or args.transfer_tolerance <= 0
            or not np.isfinite(args.place_contact_force_threshold_n) or args.place_contact_force_threshold_n <= 0
            or args.place_contact_debounce_steps < 1):
        p.error('transfer tolerance, contact threshold and debounce must be positive')
    if args.action == 'prepare':
        prepare(args)
    else:
        run(args.output)


if __name__ == '__main__':
    main()
