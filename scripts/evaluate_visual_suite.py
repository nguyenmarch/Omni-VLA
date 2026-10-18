"""Immutable paired visual controller experiments with explicit seed provenance."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess

import h5py
import numpy as np

from scripts.eval_visual_policy import evaluate, parser as visual_parser
from simulation.expert import PHASES
from utils.checkpoint import file_hash
from utils.reporting import write_json, wilson
from utils.visual_protocol import normalize_options, object_hash, check_instruction, option_report_fields
from simulation.vision import FULL_INSTRUCTION

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'omni-visual-eval-plan-v3'
PROFILES = ('iid', 'camera', 'lighting', 'friction', 'position', 'combined')
TRAINING_DATASETS = ('data/drawer_dagger_v2.h5', 'data/grounding_rgbd_v2_256.h5')


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
    if not isinstance(sets, dict) or any(not isinstance(seeds, list) for seeds in sets.values()):
        raise ValueError('Seed sets must map profiles to ordered lists')
    flat = [seed for seeds in sets.values() for seed in seeds]
    if not flat or any(type(seed) is not int or seed < 0 for seed in flat):
        raise ValueError('Seeds must be nonnegative integers')
    if any(not seeds or len(seeds) > 1000 for seeds in sets.values()):
        raise ValueError('Episode counts must be in [1,1000]')
    if len(flat) != len(set(flat)):
        raise ValueError('Overlapping evaluation seed ranges')
    used = set(historical)
    for dataset in datasets:
        with h5py.File(dataset) as file:
            used.update(int(group.attrs['seed']) for group in file['episodes'].values())
    if used.intersection(flat):
        raise ValueError('Evaluation seed overlaps training or prior evaluation')


def recorded_seeds(value):
    """Reserve used/prepared seeds, including nested H0 and development manifests."""
    found = set()
    if isinstance(value, dict):
        for key, child in value.items():
            if key == 'seed' and type(child) is int:
                found.add(child)
            elif key == 'seeds' and isinstance(child, list):
                found.update(seed for seed in child if type(seed) is int)
            elif key == 'sets' and isinstance(child, dict):
                found.update(seed for seeds in child.values() if isinstance(seeds, list)
                             for seed in seeds if type(seed) is int)
            found.update(recorded_seeds(child))
    elif isinstance(value, list):
        for child in value:
            found.update(recorded_seeds(child))
    return found


def historical_seeds():
    used = set()
    paths = list((ROOT/'artifacts').rglob('*.json')) + list((ROOT/'configs').glob('*.json'))
    for path in paths:
        try:
            used.update(recorded_seeds(json.loads(path.read_text())))
        except (ValueError, UnicodeError):
            continue
    return used


def variant_config(value):
    if value.get('schema') != 'omni-visual-variants-v1':
        raise ValueError('Unsupported variants config schema')
    variants = value.get('variants')
    if not isinstance(variants, dict) or not variants or value.get('baseline') not in variants:
        raise ValueError('Explicit baseline must name one of the variants')
    resolved = {}
    for name, options in variants.items():
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', name):
            raise ValueError('Unsafe variant name')
        if not isinstance(options, dict) or set(options)-{'perception_options', 'controller_options'}:
            raise ValueError('Unknown variant option groups')
        resolved[name] = normalize_options(
            {**value.get('perception_options', {}), **options.get('perception_options', {})},
            {**value.get('controller_options', {}), **options.get('controller_options', {})})
    return resolved


def reference_options_match(report, options):
    if report.get('effective_options') is not None:
        recorded = report['effective_options']
        if normalize_options(recorded['perception_options'], recorded['controller_options']) != options:
            return False
    # Historical H0 reports predate normalized options; compare recorded fields.
    fields = option_report_fields(options)
    historical_defaults = {'transfer_mode': 'legacy', 'placement_observer': False}
    return all(report.get(key, historical_defaults.get(key)) == val for key, val in fields.items())


def prepare(args):
    if args.startup_hold_steps<0:
        raise ValueError('startup_hold_steps must be nonnegative')
    config_path = Path(args.variants_config)
    config = json.loads(config_path.read_text())
    variants = variant_config(config)
    instruction = check_instruction(args.instruction)
    explicit_sets = json.loads(Path(args.seed_file).read_text()) if args.seed_file else None
    profiles = tuple(args.profiles or (list(explicit_sets) if explicit_sets is not None else
                                       ('iid', 'camera', 'lighting', 'position')))
    if not profiles or len(set(profiles)) != len(profiles) or set(profiles)-set(PROFILES):
        raise ValueError('Invalid or duplicate suite profiles')
    if explicit_sets is not None and set(explicit_sets) != set(profiles):
        raise ValueError('Seed file profiles must exactly match selected profiles')
    sets = explicit_sets if explicit_sets is not None else {
        profile: list(range(args.seed+i*1000, args.seed+i*1000+
                            (args.iid_episodes if profile == 'iid' else args.ood_episodes)))
        for i, profile in enumerate(profiles)}
    datasets = list(TRAINING_DATASETS) + list(args.additional_training_dataset or [])
    validate_seeds(sets, datasets)  # Training overlap is forbidden even for explicit replay.
    references = {}
    for item in args.reference_report or []:
        profile, separator, path = item.partition('=')
        if not separator or profile not in sets or profile in references:
            raise ValueError('Reference must be a unique PROFILE=REPORT for a selected profile')
        report = json.loads(Path(path).read_text())
        rows = report.get('episodes', [])
        by_seed = {row['seed']: row for row in rows}
        if (report.get('profile') != profile or len(by_seed) != len(rows)
                or report.get('attempted') != len(rows) or report.get('successes') != sum(r['success'] for r in rows)
                or not set(sets[profile]) <= set(by_seed)
                or not reference_options_match(report, variants[config['baseline']])
                or report.get('instruction') != instruction):
            raise ValueError('Reference report is incomplete or incompatible with the baseline')
        for key in ('policy', 'grounding'):
            if report.get(key+'_sha256') != file_hash(getattr(args, key)):
                raise ValueError('Reference checkpoint differs from experiment checkpoint')
        references[profile] = {'path': path, 'sha256': file_hash(path),
                               'episodes': [by_seed[seed] for seed in sets[profile]]}
    if args.purpose == 'holdout' and references:
        raise ValueError('Holdout cannot replay reference seeds')
    if args.purpose == 'regression_replay' and set(references) != set(sets):
        raise ValueError('regression_replay requires a reference report for every profile')
    allowed = {row['seed'] for ref in references.values() for row in ref['episodes']}
    validate_seeds(sets, datasets, historical_seeds()-allowed)
    if args.purpose == 'holdout':
        validate_seeds(sets, datasets, historical_seeds())
    dataset_paths = set(datasets)
    if args.diagnostic_dataset:
        dataset_paths.add(args.diagnostic_dataset)
    plan = {'schema': SCHEMA, 'purpose': args.purpose, 'sets': sets,
            'startup_hold_steps':args.startup_hold_steps,
            'observation_age_clock':args.observation_age_clock,
            'baseline': config['baseline'], 'variants': variants, 'instruction': instruction,
            'policy': args.policy, 'grounding': args.grounding,
            'policy_sha256': file_hash(args.policy), 'grounding_sha256': file_hash(args.grounding),
            'source_sha256': source_hash(),
            'config': {'path': str(config_path), 'sha256': file_hash(config_path)},
            'seed_file': {'path': args.seed_file, 'sha256': file_hash(args.seed_file)} if args.seed_file else None,
            'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'datasets_sha256': {path: file_hash(path) for path in sorted(dataset_paths)},
            'references': references, 'diagnostic_dataset': args.diagnostic_dataset,
            'selection_policy': 'All variants use identical seeds/checkpoints/physics. Options are explicit per variant. '
                                'Replay seeds are diagnostic; holdout must be unused and selected before reading results.',
            'gates': {'iid_observed_target': .8, 'iid_population_lower_bound_target': .8}}
    folder = Path(args.output)
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder/'plan.json', plan)
    write_json(folder/'plan.seal.json', {'plan_sha256': file_hash(folder/'plan.json')})
    return plan


def validate_report(plan, profile, report, plan_sha256, variant):
    seeds, options = plan['sets'][profile], plan['variants'][variant]
    rows = report.get('episodes', [])
    if (report.get('seeds_requested') != seeds or report.get('attempted') != len(seeds)
            or type(report.get('attempted')) is not int
            or len(rows) != len(seeds) or [r.get('seed') for r in rows] != seeds
            or any(type(r.get('success')) is not bool for r in rows)
            or report.get('successes') != sum(r['success'] for r in rows)
            or report.get('success_rate') != sum(r['success'] for r in rows)/len(seeds)
            or report.get('wilson_95') != wilson(sum(r['success'] for r in rows), len(seeds))
            or any(type(r.get('steps')) is not int or r['steps'] < 0 for r in rows)
            or any(r.get('phase') is not None and r['phase'] not in PHASES for r in rows)
            or any((r['success'] and r.get('failure_reason') is not None)
                   or (not r['success'] and not r.get('failure_reason')) for r in rows)
            or report.get('profile') != profile or report.get('instruction') != plan['instruction']
            or report.get('effective_options') != options or report.get('options_sha256') != object_hash(options)
            or not reference_options_match(report, options)
            or report.get('policy_sha256') != plan['policy_sha256']
            or report.get('grounding_sha256') != plan['grounding_sha256']
            or report.get('suite_plan_sha256') != plan_sha256 or report.get('suite_variant') != variant
            or report.get('suite_source_sha256') != plan['source_sha256']
            or report.get('startup_hold_steps',0)!=plan.get('startup_hold_steps',0)
            or report.get('observation_age_clock','both')!=plan.get('observation_age_clock','both')
            or report.get('loop') is not False or report.get('stop_reason') != 'episode_limit'
            or report.get('scoring_diagnostics') is not True):
        raise ValueError('Existing report incompatible or incomplete')


def validate_trace(path, report):
    if file_hash(path) != report.get('trace_sha256'):
        raise ValueError('Trace checksum mismatch')
    seeds = report['seeds_requested']
    rows = {seed: [] for seed in seeds}
    order = []
    with Path(path).open() as stream:
        for line in stream:
            row = json.loads(line)
            if (row.get('schema') != 'omni-visual-trace-v2' or row.get('seed') not in rows
                    or type(row.get('executed')) is not bool
                    or not {'action_before_override', 'action', 'ee_position_m', 'ee_rotation_world',
                            'ee_velocity_m_s', 'transition_reason', 'wrench_site', 'wrench_world',
                            'wrench_residual_world', 'calibration_status', 'contact_streak',
                            'gripper_aperture_m', 'goal_error_xy_m', 'goal_error_z_m', 'substate',
                            'perception_source_sim_age_s'} <= set(row)):
                raise ValueError('Malformed or incompatible trace row')
            if not order or order[-1] != row['seed']:
                order.append(row['seed'])
            rows[row['seed']].append(row)
    if order != seeds:
        raise ValueError('Trace seed order differs from report')
    for episode in report['episodes']:
        values = rows[episode['seed']]
        if ([r['step'] for r in values] != list(range(len(values)))
                or sum(r['executed'] for r in values) != episode['steps']
                or any(not r['executed'] for r in values[:-1])
                or any(r['executed'] and r.get('scoring_only') is None for r in values)):
            raise ValueError('Trace is incomplete or has incorrect tick order')
        executed = [r for r in values if r['executed']]
        if (executed and report['instruction'] == FULL_INSTRUCTION
                and executed[-1]['scoring_only']['after']['physical_success'] != episode['success']):
            raise ValueError('Report success disagrees with physical scoring trace')


def paired_results(baseline, candidate):
    b, c = baseline['episodes'], candidate['episodes']
    if [r['seed'] for r in b] != [r['seed'] for r in c]:
        raise ValueError('Paired episode seeds differ')
    rescued, regressed, by_failure = [], [], {}
    for old, new in zip(b, c):
        category = old['failure_reason'] or 'baseline_success'
        if category.startswith('perception_rejected:'):
            category = 'perception_rejected'
        group = by_failure.setdefault(category, {'episodes': 0, 'rescues': 0, 'regressions': 0})
        group['episodes'] += 1
        entry = {'seed': old['seed'], 'baseline_failure_reason': old['failure_reason'],
                 'candidate_failure_reason': new['failure_reason'], 'candidate_final_phase': new['phase'],
                 'candidate_contact_audit': new.get('control_audit')}
        if not old['success'] and new['success']:
            rescued.append(entry); group['rescues'] += 1
        elif old['success'] and not new['success']:
            regressed.append(entry); group['regressions'] += 1
    r, g = len(rescued), len(regressed)
    n = r+g
    p = min(1., 2*sum(math.comb(n, k) for k in range(min(r, g)+1))/2**n) if n else 1.
    return {'rescued_episodes': r, 'regressed_episodes': g, 'net_success_delta': r-g,
            'rescued': rescued, 'regressed': regressed, 'by_baseline_failure': by_failure,
            'rescues_in_transfer_descend_retreat_timeouts': sum(
                by_failure.get('visual_timeout_'+name, {}).get('rescues', 0)
                for name in ('transfer', 'descend', 'retreat')),
            'mcnemar_exact_p_two_sided': p}


def aggregate_report(report):
    episodes = report['episodes']
    audits = [row['control_audit'] for row in episodes if row.get('control_audit') is not None]
    durations = {}
    for phase in PHASES:
        key = str(PHASES.index(phase))
        values = [a['phase_duration_sim_s'][key] for a in audits if key in a['phase_duration_sim_s']]
        reached = sum(key in (e.get('control_audit') or {}).get('phase_steps', {})
                      or any(t['to'] == phase for t in e.get('phase_transitions', [])) for e in episodes)
        durations[phase] = {'episodes_reached': reached, 'reach_rate': reached/len(episodes),
            'duration_sim_s_p50': float(np.median(values)) if values else None,
            'duration_sim_s_p95': float(np.percentile(values, 95)) if values else None}
    latencies = {}
    for event in ('floor_contact_to_hold', 'floor_contact_to_release_command', 'floor_contact_to_actual_open'):
        values = [a[event] for a in audits if a[event] is not None]
        latencies[event] = {'observed_episodes': len(values),
                            **{kind+'_p95': float(np.percentile([v[kind] for v in values], 95)) if values else None
                               for kind in ('sim_s', 'wall_s')}}
    return {'successes': report['successes'], 'attempted': report['attempted'],
            'success_rate': report['success_rate'], 'wilson_95': wilson(report['successes'], report['attempted']),
            'fallback_predictions': report['fallback_predictions'], 'fallback_episodes': report['fallback_episodes'],
            'perception_rejections': report['perception_rejections'],
            'failures': dict(Counter(r['failure_reason'] for r in episodes if not r['success'])),
            'phase_statistics': durations, 'latency': latencies,
            'deadline_misses': report['deadline_misses'], 'control_ticks': report['control_ticks'],
            'deadline_miss_fraction': report['deadline_miss_fraction'],
            **{key: sum(a[key] for a in audits) for key in
               ('false_contact_triggers', 'release_commands_outside_env_envelope',
                'actual_openings_outside_env_envelope', 'contact_trigger_events')}}


def validate_inputs(folder):
    plan_path = folder/'plan.json'
    digest = file_hash(plan_path)
    if json.loads((folder/'plan.seal.json').read_text())['plan_sha256'] != digest:
        raise ValueError('Plan changed after preparation')
    plan = json.loads(plan_path.read_text())
    if plan.get('schema') != SCHEMA:
        raise ValueError('Prepare a v3 controller plan; historical plans are read-only artifacts')
    if source_hash() != plan['source_sha256']:
        raise ValueError('Source changed after plan preparation; prepare a new plan')
    paths = {plan[key]: plan[key+'_sha256'] for key in ('policy', 'grounding')}
    paths.update(plan['datasets_sha256'])
    paths[plan['config']['path']] = plan['config']['sha256']
    if plan['seed_file']:
        paths[plan['seed_file']['path']] = plan['seed_file']['sha256']
    paths.update({ref['path']: ref['sha256'] for ref in plan['references'].values()})
    for path, expected in paths.items():
        if file_hash(path) != expected:
            raise ValueError(f'Input changed after preparation: {path}')
    return plan, digest


def run(output):
    folder = Path(output)
    plan, digest = validate_inputs(folder)
    summaries, raw = {}, {}
    names = [plan['baseline']] + [name for name in plan['variants'] if name != plan['baseline']]
    for variant in names:
        summaries[variant], raw[variant] = {}, {}
        variant_dir = folder/variant
        variant_dir.mkdir(exist_ok=True)
        for profile, seeds in plan['sets'].items():
            path, trace = variant_dir/(profile+'.json'), variant_dir/(profile+'.jsonl')
            receipt_path = variant_dir/(profile+'.receipt.json')
            options = plan['variants'][variant]
            if path.exists():
                if not receipt_path.exists():
                    raise ValueError('Incomplete run: report without a completion receipt')
                receipt = json.loads(receipt_path.read_text())
                if receipt != {'plan_sha256': digest, 'report_sha256': file_hash(path), 'trace_sha256': file_hash(trace)}:
                    raise ValueError('Report/trace checksum or plan mismatch on resume')
                report = json.loads(path.read_text())
            else:
                if trace.exists() or receipt_path.exists():
                    raise ValueError('Incomplete run: use a new output directory; existing traces will not be overwritten')
                args = visual_parser().parse_args([
                    '--grounding', plan['grounding'], '--policy', plan['policy'],
                    '--instruction', plan['instruction'], '--profile', profile,
                    '--output', str(path), '--trace', str(trace), '--scoring-diagnostics'])
                for group in options.values():
                    for key, value in group.items():
                        setattr(args, key, value)
                args.seeds, args.suite_variant = seeds, variant
                args.suite_plan_sha256, args.suite_source_sha256 = digest, plan['source_sha256']
                args.diagnostic_dataset = plan['diagnostic_dataset']
                args.startup_hold_steps=plan.get('startup_hold_steps',0)
                args.observation_age_clock=plan.get('observation_age_clock','both')
                report = evaluate(args)
            validate_report(plan, profile, report, digest, variant)
            validate_trace(trace, report)
            if variant == plan['baseline'] and profile in plan['references']:
                reference = plan['references'][profile]['episodes']
                if any(old.get(key) != new.get(key) for old, new in zip(reference, report['episodes'])
                       for key in ('seed', 'success', 'steps', 'phase', 'failure_reason')):
                    raise ValueError('Baseline replay drifted from its frozen reference; investigate before paired analysis')
            if not receipt_path.exists():
                write_json(receipt_path, {'plan_sha256': digest, 'report_sha256': file_hash(path),
                                          'trace_sha256': file_hash(trace)})
            summaries[variant][profile] = aggregate_report(report)
            raw[variant][profile] = report
    paired = {variant: {profile: paired_results(raw[plan['baseline']][profile], report)
                        for profile, report in reports.items()}
              for variant, reports in raw.items() if variant != plan['baseline']}
    gates = {variant: {'iid_observed_80_gate': reports['iid']['success_rate'] >= .8,
                       'iid_lower_95_bound_80_gate': reports['iid']['wilson_95'][0] >= .8}
             if 'iid' in reports else None for variant, reports in summaries.items()}
    result = {'schema': 'omni-visual-suite-summary-v3', 'plan_sha256': digest,
              'purpose': plan['purpose'], 'baseline': plan['baseline'], 'variants': summaries,
              'paired': paired, 'gates': gates,
              'limitations': 'Replay/development results are not fresh generalization evidence. '
                             'Contact oracle is scoring-only. Missing calibrated residual/latency stays unknown.'}
    validate_inputs(folder)  # Also catch input/source changes while physics was running.
    summary = folder/'summary.json'
    if summary.exists():
        if json.loads(summary.read_text()) != result:
            raise ValueError('Existing summary differs from verified reports')
    else:
        write_json(summary, result)
    print(json.dumps(result, indent=2))
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['prepare', 'run'])
    p.add_argument('--output', required=True)
    p.add_argument('--variants-config', default='configs/p5_controller_h1.json')
    p.add_argument('--purpose', choices=['development', 'regression_replay', 'holdout'], default='development')
    p.add_argument('--seed', type=int, default=90000)
    p.add_argument('--startup-hold-steps',type=int,default=0,
                   help='Common first-episode action prefix, pinned in plan/report; does not relax age limits')
    p.add_argument('--observation-age-clock',choices=['both','simulation'],default='both',
                   help='Pin common freshness clock; simulation isolates accelerated physics from host pauses')
    p.add_argument('--seed-file', help='JSON object PROFILE: [ordered seeds]')
    p.add_argument('--reference-report', action='append', help='Explicit PROFILE=REPORT authorizing replay')
    p.add_argument('--additional-training-dataset', action='append')
    p.add_argument('--iid-episodes', type=int, default=100)
    p.add_argument('--ood-episodes', type=int, default=20)
    p.add_argument('--profiles', nargs='+', choices=PROFILES)
    p.add_argument('--policy', default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--grounding', default='weights/p5_student/best.pt')
    p.add_argument('--instruction', default=FULL_INSTRUCTION)
    p.add_argument('--diagnostic-dataset', default='data/drawer_dagger_v2.h5')
    return p


def main():
    p = parser()
    args = p.parse_args()
    try:
        prepare(args) if args.action == 'prepare' else run(args.output)
    except (ValueError, NotImplementedError) as error:
        p.error(str(error))


if __name__ == '__main__':
    main()
