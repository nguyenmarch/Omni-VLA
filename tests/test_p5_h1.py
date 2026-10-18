import copy
import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from scripts import evaluate_visual_suite as suite
from simulation.vision import VisualTaskManager, FULL_INSTRUCTION
from utils.checkpoint import file_hash
from utils.reporting import write_json, wilson
from utils.visual_diagnostics import trace_row
from utils.visual_protocol import effective_options, normalize_options, object_hash, option_report_fields
from utils.visual_scoring import EpisodeAudit, scoring_snapshot


@pytest.fixture
def harness(tmp_path, monkeypatch):
    dataset = tmp_path/'training.h5'
    with h5py.File(dataset, 'w') as f:
        f.create_group('episodes/train').attrs['seed'] = 1
    policy, grounding = tmp_path/'bc.pt', tmp_path/'p5.pt'
    policy.write_bytes(b'bc'); grounding.write_bytes(b'p5')
    state = SimpleNamespace(used=set(), source='source', calls=[], count=0)
    monkeypatch.setattr(suite, 'TRAINING_DATASETS', (str(dataset),))
    monkeypatch.setattr(suite, 'historical_seeds', lambda: state.used)
    monkeypatch.setattr(suite, 'source_hash', lambda: state.source)

    def prepare(sets=None, purpose='development', references=None, config=None):
        state.count += 1
        folder = tmp_path/str(state.count)
        folder.mkdir()
        config = config or {'schema': 'omni-visual-variants-v1', 'baseline': 'B',
                            'variants': {'B': {}, 'T': {'controller_options': {'transfer_tolerance': .025}}}}
        write_json(folder/'variants.json', config)
        write_json(folder/'seeds.json', sets or {'iid': [10, 11]})
        argv = ['prepare', '--output', str(folder/'run'), '--variants-config', str(folder/'variants.json'),
                '--seed-file', str(folder/'seeds.json'), '--purpose', purpose,
                '--policy', str(policy), '--grounding', str(grounding)]
        for profile, path in (references or {}).items():
            argv.extend(['--reference-report', profile+'='+str(path)])
        args = suite.parser().parse_args(argv)
        args.diagnostic_dataset = None
        return Path(args.output), suite.prepare(args)

    def fake_evaluate(args):
        state.calls.append((args.suite_variant, args.profile, effective_options(args)))
        options = effective_options(args)
        episodes, ticks = [], []
        for seed in args.seeds:
            success = seed % 2 == 0
            manager = VisualTaskManager([[.4, 0., .55], [.5, 0., .6]])
            manager.phase = 4
            robot = {'q': np.zeros(14), 'dq': np.zeros(14), 'ee': np.zeros(6),
                     'gripper': np.zeros(4), 'wrench': np.zeros(12)}
            row = trace_row(manager, robot, np.zeros(14), seed, 0, 0., 0., ee_velocity=np.zeros((2, 3)))
            scoring = {'bottle_floor_contact': False, 'inside_env_placement_envelope': True,
                       'physical_success': success}
            row.update(executed=True, transition_reason=None,
                       scoring_only={'before': scoring, 'after': scoring})
            ticks.append(row)
            audit = EpisodeAudit()
            audit.record(4, scoring, scoring, np.zeros(14), np.zeros(4), np.zeros(4),
                         1, 0., .04, 0., .001)
            audit.timing(.001, .04)
            episodes.append({'seed': seed, 'success': success, 'steps': 1,
                             'phase': 'close' if success else 'transfer',
                             'failure_reason': None if success else 'visual_timeout_transfer',
                             'control_audit': audit.summary(.04)})
        Path(args.trace).write_text(''.join(json.dumps(row)+'\n' for row in ticks))
        count = sum(e['success'] for e in episodes)
        report = {'seeds_requested': args.seeds, 'attempted': len(episodes), 'episodes': episodes,
                  'successes': count, 'success_rate': count/len(episodes), 'wilson_95': wilson(count, len(episodes)),
                  'profile': args.profile, 'instruction': args.instruction,
                  'policy_sha256': file_hash(args.policy), 'grounding_sha256': file_hash(args.grounding),
                  'effective_options': options, 'options_sha256': object_hash(options),
                  'suite_plan_sha256': args.suite_plan_sha256, 'suite_variant': args.suite_variant,
                  'suite_source_sha256': args.suite_source_sha256, 'loop': False,
                  'stop_reason': 'episode_limit', 'scoring_diagnostics': True,
                  'trace_sha256': file_hash(args.trace), 'fallback_predictions': 0, 'fallback_episodes': 0,
                  'perception_rejections': 0, 'deadline_misses': 0, 'control_ticks': len(episodes),
                  'deadline_miss_fraction': 0., **option_report_fields(options)}
        write_json(args.output, report)
        return report

    monkeypatch.setattr(suite, 'evaluate', fake_evaluate)
    state.prepare = prepare
    state.policy, state.grounding = str(policy), str(grounding)
    return state


def test_variants_are_explicit_and_differ_in_one_controller_option(harness):
    folder, plan = harness.prepare()
    b, t = plan['variants']['B'], plan['variants']['T']
    assert b['perception_options'] == t['perception_options']
    assert not b['perception_options']['legacy_perception']
    assert [k for k in b['controller_options'] if b['controller_options'][k] != t['controller_options'][k]] == ['transfer_tolerance']
    result = suite.run(folder)
    assert len(harness.calls) == 2
    assert harness.calls[1][2] == t
    assert result['paired']['T']['iid']['rescued_episodes'] == 0
    assert suite.run(folder) == result
    assert len(harness.calls) == 2  # Resume verifies files without rerunning physics.


@pytest.mark.parametrize('mutation', ['config', 'plan', 'source', 'report', 'trace', 'missing_receipt',
                                     'checkpoint', 'dataset', 'seed_file'])
def test_resume_refuses_changed_inputs_and_outputs(harness, mutation):
    folder, plan = harness.prepare()
    suite.run(folder)
    calls = len(harness.calls)
    if mutation == 'config':
        Path(plan['config']['path']).write_text('{}')
    elif mutation == 'plan':
        value = json.loads((folder/'plan.json').read_text()); value['purpose'] = 'holdout'
        (folder/'plan.json').write_text(json.dumps(value))
    elif mutation == 'source':
        harness.source = 'different'
    elif mutation in ('report', 'trace'):
        path = folder/'B'/('iid.json' if mutation == 'report' else 'iid.jsonl')
        path.write_text(path.read_text()+' ')
    elif mutation == 'missing_receipt':
        (folder/'B/iid.receipt.json').unlink()
    elif mutation == 'checkpoint':
        Path(plan['policy']).write_bytes(b'changed checkpoint')
    elif mutation == 'dataset':
        Path(next(iter(plan['datasets_sha256']))).write_bytes(b'changed training data')
    else:
        Path(plan['seed_file']['path']).write_text('{"iid": [9]}')
    with pytest.raises(ValueError):
        suite.run(folder)
    assert len(harness.calls) == calls


def test_report_requires_complete_ordered_unique_physical_episodes(harness):
    folder, plan = harness.prepare()
    suite.run(folder)
    report = json.loads((folder/'B/iid.json').read_text())
    digest = file_hash(folder/'plan.json')
    mutations = []
    for rows in [report['episodes'][:1], report['episodes'][::-1], [report['episodes'][0]]*2]:
        mutations.append({**report, 'episodes': rows})
    mutations.extend([{**report, 'effective_options': plan['variants']['T']},
                      {**report, 'startup_hold_steps': 5},
                      {**report, 'observation_age_clock': 'simulation'},
                      {**report, 'success_rate': 1.}, {**report, 'place_contact_mode': 'raw_debug'},
                      {**report, 'stop_reason': 'keyboard_interrupt'}])
    for value in mutations:
        with pytest.raises(ValueError, match='incompatible or incomplete'):
            suite.validate_report(plan, 'iid', value, digest, 'B')


def test_holdout_disallows_history_and_replay_requires_matching_reference(harness, tmp_path):
    harness.used = {7}
    with pytest.raises(ValueError, match='prior'):
        harness.prepare(sets={'iid': [7]}, purpose='holdout')
    with pytest.raises(ValueError, match='reference'):
        harness.prepare(sets={'iid': [7]}, purpose='regression_replay')
    with pytest.raises(ValueError, match='training'):
        harness.prepare(sets={'iid': [1]}, purpose='regression_replay')
    reference = {'profile': 'iid', 'attempted': 1, 'successes': 0, 'instruction': FULL_INSTRUCTION,
                 'episodes': [{'seed': 7, 'success': False, 'steps': 1, 'phase': 'transfer',
                               'failure_reason': 'visual_timeout_transfer'}],
                 'policy_sha256': file_hash(harness.policy), 'grounding_sha256': file_hash(harness.grounding),
                 **option_report_fields(normalize_options())}
    path = tmp_path/'reference.json'
    write_json(path, reference)
    folder, plan = harness.prepare(sets={'iid': [7]}, purpose='regression_replay', references={'iid': path})
    assert plan['references']['iid']['episodes'] == reference['episodes']
    suite.run(folder)
    with pytest.raises(ValueError, match='Holdout'):
        harness.prepare(sets={'iid': [7]}, purpose='holdout', references={'iid': path})
    with pytest.raises(ValueError, match='incomplete or incompatible'):
        harness.prepare(sets={'iid': [7, 8]}, purpose='regression_replay', references={'iid': path})


def test_training_seed_cannot_be_whitelisted_by_reference(harness, tmp_path):
    # Training validation runs before references; explicit replay is never a bypass.
    with pytest.raises(ValueError, match='training'):
        harness.prepare(sets={'iid': [1]}, references={'iid': tmp_path/'nonexistent.json'})


def test_position_only_suite_has_no_iid_gate(harness):
    folder, _ = harness.prepare(sets={'position': [10]})
    result = suite.run(folder)
    assert result['gates']['B'] is None


def test_partial_trace_is_not_overwritten(harness):
    folder, _ = harness.prepare()
    (folder/'B').mkdir()
    trace = folder/'B/iid.jsonl'
    trace.write_text('unfinished trace')
    with pytest.raises(ValueError, match='Incomplete run'):
        suite.run(folder)
    assert not harness.calls and trace.read_text() == 'unfinished trace'


def test_baseline_drift_stops_before_candidate_analysis(harness, tmp_path):
    reference = {'profile': 'iid', 'attempted': 1, 'successes': 1, 'instruction': FULL_INSTRUCTION,
                 'episodes': [{'seed': 10, 'success': True, 'steps': 2, 'phase': 'close', 'failure_reason': None}],
                 'policy_sha256': file_hash(harness.policy), 'grounding_sha256': file_hash(harness.grounding),
                 **option_report_fields(normalize_options())}
    path = tmp_path/'drift_reference.json'
    write_json(path, reference)
    folder, _ = harness.prepare(sets={'iid': [10]}, purpose='regression_replay', references={'iid': path})
    with pytest.raises(ValueError, match='Baseline replay drifted'):
        suite.run(folder)
    assert len(harness.calls) == 1
    assert not (folder/'summary.json').exists()


def test_paired_rescue_requires_final_success_and_keeps_failure_categories():
    def episode(seed, success, phase):
        return {'seed': seed, 'success': success, 'phase': phase,
                'failure_reason': None if success else 'visual_timeout_'+phase}
    baseline = {'episodes': [episode(1, False, 'transfer'), episode(2, False, 'descend'),
                             episode(3, True, 'close')]}
    candidate = {'episodes': [episode(1, False, 'retreat'), episode(2, True, 'close'),
                              episode(3, False, 'retreat')]}
    result = suite.paired_results(baseline, candidate)
    assert result['rescued_episodes'] == result['regressed_episodes'] == 1
    assert result['rescued'][0]['seed'] == 2 and result['regressed'][0]['seed'] == 3
    assert result['net_success_delta'] == 0
    assert result['by_baseline_failure']['visual_timeout_transfer']['rescues'] == 0
    assert result['rescues_in_transfer_descend_retreat_timeouts'] == 1


def test_trace_rotates_wrench_and_keeps_unknown_residual_and_fallback_age():
    manager = VisualTaskManager([[.4, 0., .55], [.5, 0., .6]], place_contact_mode='raw_debug')
    rotation = np.array([[0., 0., -1.], [0., 1., 0.], [1., 0., 0.]])
    robot = {'q': np.zeros(14), 'dq': np.zeros(14), 'ee': np.zeros(6),
             'gripper': np.zeros(4), 'wrench': np.r_[9., 0., 0., np.zeros(9)],
             'wrist_rotations': np.stack([rotation, np.eye(3)]), 'object': np.full(7, np.nan)}
    row = trace_row(manager, robot, np.zeros(14), 1, 0, .16, .2,
                    before_override=np.ones(14), ee_velocity=np.zeros((2, 3)),
                    metadata={'fallback_used': True, 'fallback_age_s': .2})
    assert row['wrench_world'][0][2] == pytest.approx(9.)
    assert row['wrench_residual_world'] is None and row['calibration_status'] == 'unavailable_H3'
    assert row['perception_source_sim_age_s'] == pytest.approx(.36)
    assert 'object' not in row and 'scoring_only' not in row
    assert row['action_before_override'] != row['action']


def test_contact_audit_does_not_treat_every_air_release_as_invalid():
    audit = EpisodeAudit()
    score = {'bottle_floor_contact': False, 'inside_env_placement_envelope': True}
    audit.record(6, score, score, np.r_[np.zeros(6), 1., np.zeros(7)],
                 np.zeros(4), np.ones(4)*.04, 1, 0., .04, 0., .01,
                 transition={'reason': 'wrench_contact'})
    result = audit.summary(.04)
    assert result['false_contact_triggers'] == 1
    assert result['release_commands_outside_env_envelope'] == 0
    assert result['actual_openings_outside_env_envelope'] == 0
    assert result['floor_contact_to_release_command'] is None


def test_scoring_snapshot_uses_actual_contact_force_and_does_not_mutate_task_flags():
    from utils.config import make_environment
    from simulation.vision import robot_observation
    from simulation.visual_runtime import hold_action
    env = make_environment()
    try:
        obs, _ = env.reset(seed=76000)
        env.data.qpos[env.bottle_adr:env.bottle_adr+3] = [.68, .10, .486]
        for _ in range(3):
            obs, *_ = env.step(hold_action(robot_observation(obs)))
        flags = env.opened, env.placed
        value = scoring_snapshot(env)
        assert (env.opened, env.placed) == flags
        assert value['bottle_floor_contact']
        assert any(row['normal_force_sum_n'] > 0 and 'drawer_floor' in row['geoms']
                   for row in value['contacts'])
    finally:
        env.close()


@pytest.mark.parametrize('controller', [{'unknown': 1}, {'transfer_tolerance': float('nan')},
                                        {'place_contact_debounce_steps': 1.5}, {'slip_monitor': 1}])
def test_invalid_options_fail_before_any_simulation(controller):
    with pytest.raises(ValueError):
        normalize_options(controller=controller)
