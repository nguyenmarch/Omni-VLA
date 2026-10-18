"""Contact sign/frame, calibration validity and false-support counterexamples."""
import copy

import numpy as np
import pytest

from models.impedance_controller import exp_rotation
from simulation.placement_contact import PlacementContactObserver
from utils.visual_protocol import normalize_options


def robot(world=(0., 0., 7.45, 0., .07, 0.), stage=0, rotation=None, width=.04):
    r = np.eye(3) if rotation is None else rotation
    return {'wrench': np.r_[r.T@world[:3], r.T@world[3:], np.zeros(6)],
            'ee': np.array([.45, .1, .72, .3, .1, .6]),
            'wrist_rotations': np.stack([r, np.eye(3)]),
            'ee_velocity': np.zeros((2, 3)), 'gripper': np.full(4, width),
            'placement_stage': stage}


def repeat(observer, sample, n=20):
    for _ in range(n):
        result = observer.update(sample, .04)
    return result


def calibrated(rotation=None):
    observer = PlacementContactObserver()
    repeat(observer, robot(rotation=rotation))
    loaded = robot((.1, -.1, 8.63, .01, .08, 0.), 4, rotation, .024)
    result = repeat(observer, loaded)
    assert result['calibration_status'] == 'calibrated'
    assert result['payload_n'] == pytest.approx(1.18)
    return observer, loaded


def support_sample(rotation=None):
    return robot((.1, -.1, 7.45, .01, .08, 0.), 5, rotation, .024)


def test_world_frame_and_negative_signed_support_residual():
    rotation = exp_rotation(np.array([.6, -.5, .3]))
    observer, loaded = calibrated(rotation)
    loaded['placement_stage'] = 5
    result = repeat(observer, loaded)
    assert result['wrench_world'][2] > 8.
    assert not result['support'] and result['load_loss_n'] == pytest.approx(0.)
    result = repeat(observer, support_sample(rotation))
    assert result['support'] and result['hold_requested']
    assert result['residual_world'][2] == pytest.approx(-1.18)
    assert result['support_confidence'] == pytest.approx(1.)


def test_hysteresis_debounce_and_stage_change_clear_support():
    observer, _ = calibrated()
    sample = support_sample()
    for _ in range(6):
        result = observer.update(sample, .04)
        assert not result['support']  # Filter fill plus a contiguous debounce.
    assert observer.update(sample, .04)['support']
    weak = robot((.1, -.1, 8.03, .01, .08, 0.), 5, width=.024)
    assert repeat(observer, weak)['support']  # 0.6 N: below on, above off.
    weak['wrench'][2] = 8.33
    assert not repeat(observer, weak)['support']
    weak['placement_stage'] = 6
    assert not repeat(observer, weak)['support']


@pytest.mark.parametrize('component,value', [(0, 2.), (3, .2)])
def test_side_force_or_moment_veto_latches_until_retraction(component, value):
    observer, _ = calibrated()
    sample = support_sample()
    sample['wrench'][component] = value
    result = repeat(observer, sample)
    assert result['collision'] and result['hold_requested'] and not result['support']
    # A side contact can later have only vertical friction; never relabel it as floor.
    result = repeat(observer, support_sample())
    assert result['collision'] and not result['support']
    observer.reset_episode()
    assert not observer.result['support'] and observer.loaded is None


def test_unknown_calibration_cannot_report_support_or_learn_from_contact():
    observer = PlacementContactObserver()
    result = repeat(observer, support_sample())
    assert result['calibration_status'] == 'uncalibrated_empty'
    assert not result['support'] and result['hold_requested']
    repeat(observer, robot())
    result = repeat(observer, support_sample())
    assert result['calibration_status'] == 'uncalibrated_payload'
    assert observer.loaded is None and not result['support']


def test_collision_veto_rearms_only_after_nominal_load_returns():
    observer, loaded = calibrated()
    sample = support_sample()
    sample['wrench'][0] = 2.
    assert repeat(observer, sample)['collision']
    assert repeat(observer, support_sample())['collision']
    loaded['placement_stage'] = 5
    result = repeat(observer, loaded)
    assert not result['collision'] and not result['support']
    assert repeat(observer, support_sample())['support']


def test_low_average_ee_speed_does_not_make_oscillating_wrist_load_support():
    observer, loaded = calibrated()
    loaded['placement_stage'] = 5
    # Arm-arm impacts can alternate low/high force while EE finite-difference
    # speed is small. A median alone invents a sustained unloading signal.
    for fz in [7.45, 7.45, 10.8, 7.45, 10.2]*8:
        loaded['wrench'][2] = fz
        result = observer.update(loaded, .04)
        assert not result['support']
    # Brief quiet low-load intervals between impacts are not stable contacts.
    for fz in [10.8, 7.45, 7.45, 7.45, 7.45]*4:
        loaded['wrench'][2] = fz
        assert not observer.update(loaded, .04)['support']


@pytest.mark.parametrize('change', ['nan', 'shape', 'rotation', 'saturated', 'missing'])
def test_bad_sensor_clears_reference_and_support(change):
    observer, _ = calibrated()
    assert repeat(observer, support_sample())['support']
    sample = support_sample()
    if change == 'nan':
        sample['wrench'][0] = np.nan
    elif change == 'shape':
        sample['gripper'] = np.zeros(1)
    elif change == 'rotation':
        sample['wrist_rotations'][0] = np.zeros((3, 3))
    elif change == 'saturated':
        sample['wrench'][0] = 201.
    else:
        del sample['wrench']
    result = observer.update(sample, .04)
    assert result['calibration_status'] == 'sensor_invalid'
    assert result['hold_requested'] and not result['support']
    assert not repeat(observer, support_sample())['support']


@pytest.mark.parametrize('change', ['velocity', 'acceleration', 'grip', 'orientation'])
def test_motion_grasp_and_orientation_bounds_invalidate_support(change):
    observer, _ = calibrated()
    sample = support_sample()
    if change == 'velocity':
        sample['ee_velocity'][0, 2] = -.1
    elif change == 'grip':
        sample['gripper'][:] = .001
    elif change == 'orientation':
        sample = support_sample(exp_rotation(np.array([.3, 0., 0.])))
    for i in range(12):
        if change == 'acceleration':
            sample['ee_velocity'][0, 2] = .04 if i % 2 else -.04
        result = observer.update(sample, .04)
        assert not result['support']


def test_reset_removes_all_calibration_and_only_robot_inputs_matter():
    observer, loaded = calibrated()
    twin = copy.deepcopy(observer)
    loaded['object'] = np.full(7, np.nan)
    loaded['scoring_only'] = {'bottle_floor_contact': True}
    assert observer.update(loaded, .04) == twin.update({k:v for k,v in loaded.items()
                                                     if k not in ('object', 'scoring_only')}, .04)
    observer.reset_episode()
    assert observer.empty is observer.loaded is observer.previous_velocity is None
    assert not observer.filtered and not observer.loaded_samples


@pytest.mark.parametrize('options', [{'placement_filter_frames': 2},
                                    {'placement_support_on_fraction': .2},
                                    {'placement_min_payload_n': 10},
                                    {'placement_lateral_limit_n': float('nan')}])
def test_invalid_observer_config_fails_before_rollout(options):
    with pytest.raises(ValueError):
        normalize_options(controller=options)


@pytest.mark.parametrize('startup,age_clock',[(0,'both'),(5,'both'),(5,'simulation')])
def test_visual_shadow_observer_samples_every_physics_tick_even_when_camera_stale(tmp_path, monkeypatch,startup,age_clock):
    import json
    import torch
    from types import SimpleNamespace
    from scripts import eval_visual_policy as visual
    from utils.config import make_environment

    class Grounder:
        image_size = 32

        def predict(self, frame, ablation):
            return np.array([[.45, -.18, .462], [.49, .10, .51]]), {}

    class Policy:
        def sample_euler(self, condition):
            return torch.zeros((1, 1, 14))

    checkpoint = {'policy_type': 'bc', 'model_config': {'chunk_size': 1}, 'action_dt': .04,
                  'stats': {'condition_mean': torch.zeros(67), 'condition_std': torch.ones(67),
                            'action_mean': torch.zeros(14), 'action_std': torch.ones(14)}}
    monkeypatch.setattr(visual, 'load_grounding_runtime', lambda path: Grounder())
    monkeypatch.setattr(visual, 'load_policy', lambda path: (Policy(), checkpoint))
    monkeypatch.setattr(visual, 'capture_head', lambda *a, **kw: {})
    monkeypatch.setattr(visual, 'make_environment', lambda **kw: make_environment(max_episode_steps=12))
    for name in ('bc.pt', 'p5.pt'):
        (tmp_path/name).write_bytes(b'fixture checkpoint')
    args = visual.parser().parse_args(['--grounding', str(tmp_path/'p5.pt'),
                                       '--policy', str(tmp_path/'bc.pt'), '--episodes', '1',
                                       '--placement-observer', '--max-sim-age', '.01',
                                       '--startup-hold-steps',str(startup),
                                       '--observation-age-clock',age_clock,
                                       '--max-wall-age','0.000000001' if age_clock=='simulation' else '.4',
                                       '--trace', str(tmp_path/'trace.jsonl'),
                                       '--output', str(tmp_path/'report.json')])
    report = visual.evaluate(args)
    assert sum(report['episodes'][0]['placement_observer_states'].values()) == 12
    rows = [json.loads(line) for line in (tmp_path/'trace.jsonl').read_text().splitlines()]
    stale = [r for r in rows if r.get('hold_reason') == 'stale_perception']
    assert stale and all(r['placement_after_update'] is not None for r in stale)
    assert all(r['phase_after'] == r['phase'] for r in stale)
    if startup:
        assert all(r['held'] and r['phase'] is None for r in rows[:startup])
    assert report['place_contact_mode'] == 'off'
    if age_clock=='simulation':assert report['wall_stale_results']>0
