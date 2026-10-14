import json
from pathlib import Path
from threading import Event
import time
import zipfile
import hashlib
import h5py
from types import SimpleNamespace

import numpy as np
import pytest

from simulation.visual_runtime import LatestWorker, Request, Result, usable, hold_action, SnapshotPerception, snapshot
from simulation.recovery import WrenchSlipMonitor
from simulation.recovery import RecoveryExecutor, RecoveryState
from simulation.randomization import DomainRandomizer
from scripts.evaluate_visual_suite import validate_seeds, validate_report
from scripts.artifact_bundle import verify_restore
from utils.config import make_environment
from utils.reporting import write_json, wilson
from utils.visual_diagnostics import TrainingEnvelope


def request(episode, step):
    return Request(episode, step, step*.04, time.perf_counter(), np.zeros(1))


def test_generation_reset_and_latest_pending_are_bounded():
    running, release, processed = Event(), Event(), []
    def process(value):
        processed.append((value.episode, value.step))
        if value.step == 0:
            running.set()
            assert release.wait(2)
        return value.step
    worker = LatestWorker(process)
    try:
        worker.reset(1)
        worker.submit(request(1, 0))
        assert running.wait(2)
        worker.submit(request(1, 1))
        worker.submit(request(1, 2))
        worker.reset(2)
        worker.submit(request(2, 3))
        release.set()
        deadline = time.perf_counter()+2
        result = None
        while result is None and time.perf_counter() < deadline:
            result = worker.poll()
            time.sleep(.001)
        assert result.request.episode == 2 and result.value == 3
        assert processed == [(1, 0), (2, 3)]
        assert worker.replaced == 1 and worker.discarded == 1
    finally:
        release.set()
        worker.close()


def test_age_checks_cover_episode_wall_sim_and_future_results():
    r = Result(Request(2, 0, 1., 10., np.zeros(1)))
    assert usable(r, 2, 1.1, .4, .4, now=10.1)
    assert not usable(r, 3, 1.1, .4, .4, now=10.1)
    assert not usable(r, 2, 1.5, .4, .4, now=10.1)
    assert not usable(r, 2, 1.1, .4, .4, now=10.5)
    assert not usable(r, 2, .9, .4, .4, now=10.1)
    action = hold_action({'gripper': np.array([.025, .025, .04, .04])})
    np.testing.assert_allclose(action[[6, 13]], [.25, 1.])
    np.testing.assert_array_equal(action[[0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]], 0)


def test_snapshot_renderer_runs_on_worker_and_matches_main_capture():
    from simulation.vision import capture_head
    class Grounder:
        def predict(self, frame, ablation):
            return np.ones((2, 3)), {}
    env = make_environment(image_size=32)
    processor = SnapshotPerception(env, Grounder(), 'none')
    worker = LatestWorker(processor, processor.close)
    try:
        worker.reset(0)
        worker.submit(snapshot(env, 0, 0))
        deadline = time.perf_counter()+5
        result = None
        while result is None and time.perf_counter() < deadline:
            result = worker.poll()
            time.sleep(.001)
        assert result is not None and result.error is None
        expected = capture_head(env)
        actual = result.value[2]
        np.testing.assert_array_equal(actual['rgb'], expected['rgb'])
        np.testing.assert_allclose(actual['depth'], expected['depth'])
        np.testing.assert_allclose(actual['T'], expected['T'])
        assert processor.data is not env.data and processor.model is not env.model
    finally:
        worker.close()
        env.close()


def robot(force, width=.025, x=0.):
    return {'wrench': np.r_[force, np.zeros(9)], 'ee': np.array([x, 0., 0., 0., 0., 0.]),
            'gripper': np.array([width, width, .04, .04])}


def test_force_detector_requires_calibration_and_rejects_dynamic_spike():
    monitor = WrenchSlipMonitor(calibration_frames=2, debounce=3)
    assert monitor.update(robot([0, 0, 5]), True, .04)['state'] == 'uncalibrated'
    for _ in range(20):
        monitor.update(robot([0, 0, 5], .04), False, .04)
    for _ in range(30):
        row = monitor.update(robot([0, 0, 7]), True, .04)
    assert row['state'] == 'loaded' and not row['slip']
    assert not monitor.update(robot([0, 0, 5], .002, x=1), True, .04)['slip']
    for _ in range(30):
        row = monitor.update(robot([0, 0, 5], .002, x=1), True, .04)
    assert row['slip'] and row['finger_collapse']
    monitor.reset()
    assert monitor.update(robot([0, 0, 5]), True, .04)['state'] == 'uncalibrated'


def test_force_comparison_uses_world_frame():
    monitor = WrenchSlipMonitor(calibration_frames=2)
    for _ in range(10):
        monitor.update(robot([0, 0, 5], .04), False, .04)
    for _ in range(30):
        monitor.update(robot([0, 0, 7]), True, .04)
    rotated = robot([7, 0, 0])
    rotated['wrist_rotation'] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    for _ in range(10):
        assert not monitor.update(rotated, True, .04)['slip']


def test_placement_unloading_does_not_trigger_force_guard():
    monitor = WrenchSlipMonitor(calibration_frames=2)
    for _ in range(20):
        monitor.update(robot([0, 0, 5], .04), False, .04)
    for _ in range(30):
        monitor.update(robot([0, 0, 7]), True, .04)
    placed = robot([0, 0, 5], .002)
    placed['slip_detection_enabled'] = False
    for _ in range(30):
        assert not monitor.update(placed, True, .04)['slip']


def test_wrench_recovery_preempts_without_reading_oracle_position():
    env = SimpleNamespace(dt=.04, data=SimpleNamespace(
        site=lambda name: SimpleNamespace(xmat=np.eye(3).flatten())))
    executor = RecoveryExecutor(env, detector='wrench')
    def observation(force, width):
        return {**robot(force, width), 'object': np.full(7, np.nan)}
    for _ in range(20):
        executor.observe(observation([0, 0, 5], .04), 0)
    for _ in range(30):
        executor.observe(observation([0, 0, 7], .025), 2)
    for _ in range(30):
        state = executor.observe(observation([0, 0, 5], .002), 3)
    assert state == RecoveryState.HOLD and executor.stage == 'hold'
    assert executor.monitor.retries == 1


@pytest.mark.parametrize('inject', [False, True])
def test_force_guard_physical_carry_and_injected_release(inject):
    from simulation.expert import ScriptedExpert
    from simulation.vision import robot_observation
    env = make_environment()
    try:
        obs, _ = env.reset(seed=14501 if inject else 14500)
        expert = ScriptedExpert(env)
        monitor = WrenchSlipMonitor()
        injection_step = detection_step = None
        ticks = 0
        for step in range(1500):
            observed = robot_observation(obs)
            observed['object'] = np.full(7, np.nan)  # poison oracle input
            observed['wrist_rotation'] = env.data.site('left_wrist').xmat.reshape(3, 3).copy()
            observed['slip_detection_enabled'] = expert.phase in (3, 4)
            row = monitor.update(observed, expert.phase in (2, 3, 4, 5), env.dt)
            if row['slip'] and detection_step is None:
                detection_step = step
            action = expert.action()
            if inject and injection_step is None and expert.phase == 3 and obs['ee'][2] > .67:
                adr = env.model.joint('bottle_free').dofadr[0]
                env.data.qvel[adr:adr+3] = [.7, 0, -.7]
                injection_step, ticks = step, 18
            if ticks:
                action[6] = 1.
                ticks -= 1
            obs, _, done, truncated, info = env.step(action)
            expert.update()
            if done or truncated or expert.failed:
                break
        if inject:
            assert injection_step is not None
            assert injection_step <= detection_step <= injection_step+20
        else:
            assert info['success'] and detection_step is None
    finally:
        env.close()


def test_randomization_reproducible_and_iid_restores_model():
    env = make_environment()
    try:
        randomizer = DomainRandomizer(env)
        env.reset(seed=7)
        a = randomizer.apply(7, 'combined')
        cam, friction = env.model.cam_pos.copy(), env.model.geom_friction.copy()
        env.reset(seed=7)
        assert randomizer.apply(7, 'combined') == a
        np.testing.assert_array_equal(env.model.cam_pos, cam)
        np.testing.assert_array_equal(env.model.geom_friction, friction)
        randomizer.apply(7, 'iid')
        np.testing.assert_array_equal(env.model.cam_pos, randomizer.original['cam_pos'])
        np.testing.assert_array_equal(env.model.geom_friction, randomizer.original['geom_friction'])
    finally:
        env.close()


def test_reports_exclusive_and_seed_plan_disjoint(tmp_path):
    path = tmp_path/'report.json'
    write_json(path, {'success': True})
    with pytest.raises(FileExistsError):
        write_json(path, {'success': False})
    assert json.loads(path.read_text()) == {'success': True}
    assert wilson(42, 50)[0] < .8 < wilson(42, 50)[1]
    with pytest.raises(ValueError, match='Overlapping'):
        validate_seeds({'iid': [1], 'camera': [1]}, [])
    with pytest.raises(ValueError, match='prior'):
        validate_seeds({'iid': [1]}, [], historical=[1])


def test_distribution_diagnostics_fit_train_only(tmp_path):
    path = tmp_path/'data.h5'
    with h5py.File(path, 'w') as file:
        file.attrs['splits'] = json.dumps({'train': ['train'], 'val': ['val'], 'test': ['test']})
        for name, value in [('train', .1), ('val', 1000.), ('test', -1000.)]:
            condition = np.zeros((3, 67), np.float32)
            condition[:, :57] = value
            condition[:, 57] = 1.
            file.create_dataset('episodes/'+name+'/condition', data=condition)
    envelope = TrainingEnvelope(path)
    condition = np.full(67, .1, np.float32)
    assert envelope.measure(condition, 0)['outside_train_envelope_fraction'] == 0
    condition[:57] = 1000.
    assert envelope.measure(condition, 0)['outside_train_envelope_fraction'] == 1
    assert envelope.measure(condition, 1) is None


def test_suite_resume_rejects_incomplete_episode_rows_and_wrong_inputs():
    plan = {'sets': {'iid': [10, 11]}, 'policy_sha256': 'policy', 'grounding_sha256': 'student'}
    report = {'seeds_requested': [10, 11], 'attempted': 2, 'episodes': [{'seed': 10, 'success': True},
              {'seed': 11, 'success': False}], 'successes': 1, 'profile': 'iid',
              'policy_sha256': 'policy', 'grounding_sha256': 'student', 'suite_plan_sha256': 'plan'}
    validate_report(plan, 'iid', report, 'plan')
    with pytest.raises(ValueError, match='incomplete'):
        validate_report(plan, 'iid', {**report, 'episodes': report['episodes'][:1]}, 'plan')
    with pytest.raises(ValueError, match='incompatible'):
        validate_report(plan, 'iid', {**report, 'policy_sha256': 'wrong'}, 'plan')


def bundle(path, member, content=b'test', digest=None):
    manifest = {'schema': 'omni-artifact-bundle-v1', 'files': {member: {
        'size': len(content), 'sha256': digest or hashlib.sha256(content).hexdigest()}}}
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('manifest.json', json.dumps(manifest))
        archive.writestr(member, content)


def test_bundle_rejects_traversal_corruption_and_conflicts_before_restore(tmp_path):
    archive = tmp_path/'bundle.zip'
    bundle(archive, '../escape.pt')
    with pytest.raises(ValueError, match='Unsafe'):
        verify_restore(archive)
    bundle(archive, 'weights/model.pt', digest='0'*64)
    with pytest.raises(ValueError, match='checksum'):
        verify_restore(archive)
    bundle(archive, 'weights/model.pt')
    destination = tmp_path/'checkout'
    destination.mkdir()
    verify_restore(archive, destination)
    verify_restore(archive, destination)
    (destination/'weights/model.pt').write_bytes(b'conflict')
    with pytest.raises(FileExistsError):
        verify_restore(archive, destination)
    assert (destination/'weights/model.pt').read_bytes() == b'conflict'
