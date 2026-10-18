import numpy as np
import pytest

from simulation.transfer import TerminalTransfer
from simulation.vision import VisualTaskManager
from utils.visual_protocol import normalize_options


def situation(**options):
    servo = TerminalTransfer({'transfer_mode': 'terminal', **options})
    goals = np.array([[.69, .10, .72], [.295, .10, .51]])
    robot = {'ee': (goals+[[-.02, 0., 0.], [0., 0., 0.]]).ravel(),
             'ee_velocity': np.zeros((2, 3)), 'ee_rotations': np.tile(np.eye(3), (2, 1, 1)),
             'object': np.full(7, np.nan)}
    points = np.array([[.69, .1, .68], [.50, .1, .51]])
    return servo, robot, goals, points


def test_servo_is_bounded_in_env_delta_convention_and_keeps_grips_closed():
    servo, robot, goals, _ = situation()
    policy = np.ones(14)
    last_v, last_w = np.zeros((2, 3)), np.zeros((2, 3))
    for _ in range(8):
        action = servo.command(policy, robot, goals, .04, True, .1)
        v = action.reshape(2, 7)[:, :3]*.015/.04
        w = action.reshape(2, 7)[:, 3:6]*.08/.04
        assert np.linalg.norm(v, axis=1).max() <= .15+1e-7
        assert np.linalg.norm(v-last_v, axis=1).max() <= .75*.04+1e-7
        assert np.linalg.norm(w-last_w, axis=1).max() <= 2.*.04+1e-7
        np.testing.assert_array_equal(action[[6, 13]], [-1., -1.])
        last_v, last_w = v, w
    assert servo.state == 'settle'


def test_readiness_requires_multiple_good_ticks_and_reset_after_stale_or_fast_motion():
    servo, robot, goals, points = situation()
    servo.command(np.zeros(14), robot, goals, .04, True, .1)
    robot['ee'] = goals.ravel()
    assert not servo.observe(robot, goals, points, True, .04, True, .1)
    robot['ee_velocity'][0, 0] = .1
    assert not servo.observe(robot, goals, points, True, .04, True, .1)
    assert servo.stable == 0
    robot['ee_velocity'][:] = 0.
    assert not servo.observe(robot, goals, points, True, .04, True, .5)
    for _ in range(2):
        assert not servo.observe(robot, goals, points, True, .04, True, .1)
    assert servo.observe(robot, goals, points, True, .04, True, .1)
    assert servo.state == 'ready_to_place'


@pytest.mark.parametrize('blocked', ['clearance', 'orientation', 'opening', 'goal_jump'])
def test_terminal_transfer_cannot_advance_without_geometry_guards(blocked):
    servo, robot, goals, points = situation()
    servo.command(np.zeros(14), robot, goals, .04, True, .1)
    robot['ee'] = goals.ravel()
    opened = True
    if blocked == 'clearance':
        points[0, 2] = .55
    elif blocked == 'orientation':
        robot['ee_rotations'][0] = np.diag([1., -1., -1.])
    elif blocked == 'opening':
        opened = False
    else:
        goals = goals.copy(); goals[:, 0] += .1
    for _ in range(5):
        assert not servo.observe(robot, goals, points, opened, .04, True, .1)
    assert servo.state == 'settle'


def test_stale_frame_holds_without_opening_and_settle_budget_never_forces_place():
    servo, robot, goals, points = situation(transfer_settle_budget_steps=4, transfer_stable_steps=2)
    servo.command(np.ones(14), robot, goals, .04, True, .1)
    for _ in range(4):
        action = servo.command(np.ones(14), robot, goals, .04, False, None)
        np.testing.assert_array_equal(action.reshape(2, 7)[:, :6], 0.)
        assert not servo.observe(robot, goals, points, True, .04, False, None)
    assert servo.failure_reason == 'visual_transfer_settle_timeout'
    assert servo.state != 'ready_to_place'


def test_no_progress_is_distinct_from_success_and_reset_discards_servo_state():
    servo, robot, goals, points = situation(transfer_no_progress_steps=2)
    servo.command(np.zeros(14), robot, goals, .04, True, .1)
    for _ in range(3):
        assert not servo.observe(robot, goals, points, True, .04, True, .1)
    assert servo.failure_reason == 'visual_transfer_settle_no_progress'
    servo.reset()
    assert servo.state == 'bc_transfer' and servo.reference_rotations is None
    assert servo.stable == servo.age == 0 and servo.failure_reason == ''


def test_manager_mode_is_opt_in_and_input_schema_stays_67d():
    points = np.array([[.69, .1, .68], [.7, .1, .51]])
    legacy = VisualTaskManager(points)
    terminal = VisualTaskManager(points, transfer_options={'transfer_mode': 'terminal'})
    robot = {'q': np.zeros(14), 'dq': np.zeros(14), 'ee': np.zeros(6),
             'gripper': np.zeros(4), 'wrench': np.zeros(12)}
    assert legacy.terminal_transfer is None
    assert terminal.condition(robot).shape == (67,)
    terminal.terminal_transfer.state = 'settle'
    terminal.reset_episode(points)
    assert terminal.terminal_transfer.state == 'bc_transfer'
    with pytest.raises(ValueError, match='finite EE pose'):
        terminal.terminal_transfer.command(np.zeros(14), robot, np.ones((2, 3)), .04, True, .1)
    with pytest.raises(ValueError):
        normalize_options(controller={'transfer_ready_tolerance_m': .04})


def test_loaded_handle_pose_and_bc_preload_are_retained_on_already_good_axes():
    servo, robot, goals, _ = situation()
    position = np.asarray(robot['ee']).reshape(2, 3)
    position[1, 0] += .006
    policy = np.zeros(14); policy[7] = -.1; policy[[6, 13]] = -1.
    servo.command(policy, robot, goals, .04, True, .1)
    assert not servo.position_mask[1].any()
    assert servo.reference_positions[1, 0] == pytest.approx(position[1, 0])
    assert servo.feedforward[1, 0] == pytest.approx(-.1)
    action = servo.command(policy, robot, goals, .04, True, .1)
    assert action[7] < 0.
    with pytest.raises(ValueError, match='action delta'):
        servo.command(policy, robot, goals, .2, True, .1)


def test_manager_checks_current_drawer_opening_even_after_it_was_previously_open():
    manager = VisualTaskManager([[.72, .1, .68], [.7, .1, .51]],
                                transfer_options={'transfer_mode': 'terminal'})
    manager.phase = 4
    manager.opened = True
    manager.observe([[.72, .1, .68], [.53, .1, .51]])
    goals = np.stack(manager.goals())
    robot = {'ee': (goals+[[-.02, 0., 0.], [0., 0., 0.]]).ravel(),
             'ee_velocity': np.zeros((2, 3)), 'ee_rotations': np.tile(np.eye(3), (2, 1, 1)),
             'gripper': np.ones(4)*.02, 'wrench': np.zeros(12)}
    manager.adapt_action(np.zeros(14), robot, .04, True, .1)
    # Drawer opening falls from 17 cm to 15 cm, within the goal jump limit.
    manager.observe([[.74, .1, .68], [.55, .1, .51]])
    robot['ee'] = np.stack(manager.goals()).ravel()
    for _ in range(4):
        manager.update(robot, .04, True, .1)
    assert manager.opened  # Historical task flag remains true.
    assert manager.phase == 4
    assert not manager.terminal_transfer.guards['clearance_ready']
