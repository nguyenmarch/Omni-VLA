import numpy as np
import pytest

from scripts.eval_visual_policy import evaluate, parser
from simulation.vision import VisualTaskManager


def manager_and_robot(**options):
    manager = VisualTaskManager([[.4, 0., .55], [.5, 0., .6]], **options)
    manager.phase = 5
    left, right = manager.goals()
    robot = {'ee': np.r_[left+[0., 0., .04], right],
             'gripper': np.zeros(4), 'wrench': np.r_[0., 0., 1000., np.zeros(9)],
             'left_wrist_rotation': np.eye(3)}
    return manager, robot


@pytest.mark.parametrize('wrench', [np.full(12, np.nan), np.ones(12)*1e6, np.ones(1)])
def test_force_off_preserves_pose_release_regardless_of_wrench(wrench):
    manager, robot = manager_and_robot()
    robot['wrench'] = wrench
    for _ in range(25):
        manager.update(robot)
    assert manager.phase == 5
    assert manager.place_contact_streak == 0
    assert manager.place_contact_fz_n is None
    left, right = manager.goals()
    robot['ee'] = np.r_[left, right]
    manager.update(robot)
    assert manager.phase == 6
    assert manager.last_transition_reason == 'place_goal_reached'
    for _ in range(19):
        manager.update(robot)
    assert manager.phase == 6
    manager.update(robot)
    assert manager.phase == 7


def test_default_transfer_keeps_12mm_baseline():
    manager, robot = manager_and_robot()
    manager.phase = 4
    left, right = manager.goals()
    robot['ee'] = np.r_[left+[.02, 0., 0.], right]
    manager.update(robot)
    assert manager.phase == 4
    robot['ee'] = np.r_[left+[.011, 0., 0.], right]
    manager.update(robot)
    assert manager.phase == 5


def test_episode_reset_discards_pending_contact_and_geometry():
    manager, robot = manager_and_robot(place_contact_mode='raw_debug')
    manager.update(robot)
    manager.update(robot)
    assert manager.place_contact_streak == 2
    new_points = np.array([[.42, 0., .55], [.51, 0., .6]], np.float32)
    manager.failed = manager.completed = manager.opened = manager.placed = True
    manager.reset_episode(new_points)
    assert manager.phase == manager.age == manager.place_contact_streak == 0
    assert manager.place_contact_fz_n is None and manager.last_transition_reason == ''
    assert not (manager.failed or manager.completed or manager.opened or manager.placed)
    np.testing.assert_array_equal(manager.initial, new_points)
    new_points[:] = 0.
    assert manager.initial[0, 0] == pytest.approx(.42)
    manager.phase = 5
    left, right = manager.goals()
    robot['ee'] = np.r_[left+[0., 0., .04], right]
    manager.update(robot)
    assert manager.phase == 5 and manager.place_contact_streak == 1


def test_calibrated_h4_is_opt_in_and_enables_observer_without_raw_force():
    from utils.visual_protocol import effective_options
    manager, _ = manager_and_robot(place_contact_mode='calibrated')
    assert manager.verified_placement is not None
    assert manager.placement_evidence == {}
    args = parser().parse_args(['--grounding', 'unused.pt', '--place-contact-mode', 'calibrated'])
    assert effective_options(args)['controller_options']['placement_observer']
    assert args.place_contact_force_threshold_n == 8.  # Diagnostic option remains inert in H4.


def test_cli_defaults_match_baseline_and_require_debug_opt_in():
    args = parser().parse_args(['--grounding', 'unused.pt'])
    assert args.place_contact_mode == 'off'
    assert args.transfer_tolerance == .012
    assert not (args.legacy_perception or args.enable_arm_mask or args.enable_temporal_smoothing)
    with pytest.raises(ValueError, match='Unknown place contact mode'):
        manager_and_robot(place_contact_mode='typo')
