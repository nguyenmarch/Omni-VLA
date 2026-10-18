"""Guarded release and verified opening/vertical retreat counterexamples."""
import copy

import numpy as np
import pytest

from simulation.verified_placement import VerifiedPlacement
from simulation.vision import VisualTaskManager
from utils.visual_protocol import normalize_options


def fixture():
    fsm=VerifiedPlacement()
    robot={'ee':np.array([.47,.1,.56,.28,.1,.51]),'ee_velocity':np.zeros((2,3)),
           'ee_rotations':np.stack([np.eye(3)]*2),'gripper':np.array([.024,.024,.012,.012])}
    points=np.array([[.47,.1,.504],[.28,.1,.51]])
    evidence={'calibration_status':'calibrated','support':False,'collision':False,
              'payload_n':1.18,'force_stable':True,'load_loss_n':0.,'raw_load_loss_n':0.}
    return fsm,robot,points,evidence


def observe(fsm,robot,points,evidence,phase=5,usable=True,age=.04):
    return fsm.observe(robot,phase,points,.2,np.array([[.47,.1,.54],[.28,.1,.51]]),.04,usable,age,evidence)


def authorize():
    f,r,p,e=fixture()
    for _ in range(3):
        action=f.command(np.zeros(14),r,5,p,.2,.04,True,.04,e)
        assert action[6]==action[13]==-1
        next_phase,reason=observe(f,r,p,e)
    assert next_phase==6 and reason=='verified_valid_pose'
    return f,r,p,e


def opened():
    f,r,p,e=authorize()
    action=f.command(np.zeros(14),r,6,p,.2,.04,True,.04,e)
    assert action[6]==1 and action[13]==-1
    r['gripper'][:2]=.04
    p[0,2]=.492
    e.update(calibration_status='grasp_changed',load_loss_n=1.18,raw_load_loss_n=1.18)
    for _ in range(3):
        phase,reason=observe(f,r,p,e,phase=6)
    assert phase==7 and reason=='gripper_open_and_unloaded_verified'
    return f,r,p,e


def test_pose_drop_can_authorize_without_claiming_contact():
    f,_,_,_=authorize()
    assert f.authorization=='verified_valid_pose' and not f.open_commanded


@pytest.mark.parametrize('bad', ['stale','fallback_age','outside','high_drop','fast','unknown','collapsed','right_lost'])
def test_no_release_on_bad_geometry_dynamics_or_unknown_sensor(bad):
    f,r,p,e=fixture();usable=True;age=.04
    if bad=='stale':usable=False
    elif bad=='fallback_age':age=.4
    elif bad=='outside':p[0,1]+=.2
    elif bad=='high_drop':p[0,2]=.6
    elif bad=='fast':r['ee_velocity'][0,2]=.1
    elif bad=='unknown':e['calibration_status']='uncalibrated_payload'
    elif bad=='collapsed':r['gripper'][:2]=0.
    elif bad=='right_lost':r['gripper'][2:]=0.
    e['support']=True
    for _ in range(5):
        assert observe(f,r,p,e,usable=usable,age=age)[0] is None
    assert not f.open_commanded


def test_calibrated_support_needs_floor_and_grasp_geometry():
    f,r,p,e=fixture();e['support']=True
    p[0,2]=.492;r['ee'][2]=.575
    for _ in range(3):phase,reason=observe(f,r,p,e)
    assert phase==6 and reason=='calibrated_support'
    f.reset();r['ee'][2]=.8
    for _ in range(4):assert observe(f,r,p,e)[0] is None


def test_side_collision_hold_correction_is_bounded_and_never_opens():
    f,r,p,e=fixture();e['collision']=True;e['support']=True
    for _ in range(130):
        observe(f,r,p,e)
        action=f.command(np.zeros(14),r,5,p,.2,.04,True,.04,e)
        assert action[6]==action[13]==-1
        if f.correction_goal is not None:
            assert 0<f.correction_goal[0,2]-f.anchor[0,2]<=.016+1e-9
        if f.failure_reason:break
    assert f.failure_reason=='visual_placement_collision_timeout'


def test_historical_collision_can_rearm_only_after_restored_load_and_quiet_window():
    f,r,p,e=fixture();e['collision']=True
    e.update(wrench_world=[0.,.55,8.63,0.,.07,0.],expected_free_wrench_world=[0.,0.,8.63,0.,.07,0.],
             residual_world=[0.,.55,0.,0.,0.,0.])
    for _ in range(4):
        assert observe(f,r,p,e)[0] is None
        assert not f.needs_policy(e)
    observe(f,r,p,e)
    assert f.collision_rearmed and f.needs_policy(e)
    e['wrench_world'][1]=2.
    assert not f.needs_policy(e)  # New impact must re-arm the veto immediately.
    f.reset();e['wrench_world'][1]=.55;e['load_loss_n']=e['raw_load_loss_n']=1.
    for _ in range(8):observe(f,r,p,e)
    assert not f.collision_rearmed


def test_aperture_and_new_pixels_and_unload_all_required_before_retreat():
    f,r,p,e=authorize()
    f.command(np.zeros(14),r,6,p,.2,.04,True,.04,e)
    for _ in range(5):assert observe(f,r,p,e,6)[0] is None
    r['gripper'][:2]=.04;p[0,2]=.492
    for _ in range(5):assert observe(f,r,p,e,6)[0] is None  # Still carrying load.
    e.update(calibration_status='grasp_changed',load_loss_n=1.18,raw_load_loss_n=1.18)
    f.open_since=f.elapsed
    for _ in range(3):assert observe(f,r,p,e,6,age=.2)[0] is None  # Pre-open image.
    assert observe(f,r,p,e,6,age=.04)[0]==7


def test_open_timeout_and_sensor_invalid_are_not_forced_transitions():
    f,r,p,e=authorize()
    for _ in range(60):
        f.command(np.zeros(14),r,6,p,.2,.04,True,.04,e)
        assert observe(f,r,p,e,6)[0] is None
    assert f.failure_reason=='visual_release_open_timeout'
    f,r,p,e=authorize();e['calibration_status']='sensor_invalid'
    action=f.command(np.zeros(14),r,6,p,.2,.04,True,.04,e)
    assert action[6]==-1 and not f.open_commanded


def test_retreat_uses_bounded_z_servo_and_close_waits_for_clearance():
    f,r,p,e=opened()
    previous=np.zeros((2,3))
    for _ in range(6):
        action=f.command(np.ones(14),r,7,p,.2,.04,True,.04,e)
        commands=action.reshape(2,7)[:,:3]*.015/.04
        assert np.max(np.linalg.norm(commands,axis=1))<=.15+1e-7
        assert np.max(np.linalg.norm(commands-previous,axis=1))<=.75*.04+1e-7
        previous=commands
        assert action[6]==1 and action[13]==-1 and action[2]>0
        assert observe(f,r,p,e,7)[0] is None
    r['ee'][2]=.78
    for _ in range(3):next_phase,reason=observe(f,r,p,e,7)
    assert next_phase==8 and reason=='retreat_clear_verified'
    r['ee'][2]=.7
    action=f.command(np.ones(14),r,8,p,.2,.04,True,.04,e)
    assert np.all(action[7:13]==0) and not f.close_allowed(r)
    r['ee'][2]=.78
    assert f.close_allowed(r)


def test_retreat_no_progress_and_closed_gripper_are_explicit_failures():
    f,r,p,e=opened()
    for _ in range(50):assert observe(f,r,p,e,7)[0] is None
    assert f.failure_reason=='visual_retreat_no_progress'
    f,r,p,e=opened();r['gripper'][0]=.02
    assert observe(f,r,p,e,7)[0] is None and f.failure_reason=='visual_retreat_gripper_closed'


@pytest.mark.parametrize('phase',[6,7])
def test_phase_token_alone_cannot_open_or_retreat(phase):
    f,r,p,e=fixture()
    a=f.command(np.zeros(14),r,phase,p,.2,.04,True,.04,e)
    assert a[6]==-1 and np.all(a[:6]==0)
    assert f.failure_reason


def test_geometry_timeout_reset_and_oracle_poisoning():
    f,r,p,e=fixture();twin=copy.deepcopy(f)
    r['object']=np.full(7,np.nan);r['scoring_only']={'success':True}
    assert observe(f,r,p,e)==observe(twin,r,p,e)
    for _ in range(25):observe(f,r,p,e,usable=False)
    assert f.failure_reason=='visual_placement_perception_timeout'
    f.reset()
    assert not f.open_commanded and not f.open_verified and f.anchor is None and not f.failure_reason


def test_manager_keeps_67d_tokens_and_resets_fsm_and_observer():
    from simulation.placement_contact import PlacementContactObserver
    manager=VisualTaskManager([[.45,-.18,.462],[.49,.1,.51]],place_contact_mode='calibrated')
    _,r,_,_=fixture();r.update(q=np.zeros(14),dq=np.zeros(14),wrench=np.zeros(12))
    assert manager.condition(r).shape==(67,)
    manager.placement_observer=PlacementContactObserver()
    manager.verified_placement.open_verified=True
    manager.reset_episode(manager.initial)
    assert not manager.verified_placement.open_verified and manager.placement_observer.loaded is None


@pytest.mark.parametrize('jam',[False,True])
def test_real_physics_open_before_retreat_and_jammed_fingers_never_retreat(jam):
    """Original scene, friction grasp. Ideal landmarks isolate the mechanics;
    the separate B/T/F/TF benchmark supplies learned P5 landmarks end to end.
    """
    from utils.config import make_environment
    from utils.visual_scoring import scoring_snapshot
    from simulation.expert import ScriptedExpert
    from simulation.placement_contact import PlacementContactObserver
    from simulation.vision import robot_observation
    from scripts.evaluate_placement_contact import observer_sample
    env=make_environment()
    try:
        obs,_=env.reset(seed=76001)
        initial=obs['object'][:6].reshape(2,3).copy()
        expert=ScriptedExpert(env);sensor=PlacementContactObserver()
        previous=obs['ee'].reshape(2,3).copy()
        for _ in range(1000):
            phase=expert.phase
            obs,*_=env.step(expert.action())
            sensor.update(observer_sample(env,obs,previous,phase),env.dt)
            previous=obs['ee'].reshape(2,3).copy()
            expert.update()
            if expert.phase==6:break
            assert not expert.failed
        assert expert.phase==6 and sensor.result['calibration_status']=='calibrated'
        manager=VisualTaskManager(initial,place_contact_mode='calibrated')
        manager.phase=5;manager.placement_observer=sensor
        manager.observe(obs['object'][:6].reshape(2,3))
        velocity=np.zeros((2,3));transitions=[]
        for _ in range(300):
            before=obs['ee'].reshape(2,3).copy()
            robot=robot_observation(obs)
            robot.update(ee_velocity=velocity,ee_rotations=np.stack([
                env.data.site(s+'_ee').xmat.reshape(3,3).copy() for s in ('left','right')]))
            phase=manager.phase
            action=manager.adapt_action(expert.action(),robot,env.dt,True,0.)
            if phase==7:
                assert manager.verified_placement.open_verified
                assert np.all(obs['gripper'][:2]>.032) and action[6]==1.
            if jam and phase==6:
                # A real joint stop prevents aperture confirmation even when
                # the controller commands open; no sensor/contact is forged.
                for finger in (1,2):
                    env.model.jnt_range[env.model.joint(f'left_finger_joint{finger}').id,1]=.025
            obs,*_=env.step(action)
            velocity=(obs['ee'].reshape(2,3)-before)/env.dt
            sensor.update(observer_sample(env,obs,before,phase),env.dt)
            robot=robot_observation(obs)
            robot.update(ee_velocity=velocity,ee_rotations=np.stack([
                env.data.site(s+'_ee').xmat.reshape(3,3).copy() for s in ('left','right')]))
            manager.observe(obs['object'][:6].reshape(2,3))
            manager.update(robot,env.dt,True,0.)
            if manager.phase!=phase:
                transitions.append((phase,manager.phase))
                if manager.phase==7:assert np.all(obs['gripper'][:2]>.032)
            if manager.failed or manager.phase==8:break
        if jam:
            assert manager.phase==6 and manager.failure_reason=='visual_release_open_timeout'
            assert not manager.verified_placement.open_verified and np.all(obs['gripper'][:2]<.032)
            assert transitions==[(5,6)]
        else:
            assert not manager.failed and manager.phase==8
            assert transitions==[(5,6),(6,7),(7,8)]
            assert obs['ee'][2]>=.768 and np.all(obs['gripper'][:2]>.032)
            assert scoring_snapshot(env)['bottle_floor_contact']
    finally:env.close()


@pytest.mark.parametrize('options',[{'release_max_drop_m':.08},{'release_retreat_height_m':.5},
                                   {'release_open_threshold_m':.02},{'release_stable_steps':1.5}])
def test_invalid_release_options_fail_before_rollout(options):
    with pytest.raises(ValueError):normalize_options(controller=options)
