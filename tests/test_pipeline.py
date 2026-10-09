import json
import h5py
import mujoco
import numpy as np
import pytest
import torch
from gymnasium.utils.env_checker import check_env
from simulation.bimanual_env import BimanualBottleDrawerEnv
from simulation.expert import ScriptedExpert,state_condition,COND_DIM
from models.impedance_controller import exp_rotation,rotation_error
from models.flow_matching_policy import FlowMatchingActionPolicy
from models.affordance_head import VisualAffordanceHead
from utils.dataset import BimanualFlowDataset
from utils.camera import calibration,project,unproject
from simulation.recovery import SlipMonitor,RecoveryState


def test_reset_timing_and_space():
    env=BimanualBottleDrawerEnv()
    a,_=env.reset(seed=7);b,_=env.reset(seed=7)
    assert all(np.array_equal(a[k],b[k]) for k in a)
    assert env.observation_space.contains(a)
    assert state_condition(a,0).shape==(COND_DIM,)
    env.step(np.r_[np.zeros(6),1,np.zeros(6),1])
    assert np.isclose(env.data.time,.04)
    env.reset(seed=7)
    assert env.step_counter==0 and env.data.time==0
    with pytest.raises(ValueError):env.step(np.ones(13))
    check_env(env,skip_render_check=True)
    env.close()


def test_physics_stability_and_free_space_tracking():
    env=BimanualBottleDrawerEnv()
    initial=env.site_pos('left_ee')
    target=initial+[0,0,-.03]
    hold=np.r_[np.zeros(6),1,np.zeros(6),1]
    for _ in range(80):
        action=hold.copy();action[:3]=np.clip((target-env.site_pos('left_ee'))/.015,-1,1)
        env.step(action)
    assert np.linalg.norm(env.site_pos('left_ee')-target)<.01
    for _ in range(420):env.step(hold)
    assert np.isfinite(env.data.qpos).all() and np.isfinite(env.data.qvel).all()
    assert np.max(np.abs(env.data.ctrl))<=87
    env.close()


def test_expert_moves_objects_via_physics():
    env=BimanualBottleDrawerEnv();env.reset(seed=0);expert=ScriptedExpert(env)
    for _ in range(1500):
        _,_,done,trunc,info=env.step(expert.action());expert.update()
        if done or trunc or expert.failed:break
    assert info['drawer_open'] and info['bottle_placed'] and info['success']
    assert env.site_pos('bottle_site')[1]>0
    env.close()


def test_geometry_roundtrip_and_rotation():
    env=BimanualBottleDrawerEnv()
    k,t=calibration(env.model,env.data,'head_cam',128,128)
    p=env.site_pos('bottle_site');uv,z=project(p,k,t)
    wrist_id=env.model.camera('right_wrist_cam').id
    wrist_forward=-env.data.cam_xmat[wrist_id].reshape(3,3)[:,2]
    tool_forward=env.data.site('right_ee').xmat.reshape(3,3)[:,2]
    assert wrist_forward @ tool_forward > .8
    assert z>0
    assert np.allclose(unproject(uv,z,k,t),p,atol=1e-6)
    r=exp_rotation(np.array([.1,-.2,.3]))
    assert np.allclose(rotation_error(r,np.eye(3)),[.1,-.2,.3],atol=1e-6)
    env.close()


def test_chunk_boundary_mask_and_train_stats(tmp_path):
    path=tmp_path/'data.h5'
    with h5py.File(path,'w') as h:
        h.attrs['schema_version']='omni-state-v1'
        h.attrs['splits']=json.dumps({'train':['a'],'val':['b'],'test':['c']})
        for name,value in [('a',1),('b',100),('c',200)]:
            g=h.create_group('episodes/'+name);g.attrs['success']=True
            g['condition']=np.full((3,67),value,np.float32);g['action']=np.full((3,14),value,np.float32)
    ds=BimanualFlowDataset(path,chunk_size=4,split='val')
    assert np.all(ds.stats['condition_mean']==1)
    sample=ds[2]
    assert sample['mask'].sum()==1
    assert torch.all(sample['action'][1:]==0)
    assert ds.index[2]==('b',2)
    cached=BimanualFlowDataset(path,chunk_size=4,split='val',stats=ds.stats,cache_state=True)
    assert all(torch.equal(cached[2][k],sample[k]) for k in sample)
    ds.close()


def test_flow_numerics_and_determinism():
    model=FlowMatchingActionPolicy(hidden_dim=32);c=torch.randn(2,67);x=torch.randn(2,16,14)
    assert model(x,torch.tensor([.2,.8]),c).shape==x.shape
    a=model.sample_euler(c,4,noise=x);b=model.sample_euler(c,4,noise=x)
    assert torch.equal(a,b) and torch.isfinite(a).all()
    with pytest.raises(ValueError):model.sample_euler(c,0)


def test_grounding_spatial_shape_and_depth_requirement():
    model=VisualAffordanceHead(feature_dim=32,latent_dim=16,n_heads=4,img_size=32)
    rgb=torch.rand(2,3,32,32);ids=model.language.tokenize(['grasp bottle','pull drawer handle'])
    heat,latent,points,_=model.from_rgb(rgb,ids)
    assert heat.shape==(2,1,32,32) and latent.shape==(2,16) and points is None
    with pytest.raises(ValueError):model.from_rgb(rgb,ids,camera_matrix=torch.eye(3).repeat(2,1,1))
    _,_,points,_=model.from_rgb(rgb,ids,camera_matrix=torch.eye(3).repeat(2,1,1),depth=torch.ones(2,32,32),world_from_camera=torch.eye(4).repeat(2,1,1))
    assert points.shape==(2,3) and torch.isfinite(points).all()


def test_recovery_debounce_and_retry_bound():
    monitor=SlipMonitor(debounce=3,max_retries=1)
    for _ in range(2):assert monitor.update([0,0,0],[1,0,0],np.zeros(12),True)==RecoveryState.RUNNING
    assert monitor.update([0,0,0],[1,0,0],np.zeros(12),True)==RecoveryState.HOLD
    assert monitor.advance()==RecoveryState.REGROUND
    assert monitor.advance(grounding_ready=True)==RecoveryState.REGRASP
    assert monitor.advance(grasp_confirmed=True)==RecoveryState.RUNNING
    for _ in range(3):monitor.update([0,0,0],[1,0,0],np.zeros(12),True)
    assert monitor.state==RecoveryState.FAILED


def test_environment_config_portability_and_timing(tmp_path):
    from utils.config import make_environment
    env=make_environment();assert env.substeps==20;env.close()
    path=tmp_path/'bad.yaml'
    path.write_text('env:\n  physics_dt: 0.002\n  controller_hz: 200\n')
    with pytest.raises(ValueError):make_environment(path)


def test_optional_policy_architectures():
    from utils.checkpoint import make_policy
    config=dict(action_dim=14,chunk_size=16,cond_dim=67,hidden_dim=32,direct_skip=True,phase_heads=True)
    condition=torch.randn(2,67)
    for kind in ['bc','cfm']:
        policy=make_policy(kind,config)
        result=policy.sample_euler(condition,4)
        assert result.shape==(2,16,14) and torch.isfinite(result).all()
