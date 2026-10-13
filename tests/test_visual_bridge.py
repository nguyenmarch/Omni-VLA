import numpy as np
import pytest
import torch
import h5py
import json
from models.grounding3d import Grounding3D
from utils.grounding_dataset import GroundingFrames
from simulation.vision import VisualTaskManager,robot_observation,OPEN_INSTRUCTION,FULL_INSTRUCTION


def test_proprioception_strips_oracle_and_condition_uses_perception():
    obs={k:np.zeros(n,np.float32) for k,n in [('q',14),('dq',14),('gripper',4),('wrench',12),('ee',6)]}
    obs.update(object=np.full(7,np.nan),phase=9,success=True)
    robot=robot_observation(obs)
    points=np.array([[.4,.1,.6],[.7,-.1,.5]],np.float32)
    manager=VisualTaskManager(points)
    condition=manager.condition(robot)
    assert condition.shape==(67,)
    np.testing.assert_array_equal(condition[50:56],points.flatten())
    assert np.isfinite(condition).all() and 'object' not in robot
    points[1,0]-=.18;manager.observe(points)
    assert manager.condition(robot)[56]==pytest.approx(.18)


def test_finite_instruction_changes_arm_selection():
    points=np.ones((2,3),np.float32);action=np.full(14,.5,np.float32)
    full=VisualTaskManager(points,FULL_INSTRUCTION).adapt_action(action)
    opening=VisualTaskManager(points,OPEN_INSTRUCTION).adapt_action(action)
    np.testing.assert_array_equal(full,action)
    np.testing.assert_array_equal(opening[:6],0)
    assert opening[6]==1 and np.array_equal(opening[7:],action[7:])
    with pytest.raises(ValueError):VisualTaskManager(points,'unsupported instruction')
    with pytest.raises(ValueError):VisualTaskManager(np.full((2,3),np.nan))


def test_depth_decode_with_offset_and_camera_translation():
    model=Grounding3D(img_size=4)
    heat=torch.zeros(1,1,4,4);heat[0,0,2,1]=10
    transform=torch.eye(4)[None];transform[0,:3,3]=torch.tensor([10.,20.,30.])
    point,uv,z=model.decode(heat,torch.tensor([.05]),torch.full((1,4,4),2.),torch.eye(3)[None],transform)
    torch.testing.assert_close(uv,torch.tensor([[1.,2.]]))
    torch.testing.assert_close(z,torch.tensor([2.05]))
    torch.testing.assert_close(point,torch.tensor([[12.05,24.1,32.05]]))


def test_dataset_visibility_and_episode_leakage(tmp_path):
    path=tmp_path/'frames.h5'
    with h5py.File(path,'w') as h:
        h.attrs['schema_version']='omni-grounding-rgbd-v2'
        h.attrs['splits']=json.dumps({'train':['a'],'val':[],'test':[]})
        g=h.create_group('episodes/a');g.attrs['success']=True
        values={'rgb':np.zeros((1,4,4,3),np.uint8),'depth':np.ones((1,4,4),np.float32),
                'K':np.eye(3,dtype=np.float32)[None],'T':np.eye(4,dtype=np.float32)[None],
                'targets':np.ones((1,2,3),np.float32),'phase':np.zeros(1,np.int64)}
        for key,value in values.items():g.create_dataset(key,data=value)
    assert len(GroundingFrames(path,'train'))==2
    with h5py.File(path,'a') as h:h['episodes/a/depth'][:]=0
    ds=GroundingFrames(path,'train')
    assert len(ds)==0 and ds.skipped==2
    with h5py.File(path,'a') as h:h.attrs['splits']=json.dumps({'train':['a'],'val':['a'],'test':[]})
    with pytest.raises(ValueError,match='leakage'):GroundingFrames(path,'train')
