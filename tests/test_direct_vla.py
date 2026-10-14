import numpy as np
import pytest
import torch
import h5py
import json
from utils.vla_dataset import VLADataset
from models.direct_vla import DirectVLA, measured_image, proprioception, tokenize, TASK_PHRASES, PARAPHRASES


def test_model_uses_language_and_history_without_oracle_inputs():
    torch.manual_seed(42)
    model=DirectVLA(width=32,chunk_size=5).eval()
    images=torch.rand(1,3,7,32,32);robot=torch.rand(1,3,50);mask=torch.tensor([[True,True,False]])
    tokens=tokenize([TASK_PHRASES[0][0]])
    with torch.no_grad():
        first=model(images,robot,tokens,mask)['action']
        changed=model(images,robot,tokenize([TASK_PHRASES[1][0]]),mask)['action']
        images[:,2]=999;robot[:,2]=999
        padded=model(images,robot,tokens,mask)['action']
    assert first.shape==(1,5,14) and torch.isfinite(first).all()
    assert not torch.allclose(first,changed)
    torch.testing.assert_close(first,padded)
    assert tokenize(PARAPHRASES).shape==(2,24)
    with pytest.raises(ValueError):tokenize(['invented unsupported object'])


def test_calibrated_visual_features_and_robot_oracle_poisoning():
    frame={'rgb':np.zeros((4,4,3),np.uint8),'depth':np.ones((4,4),np.float32),
           'K':np.eye(3),'T':np.eye(4)}
    a=measured_image(frame);assert a.shape==(7,4,4)
    frame['T'][0,3]=.1;b=measured_image(frame)
    assert b[4,0,0]-a[4,0,0]==pytest.approx(.1)
    obs={key:np.zeros(n,np.float32) for key,n in [('q',14),('dq',14),('gripper',4),('wrench',12),('ee',6)]}
    obs.update(object=np.full(7,np.nan),phase=np.nan)
    assert np.isfinite(proprioception(obs)).all() and proprioception(obs).shape==(50,)
    frame['depth'][:]=np.nan
    with pytest.raises(ValueError,match='depth support'):measured_image(frame)


def test_vla_episode_history_and_train_only_normalization(tmp_path):
    path=tmp_path/'data.h5'
    with h5py.File(path,'w') as f:
        f.attrs.update(schema='omni-direct-vla-v1',image_size=4,chunk_size=5,action_dt=.04,
                       splits=json.dumps({'train':['a'],'val':['b'],'test':['c']}))
        for name,seed,value in [('a',1,1.),('b',2,1000.),('c',3,-1000.)]:
            g=f.create_group('episodes/'+name);g.attrs.update(seed=seed,task=0)
            data={'rgb':np.zeros((2,4,4,3),np.uint8),'depth':np.ones((2,4,4),np.float32),
                  'K':np.repeat(np.eye(3)[None],2,0),'T':np.repeat(np.eye(4)[None],2,0),
                  'robot':np.full((2,50),value,np.float32),'action':np.zeros((2,5,14),np.float32),
                  'action_mask':np.ones((2,5,1),np.float32),'phase':np.zeros(2,np.int64),
                  'landmarks':np.zeros((2,6),np.float32)}
            for key,v in data.items():g.create_dataset(key,data=v)
    val=VLADataset(path,'val',history=3)
    np.testing.assert_array_equal(val.stats['mean'],np.ones(50))
    assert val[0]['history_mask'].sum()==1 and val[1]['history_mask'].sum()==2
    assert val[0]['images'].shape==(3,7,4,4)
    with h5py.File(path,'a') as f:f['episodes/c'].attrs['seed']=1
    with pytest.raises(ValueError,match='leakage'):VLADataset(path,'train')
