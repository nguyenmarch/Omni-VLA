import hashlib
import json
import h5py
import numpy as np
import pytest
from scripts.merge_datasets import merge
from scripts.audit_state_replay import audit
from simulation.bimanual_env import BimanualBottleDrawerEnv, ROOT
from simulation.expert import state_condition
from utils.checkpoint import file_hash


def write_dataset(path, scene='scene', correction=False):
    with h5py.File(path, 'w') as h:
        h.attrs.update(schema_version='omni-state-v1', action_dt=.04, condition_dim=67,
                       action_dim=14, action_convention='test', scene_sha256=scene)
        names=['000000'] if correction else ['train','correction_000000','val','test']
        h.attrs['splits']=json.dumps({'train':names if correction else names[:2],
                                     'val':[] if correction else ['val'],
                                     'test':[] if correction else ['test']})
        for i,name in enumerate(names):
            g=h.create_group('episodes/'+name);g.attrs['seed']=100+i if correction else i
            g['condition']=np.full((2,67),i,dtype='f4')
            g['action']=np.zeros((2,14),dtype='f4');g.create_group('obs')


def test_merge_collision_preserves_heldout_and_rejects_seed_leakage(tmp_path):
    base=tmp_path/'base.h5';correction=tmp_path/'correction.h5';out=tmp_path/'merged.h5'
    write_dataset(base);write_dataset(correction,correction=True)
    merge(base,correction,out)
    with h5py.File(base) as a,h5py.File(out) as b:
        sa=json.loads(a.attrs['splits']);sb=json.loads(b.attrs['splits'])
        assert sb['val']==sa['val'] and sb['test']==sa['test']
        assert 'correction_1_000000' in sb['train']
        for name in sa['val']+sa['test']:
            assert np.array_equal(a['episodes'][name]['condition'][:],b['episodes'][name]['condition'][:])
    with h5py.File(correction,'r+') as h:h['episodes/000000'].attrs['seed']=2
    with pytest.raises(ValueError,match='held-out'):merge(base,correction,tmp_path/'leaked.h5')
    assert not (tmp_path/'leaked.h5').exists()


def test_scene_migration_requires_dataset_bound_audit(tmp_path):
    base=tmp_path/'base.h5';correction=tmp_path/'correction.h5';out=tmp_path/'out.h5'
    write_dataset(base,'legacy');write_dataset(correction,'current',True)
    with pytest.raises(ValueError,match='scene_sha256'):merge(base,correction,out)
    assert not out.exists()
    p=tmp_path/'audit.json'
    p.write_text(json.dumps({'passed':True,'dataset_sha256':file_hash(base),
                            'legacy_scene_sha256':'legacy','current_scene_sha256':'current'}))
    merge(base,correction,out,p)
    with h5py.File(out) as h:
        assert h.attrs['scene_sha256']=='current' and h.attrs['legacy_scene_sha256']=='legacy'
    with h5py.File(base,'r+') as h:h.attrs['changed']=True
    with pytest.raises(ValueError,match='scene_sha256'):merge(base,correction,tmp_path/'stale.h5',p)


def test_replay_detects_changed_observation(tmp_path):
    p=tmp_path/'replay.h5';env=BimanualBottleDrawerEnv()
    obs,_=env.reset(seed=17)
    try:
        with h5py.File(p,'w') as h:
            h.attrs['scene_sha256']=hashlib.sha256((ROOT/'simulation/assets/bottle_drawer_scene.xml').read_bytes()).hexdigest()
            g=h.create_group('episodes/a');g.attrs['seed']=17
            g['condition']=state_condition(obs,0)[None];g['phase']=np.array([0])
            g['action']=np.r_[np.zeros(6),1,np.zeros(6),1][None]
            for k,v in obs.items():g['obs/'+k]=v[None]
        result=audit(p,tmp_path/'passed.json')
        assert max(result['max_errors'].values())==0
        with h5py.File(p,'r+') as h:h['episodes/a/condition'][0,0]+=1
        with pytest.raises(ValueError,match='diverges'):audit(p,tmp_path/'failed.json')
        assert not (tmp_path/'failed.json').exists()
    finally:env.close()


def test_phase_sampling_uses_train_only_and_equal_phase_mass(tmp_path):
    from utils.dataset import BimanualFlowDataset, phase_sample_weights
    p=tmp_path/'sampling.h5'
    write_dataset(p)
    with h5py.File(p,'r+') as h:
        for name in h['episodes']:
            g=h['episodes'][name];g.attrs['success']=True
            g['phase']=np.array([0,1] if name=='train' else [0,0])
    ds=BimanualFlowDataset(p,chunk_size=1,cache_state=True)
    weights,counts=phase_sample_weights(ds)
    assert counts==[3,1]  # The validation/test observations do not contribute.
    assert np.isclose(weights[[0,2,3]].sum(),weights[1])
    capped,_=phase_sample_weights(ds,min_count=3)
    assert np.isclose(capped[1],1/3)  # A singleton phase is not upweighted to a whole phase mass.
    ds.close()
