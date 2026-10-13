import json
import h5py
import numpy as np
import pytest
from scripts.compare_grounding_resolution import audit_pair
from simulation.bimanual_env import BimanualBottleDrawerEnv
from utils.camera import calibration,project


def test_resolution_changes_intrinsics_not_scene_or_physical_labels():
    env=BimanualBottleDrawerEnv(images=False)
    try:
        env.reset(seed=8000)
        k128,t128=calibration(env.model,env.data,'head_cam',128,128)
        k256,t256=calibration(env.model,env.data,'head_cam',256,256)
        point=env.site_pos('handle_site')
        uv128,z128=project(point,k128,t128);uv256,z256=project(point,k256,t256)
        np.testing.assert_allclose(uv256,2*uv128)
        assert z256==z128
        np.testing.assert_array_equal(t128,t256)
    finally:env.close()


def test_pair_audit_rejects_changed_trajectory_and_calibration(tmp_path):
    paths=[tmp_path/'low.h5',tmp_path/'high.h5']
    for path,size in zip(paths,(128,256)):
        with h5py.File(path,'w') as h:
            h.attrs['scene_sha256']='same-scene';h.attrs['splits']=json.dumps({'train':['a'],'val':[],'test':[]})
            g=h.create_group('episodes/a');g.attrs.update(seed=8000,success=True)
            g.create_dataset('rgb',data=np.zeros((1,size,size,3),np.uint8))
            for key in ('targets','phase','step','q','dq','ee','gripper','wrench','action','condition'):
                g.create_dataset(key,data=np.zeros((1,1),np.float32))
            k=np.eye(3,dtype=np.float32);k[:2]*=size/128
            g.create_dataset('K',data=k[None]);g.create_dataset('T',data=np.eye(4,dtype=np.float32)[None])
    assert audit_pair(*paths)['trajectory_values_identical']
    with h5py.File(paths[1],'a') as h:h['episodes/a/q'][:]=1
    with pytest.raises(ValueError,match='Trajectory mismatch'):audit_pair(*paths)
    with h5py.File(paths[1],'a') as h:
        h['episodes/a/q'][:]=0;h['episodes/a/K'][0,0,0]=3
    with pytest.raises(ValueError,match='Intrinsics'):audit_pair(*paths)
