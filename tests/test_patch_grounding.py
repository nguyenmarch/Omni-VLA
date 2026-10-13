import json
from types import SimpleNamespace
import h5py
import numpy as np
import torch
import pytest
from scripts.prepare_patch_grounding import prepare
from utils.patch_grounding_dataset import PatchGroundingFrames
from scripts.train_patch_grounding import summarize, selection_key
from models.grounding3d import Grounding3D, GroundingRuntime


def test_labels_use_patch_surface_and_keep_all_eval_targets(tmp_path):
    source=tmp_path/'raw.h5';output=tmp_path/'patch.h5'
    with h5py.File(source,'w') as h:
        h.attrs['schema_version']='omni-grounding-rgbd-v2'
        h.attrs['splits']=json.dumps({'train':['train'],'val':['val'],'test':['test']})
        episodes=h.create_group('episodes')
        for name in ('train','val','test'):
            g=episodes.create_group(name);g.attrs['success']=True
            g['rgb']=np.zeros((1,16,16,3),dtype=np.uint8)
            depth=np.ones((1,16,16),dtype=np.float32);depth[0,8,8]=4.;g['depth']=depth
            g['K']=np.array([[[100.,0.,8.],[0.,100.,8.],[0.,0.,1.]]],dtype=np.float32)
            g['T']=np.eye(4,dtype=np.float32)[None]
            g['targets']=np.array([[[0.,0.,1.03],[0.,0.,2.]]],dtype=np.float32);g['phase']=np.array([0])
    prepare(SimpleNamespace(source=str(source),output=str(output),quantile=.5))
    with h5py.File(output) as h:
        g=h['episodes/train']
        assert g['patch_surface'][0,0]==1
        assert g['patch_offset'][0,0]==pytest.approx(.03,abs=1e-6)
        assert g['patch_label_valid'][0].tolist()==[True,False]
    with h5py.File(source) as h:assert 'patch_offset' not in h['episodes/train']
    assert len(PatchGroundingFrames(output,'train'))==1
    assert len(PatchGroundingFrames(output,'val'))==2


def test_rejections_count_in_coverage_and_model_selection():
    report=summarize([.01,float('nan'),.02],[True,False,True],[True,False,True])
    assert report['coverage']==pytest.approx(2/3)
    assert report['within_1_5cm_all_fraction']==pytest.approx(1/3)
    assert report['penalized_error_m']==pytest.approx(1.03/3)
    rejected=summarize([float('nan')],[False],[False])
    assert rejected['valid_mean_error_m'] is None and rejected['penalized_error_m']==1


def test_runtime_rejects_bad_depth_without_fallback(tmp_path):
    cfg={'img_size':32,'feature_dim':64,'latent_dim':64,'decoder_config':{'patch_size':7}}
    model=Grounding3D(**cfg);checkpoint=tmp_path/'policy.pt'
    torch.save({'schema_version':'omni-grounding-patch-v3','model_config':cfg,'model':model.state_dict()},checkpoint)
    runtime=GroundingRuntime(checkpoint)
    frame={'rgb':np.zeros((32,32,3),np.uint8),'depth':np.full((32,32),np.nan,np.float32),
        'K':np.eye(3,dtype=np.float32),'T':np.eye(4,dtype=np.float32)}
    with pytest.raises(ValueError,match='no fallback'):runtime.predict(frame)
    cfg.pop('decoder_config')
    torch.save({'schema_version':'omni-grounding-patch-v3','model_config':cfg,'model':model.state_dict()},checkpoint)
    with pytest.raises(ValueError,match='contract'):GroundingRuntime(checkpoint)


def test_accurate_coverage_precedes_accepting_inaccurate_points():
    wrong=summarize([.4,.4],[True,True],[True,True])
    useful=summarize([.01,float("nan")],[True,False],[True,True])
    assert wrong["penalized_error_m"]<useful["penalized_error_m"]
    assert selection_key(useful)<selection_key(wrong)
