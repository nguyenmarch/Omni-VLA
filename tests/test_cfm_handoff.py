import io
import json
import zipfile
import pytest
import torch
from scripts.evaluate_cfm_candidate import import_candidate
from utils.checkpoint import make_policy, file_hash


def archive(path, dataset, mismatch=False):
    config=dict(action_dim=14,cond_dim=67,chunk_size=2,hidden_dim=32,direct_skip=True)
    checkpoint=dict(schema_version='omni-state-v1',policy_type='cfm',model_config=config,
                    action_dt=.04,dataset_sha256='bad' if mismatch else file_hash(dataset),
                    model=make_policy('cfm',config).state_dict())
    raw=io.BytesIO();torch.save(checkpoint,raw)
    meta={k:v for k,v in checkpoint.items() if k!='model'}
    with zipfile.ZipFile(path,'w') as z:
        for name,contents in {'best.pt':raw.getvalue(),'policy.onnx':b'test-only',
                              'policy.json':json.dumps(meta).encode(),'metrics.jsonl':b'{}\n'}.items():
            z.writestr('cfm/'+name,contents)


def test_candidate_rejects_dataset_mismatch_without_partial_import(tmp_path):
    dataset=tmp_path/'dataset';dataset.write_bytes(b'v2')
    p=tmp_path/'outputs.zip';archive(p,dataset,True)
    with pytest.raises(ValueError,match='expected dataset'):
        import_candidate(p,dataset,tmp_path/'weights/candidate')
    assert not (tmp_path/'weights/candidate').exists()


def test_import_is_repeatable_but_does_not_replace_different_checkpoint(tmp_path):
    dataset=tmp_path/'dataset';dataset.write_bytes(b'v2')
    p=tmp_path/'outputs.zip';archive(p,dataset)
    dest=tmp_path/'weights/candidate'
    _,c=import_candidate(p,dataset,dest)
    assert c['policy_type']=='cfm'
    import_candidate(p,dataset,dest)
    (dest/'best.pt').write_bytes(b'other-checkpoint')
    with pytest.raises(ValueError,match='different outputs'):import_candidate(p,dataset,dest)
    assert (dest/'best.pt').read_bytes()==b'other-checkpoint'


def test_zip_path_traversal_is_rejected(tmp_path):
    p=tmp_path/'bad.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('cfm/../../bad.pt',b'bad')
    with pytest.raises(ValueError,match='Unexpected ZIP'):import_candidate(p,tmp_path/'dataset',tmp_path/'candidate')
    assert not (tmp_path/'bad.pt').exists()


def test_wrong_chunk_is_rejected_before_import(tmp_path):
    dataset=tmp_path/'dataset';dataset.write_bytes(b'v2')
    p=tmp_path/'outputs.zip';archive(p,dataset)
    with pytest.raises(ValueError,match='chunk_size'):
        import_candidate(p,dataset,tmp_path/'h1',expected_chunk_size=1)
    assert not (tmp_path/'h1').exists()


def test_action_diagnostics_separate_noise_variability_and_clipping():
    import numpy as np
    from scripts.diagnose_policy_actions import summarize_actions
    targets=np.zeros((2,14),dtype=np.float32);targets[:,[6,13]]=1
    predictions=np.broadcast_to(targets,(2,2,14)).copy()
    predictions[0,0,0]=2
    predictions[1,0,[6,13]]=-1
    r=summarize_actions(predictions,targets,np.array([0,1]))
    assert r['approach']['states']==1
    assert r['approach']['gripper_sign_mismatch_fraction']==.5
    assert r['approach']['raw_action_out_of_bounds_fraction']==1/28
    assert r['approach']['noise_action_std_mean']>0
    assert r['descend']['first_action_mse']==0


def test_trace_jitter_excludes_phase_boundaries_and_episode_boundaries(tmp_path):
    from scripts.analyze_policy_trace import analyze
    rows=[]
    for seed in [1,2]:
        for phase,action in [('approach',0),('approach',0),('descend',1)]:
            rows.append(dict(seed=seed,phase=phase,action=[action]*14,expert_action=[0]*14,
                             position_error_m=[.1,.2],rotation_error_rad=[.1,.1],action_mae=action))
    trace=tmp_path/'trace.jsonl';trace.write_text('\n'.join(json.dumps(r) for r in rows))
    evaluation=tmp_path/'eval.json';evaluation.write_text(json.dumps({'episodes':[
        dict(seed=s,success=False,phase='descend',failure_reason='timeout_descend') for s in [1,2]]}))
    r=analyze(trace,evaluation,tmp_path/'result.json')
    assert r['phase_action_errors']['approach']['consecutive_action_delta_mean_per_dim']==[0]*14
    assert r['phase_action_errors']['descend']['consecutive_action_delta_mean_per_dim'] is None
    assert len(r['all_phase_failures'])==2
