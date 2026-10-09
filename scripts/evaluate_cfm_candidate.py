"""Import CFM outputs and compare 4/6/8 Euler sampling with the frozen BC v2."""
import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
import onnxruntime as ort
from utils.checkpoint import load_policy, file_hash
from scripts.eval_closed_loop import evaluate


def import_candidate(archive, dataset, destination):
    destination=Path(destination)
    with zipfile.ZipFile(archive) as z:
        entries={}
        for entry in z.infolist():
            if entry.is_dir():continue
            p=Path(entry.filename)
            if len(p.parts)!=2 or p.parts[0]!='cfm' or p.name in entries:
                raise ValueError(f'Unexpected ZIP entry: {p}')
            entries[p.name]=z.read(entry)
    if not {'best.pt','policy.onnx','policy.json','metrics.jsonl'}<=entries.keys():
        raise ValueError('ZIP lacks required CFM outputs')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as tmp:
        folder=Path(tmp)/'cfm';folder.mkdir()
        for name,contents in entries.items():(folder/name).write_bytes(contents)
        model,c=load_policy(folder/'best.pt')
        if c['policy_type']!='cfm' or c['dataset_sha256']!=file_hash(dataset):
            raise ValueError('Require CFM trained on the expected dataset')
        if c['model_config']['cond_dim']!=67 or c['model_config']['action_dim']!=14 or abs(c['action_dt']-.04)>1e-9:
            raise ValueError('Checkpoint incompatible with state simulation')
        meta=json.loads((folder/'policy.json').read_text())
        for key in ('model_config','dataset_sha256','action_dt','policy_type','schema_version'):
            if meta[key]!=c[key]:raise ValueError(f'ONNX metadata mismatch: {key}')
        if destination.exists():
            if any(not (destination/name).is_file() or (destination/name).read_bytes()!=contents for name,contents in entries.items()):
                raise ValueError('Destination contains different outputs; use a new destination')
        else:folder.rename(destination)
    return model,c


def onnx_parity(model, checkpoint, path, steps):
    options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
    session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
    rng=np.random.default_rng(42)
    condition=rng.normal(size=(1,67)).astype(np.float32)
    noise=rng.normal(size=(1,checkpoint['model_config']['chunk_size'],14)).astype(np.float32)
    errors={}
    for count in steps:
        reference=model.sample_euler(torch.from_numpy(condition),count,noise=torch.from_numpy(noise)).numpy()
        x=noise.copy()
        for i in range(count):
            x+=session.run(None,{'x_t':x,'t':np.array([i/count],np.float32),'condition':condition})[0]/count
        error=float(np.max(np.abs(reference-x)))
        if not np.isfinite(error) or error>1e-4:raise ValueError(f'ONNX Euler parity {count} steps: {error}')
        errors[str(count)]=error
    return errors


def run(archive, dataset, destination, output_dir, episodes=50, steps=(4,6,8)):
    torch.set_num_threads(2)
    baseline_path=Path('weights/bc_kaggle_v2/best.pt')
    baseline_hash=file_hash(baseline_path)
    baselines={}
    for label,start in [('same',5000),('fresh',7000)]:
        r=json.loads(Path(f'artifacts/bc_kaggle_v2_{label}.json').read_text())
        if [e['seed'] for e in r['episodes']]!=list(range(start,start+50)) or r['execute_horizon']!=1:
            raise ValueError('BC baseline seed protocol mismatch')
        baselines[label]={k:r[k] for k in ('successes','attempted','success_rate','wilson_95','sampler_ms_p50','sampler_ms_p95')}
    _,bc=load_policy(baseline_path)
    if bc['dataset_sha256']!=file_hash(dataset):raise ValueError('BC and CFM must use identical dataset')
    model,c=import_candidate(archive,dataset,destination)
    parity=onnx_parity(model,c,Path(destination)/'policy.onnx',steps)
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    checkpoint_path=Path(destination)/'best.pt';candidate_hash=file_hash(checkpoint_path)
    rows=[]
    for count in steps:
        for label,seed in [('same',5000),('fresh',7000)]:
            output=output_dir/f'euler_{count}_{label}.json'
            if output.exists():
                r=json.loads(output.read_text())
                if (r.get('checkpoint_sha256')!=candidate_hash or r['euler_steps']!=count
                    or r['execute_horizon']!=1 or r['cpu_threads']!=2 or r.get('interrupted')
                    or r.get('seeds_requested')!=list(range(seed,seed+episodes))):
                    raise ValueError(f'Existing report incompatible: {output}')
            else:
                args=SimpleNamespace(checkpoint=str(checkpoint_path),expert=False,threads=2,seed=seed,
                                     config=None,gui=False,episodes=episodes,inject_slip=False,recovery=False,
                                     euler_steps=count,execute_horizon=1,output=str(output))
                r=evaluate(args)
            rows.append({'euler_steps':count,'set':label,
                         **{k:r[k] for k in ('successes','attempted','success_rate','wilson_95','sampler_ms_p50','sampler_ms_p95')},
                         'difference_vs_bc_percentage_points':100*(r['success_rate']-baselines[label]['success_rate']) if episodes==50 else None,
                         'failures':{reason:sum(e['failure_reason']==reason for e in r['episodes']) for reason in sorted({e['failure_reason'] for e in r['episodes'] if not e['success']})}})
    if baseline_hash!=file_hash(baseline_path):raise RuntimeError('BC baseline changed during evaluation')
    complete=episodes==50 and set(steps)=={4,6,8}
    result={'checkpoint_sha256':candidate_hash,'baseline_checkpoint_sha256':baseline_hash,
            'dataset_sha256':c['dataset_sha256'],'archive_sha256':file_hash(archive),
            'benchmark_complete':complete,'baseline':baselines,'cfm':rows,'onnx_euler_parity':parity,
            'target_80_percent_both_sets_by_euler':{str(n):all(r['success_rate']>=.8 for r in rows if r['euler_steps']==n) for n in steps} if complete else None,
            'scope':'Torch CPU, 2 threads; state/oracle context; no teacher/recovery; execute horizon 1 for BC and CFM; BC chunk 1, CFM chunk '+str(c['model_config']['chunk_size']),
            'noise_protocol':'numpy_default_rng(episode_seed + action_step); same starting noise at matching seed/action-step for 4/6/8'}
    (output_dir/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# CFM vs BC v2 CPU benchmark','',result['scope'],'',
           '| Policy | Set | Success | Sampler p50/p95 ms | Delta vs BC (pp) |','|---|---|---|---|---|']
    for label,r in baselines.items():lines.append(f"| BC v2 | {label} | {r['successes']}/{r['attempted']} | {r['sampler_ms_p50']:.3f}/{r['sampler_ms_p95']:.3f} | — |")
    for r in rows:
        delta=r['difference_vs_bc_percentage_points'];delta='smoke only' if delta is None else f'{delta:+.1f}'
        lines.append(f"| CFM Euler {r['euler_steps']} | {r['set']} | {r['successes']}/{r['attempted']} | {r['sampler_ms_p50']:.3f}/{r['sampler_ms_p95']:.3f} | {delta} |")
    (output_dir/'comparison.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True);p.add_argument('--dataset',default='data/drawer_dagger_v2.h5')
    p.add_argument('--destination',default='weights/cfm_kaggle_v2');p.add_argument('--output-dir',default='artifacts/cfm_kaggle_v2')
    p.add_argument('--episodes',type=int,default=50,help='50 for full benchmark; smaller counts are smoke checks only')
    p.add_argument('--euler-steps',nargs='+',type=int,default=[4,6,8])
    a=p.parse_args()
    if a.episodes<1 or a.episodes>50 or any(n<1 for n in a.euler_steps) or len(a.euler_steps)!=len(set(a.euler_steps)):
        p.error('episodes must be 1..50; Euler settings must be positive and unique')
    run(a.archive,a.dataset,a.destination,a.output_dir,a.episodes,a.euler_steps)
