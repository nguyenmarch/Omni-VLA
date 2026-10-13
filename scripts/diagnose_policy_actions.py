"""Compare first-action errors by phase on fixed validation states, not rollouts."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from simulation.expert import PHASES
from utils.checkpoint import load_policy, file_hash


def summarize_actions(predictions, targets, phases):
    """predictions shape [noise draws, states, 14], in raw action units."""
    clipped=np.clip(predictions,-1,1);result={}
    for phase in sorted(set(phases.tolist())):
        ids=phases==phase;p=predictions[:,ids];q=clipped[:,ids];y=targets[ids][None]
        error=np.abs(q-y);grip_y=y[..., [6,13]];grip_q=q[..., [6,13]]
        valid=np.broadcast_to(np.abs(grip_y)>.5,grip_q.shape)
        mismatch=(grip_q<0)!=(grip_y<0)
        result[PHASES[phase]]={'states':int(ids.sum()),
                              'first_action_mse':float(np.mean((q-y)**2)),
                              'first_action_mae_per_dim':error.mean(axis=(0,1)).tolist(),
                              'xyz_mae':float(error[..., [0,1,2,7,8,9]].mean()),
                              'rotation_mae':float(error[..., [3,4,5,10,11,12]].mean()),
                              'gripper_mae':float(error[..., [6,13]].mean()),
                              'gripper_sign_mismatch_fraction':float(mismatch[valid].mean()) if valid.any() else None,
                              'raw_action_out_of_bounds_fraction':float((np.abs(p)>1).mean()),
                              'noise_action_std_mean':float(q.std(axis=0).mean())}
    return result


def diagnose(checkpoints, dataset, output, steps=(4,6,8), draws=3):
    torch.set_num_threads(2)
    conditions=[];actions=[];phases=[]
    with h5py.File(dataset) as h:
        split=json.loads(h.attrs['splits'])['val']
        for name in split:
            g=h['episodes'][name]
            if not g.attrs.get('labels_valid',g.attrs['success']):continue
            conditions.append(g['condition'][:]);actions.append(g['action'][:]);phases.append(g['phase'][:])
    if not conditions:raise ValueError('No valid validation labels')
    conditions=np.concatenate(conditions);actions=np.concatenate(actions);phases=np.concatenate(phases)
    models=[(p,*load_policy(p)) for p in checkpoints]
    maximum=max(c['model_config']['chunk_size'] for _,_,c in models)
    noise=np.random.default_rng(42).normal(size=(draws,len(actions),maximum,14)).astype(np.float32)
    results=[];dataset_hash=file_hash(dataset)
    for path,model,c in models:
        if c['dataset_sha256']!=dataset_hash:raise ValueError('Require identical dataset for diagnostics')
        stats={k:v.numpy() for k,v in c['stats'].items()}
        normalized=(conditions-stats['condition_mean'])/stats['condition_std']
        for count in steps if c['policy_type']=='cfm' else [1]:
            samples=[]
            for draw in range(draws if c['policy_type']=='cfm' else 1):
                batches=[]
                for offset in range(0,len(actions),256):
                    x=torch.from_numpy(normalized[offset:offset+256])
                    z=torch.from_numpy(noise[draw,offset:offset+256,:c['model_config']['chunk_size']])
                    with torch.no_grad():a=model.sample_euler(x,count,noise=z).numpy()[:,0]
                    batches.append(a*stats['action_std']+stats['action_mean'])
                samples.append(np.concatenate(batches))
            results.append({'checkpoint':str(path),'checkpoint_sha256':file_hash(path),
                            'policy':c['policy_type'],'chunk_size':c['model_config']['chunk_size'],
                            'euler_steps':count,'phases':summarize_actions(np.stack(samples),actions,phases)})
    r={'dataset_sha256':dataset_hash,'split':'val','episodes':split,'states':len(actions),
       'noise_draws':draws,'results':results,
       'scope':'Fixed validation states, first executed action after clipping; not success/closed-loop. Same noise prefixes across chunks/settings. Never used as training data.'}
    Path(output).parent.mkdir(parents=True,exist_ok=True);Path(output).write_text(json.dumps(r,indent=2)+'\n')
    return r


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--dataset',default='data/drawer_dagger_v2.h5')
    p.add_argument('--output',required=True);p.add_argument('--noise-draws',type=int,default=3)
    a=p.parse_args()
    if a.noise_draws<2:p.error('Use at least two noise draws to estimate variability')
    r=diagnose(a.checkpoints,a.dataset,a.output,draws=a.noise_draws)
    print(json.dumps({'states':r['states'],'models/settings':len(r['results']),'output':a.output}))
