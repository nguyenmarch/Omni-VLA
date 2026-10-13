"""Train patch-supervised grounding; select on accurate coverage, then penalized validation error."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.grounding3d import Grounding3D
from utils.patch_grounding_dataset import PatchGroundingFrames
from utils.checkpoint import atomic_save, file_hash
from simulation.vision import QUERIES


def summarize(errors, valid, eligible):
    errors=np.asarray(errors);valid=np.asarray(valid,dtype=bool);eligible=np.asarray(eligible,dtype=bool)
    accepted=errors[valid]
    return {'targets':len(errors),'label_eligible':int(eligible.sum()),'valid_predictions':int(valid.sum()),
        'coverage':float(valid.mean()),'valid_mean_error_m':float(accepted.mean()) if len(accepted) else None,
        'valid_p95_error_m':float(np.percentile(accepted,95)) if len(accepted) else None,
        'within_1_5cm_count':int((valid & (errors<=.015)).sum()),
        'within_1_5cm_all_fraction':float((valid & (errors<=.015)).mean()),
        'penalized_error_m':float(np.where(valid,errors,1.).mean())}


@torch.no_grad()
def evaluate(model,loader,ids,device,ablation='none'):
    errors=[];valid=[];eligible=[];targets=[];phases=[];pixels=[]
    gates={key:0 for key in ('has_peak','sufficient_support','stable_depth','in_workspace')}
    for sample in loader:
        s={k:v.to(device) for k,v in sample.items()}
        image=torch.zeros_like(s['rgb']) if ablation=='drop-image' else s['rgb']
        target=1-s['target'] if ablation=='swap-text' else s['target']
        heat,offset=model(image,ids[target]);r=model.decode_detailed(heat,offset,s['depth'],s['K'],s['T'])
        errors.extend(torch.linalg.vector_norm(r['points']-s['point'],dim=1).cpu().tolist())
        pixels.extend(torch.linalg.vector_norm(r['pixels']-s['uv'],dim=1).cpu().tolist())
        valid.extend(r['valid'].cpu().tolist());eligible.extend(s['label_valid'].cpu().tolist())
        targets.extend(s['target'].cpu().tolist());phases.extend(s['phase'].cpu().tolist())
        for key in gates:gates[key]+=int((~r[key]).sum())
    if not errors:raise ValueError('Empty evaluation split')
    result=summarize(errors,valid,eligible);result['mean_pixel_error']=float(np.mean(pixels))
    result['failed_gates_nonexclusive']=gates
    e=np.asarray(errors);v=np.asarray(valid);l=np.asarray(eligible)
    result['per_target']={query:summarize(e[np.asarray(targets)==i],v[np.asarray(targets)==i],l[np.asarray(targets)==i]) for i,query in enumerate(QUERIES)}
    result['per_phase']={str(p):summarize(e[np.asarray(phases)==p],v[np.asarray(phases)==p],l[np.asarray(phases)==p]) for p in sorted(set(phases))}
    return result


def selection_key(metric):
    return (-metric['within_1_5cm_count'],metric['penalized_error_m'])


def train(args):
    torch.manual_seed(42);torch.set_num_threads(2)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds=PatchGroundingFrames(args.dataset,'train');val=PatchGroundingFrames(args.dataset,'val')
    if not len(ds) or not len(val):raise ValueError('Require nonempty train/val')
    cfg={'img_size':ds.size,'feature_dim':64,'latent_dim':64,'decoder_config':ds.decoder_config}
    if val.decoder_config!=ds.decoder_config:raise ValueError('Decoder contract mismatch')
    model=Grounding3D(**cfg).to(device);ids=model.head.language.tokenize(QUERIES)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True);vl=DataLoader(val,batch_size=args.batch_size)
    y,x=torch.meshgrid(torch.arange(ds.size,device=device),torch.arange(ds.size,device=device),indexing='ij')
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new output directory')
    out.mkdir(parents=True);best=(float('inf'),float('inf'));data_hash=file_hash(args.dataset)
    print(json.dumps({'device':str(device),'train_targets':len(ds),'val_targets_all':len(val),'decoder_config':ds.decoder_config}),flush=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for sample in loader:
            s={k:v.to(device) for k,v in sample.items()}
            if not s['label_valid'].all():raise ValueError('Invalid training supervision')
            heat,offset=model(s['rgb'],ids[s['target']])
            d=(x[None]-s['uv'][:,0,None,None])**2+(y[None]-s['uv'][:,1,None,None])**2
            label=torch.softmax((-d/(2*args.heatmap_sigma**2)).flatten(1),dim=1)
            loss=-(label*F.log_softmax(heat.flatten(1),dim=1)).sum(1).mean()+F.smooth_l1_loss(offset/.05,s['offset']/.05)
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step();losses.append(loss.item())
        model.eval();metrics=evaluate(model,vl,ids,device)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**metrics};print(json.dumps(row,allow_nan=False),flush=True)
        payload={'schema_version':'omni-grounding-patch-v3','model_config':cfg,'model':model.state_dict(),
            'dataset_sha256':data_hash,'train_episodes':ds.episodes,'val_episodes':val.episodes,'instructions':QUERIES,
            'training_config':{'epochs':args.epochs,'batch_size':args.batch_size,'seed':42,'learning_rate':.0005,'heatmap_sigma_px':args.heatmap_sigma},
            'selection':'maximize valid predictions within 1.5cm / all validation targets; then minimize mean error with invalid=1m','metric':row}
        atomic_save(payload,out/'last.pt')
        atomic_save(payload,out/f'epoch_{epoch+1:03d}.pt')
        score=selection_key(metrics)
        if score<best:
            best=score;atomic_save(payload,out/'best.pt')
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',default='data/grounding_patch_v3_256.h5');p.add_argument('--output',default='weights/grounding_patch_v3_cpu_pilot')
    p.add_argument('--epochs',type=int,default=10);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device')
    p.add_argument('--heatmap-sigma',type=float,default=6.)
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1 or not np.isfinite(a.heatmap_sigma) or a.heatmap_sigma<=0:p.error('Invalid training parameters')
    train(a)
