"""Train finite-language RGB-D grounding; evaluate 3D, image/text ablations."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.grounding3d import Grounding3D
from utils.grounding_dataset import GroundingFrames
from utils.checkpoint import atomic_save,file_hash
from simulation.vision import QUERIES


@torch.no_grad()
def evaluate_grounder(model,loader,ids,device,ablation='none'):
    distances=[];pixels=[];depth_errors=[];by_target={0:[],1:[]};by_phase={};invalid=0
    for sample in loader:
        s={k:v.to(device) for k,v in sample.items()};image=s['rgb']
        if ablation=='drop-image':image=torch.zeros_like(image)
        target=1-s['target'] if ablation=='swap-text' else s['target']
        heat,offset=model(image,ids[target]);points,uv,z=model.decode(heat,offset,s['depth'],s['K'],s['T'])
        error=torch.linalg.vector_norm(points-s['point'],dim=1).cpu().numpy()
        pixel=torch.linalg.vector_norm(uv-s['uv'],dim=1).cpu().numpy()
        truth_h=torch.cat([s['point'],torch.ones_like(z[:,None])],dim=1)
        truth_z=torch.linalg.solve(s['T'],truth_h[:,:,None]).squeeze(-1)[:,2]
        depth_errors.extend((z-truth_z).abs().cpu().tolist())
        valid=(torch.isfinite(points).all(1)&(z>.05)&(z<5)).cpu().numpy()
        invalid+=int((~valid).sum())
        distances.extend(error.tolist());pixels.extend(pixel.tolist())
        for t,e in zip(sample['target'].tolist(),error):by_target[t].append(float(e))
        for phase,e in zip(sample['phase'].tolist(),error):by_phase.setdefault(int(phase),[]).append(float(e))
    if not distances:raise ValueError('No validation targets')
    return {'targets':len(distances),'invalid_predictions':invalid,'mean_3d_error_m':float(np.mean(distances)),
            'p95_3d_error_m':float(np.percentile(distances,95)),'mean_pixel_error':float(np.mean(pixels)),
            'mean_abs_camera_depth_error_m':float(np.mean(depth_errors)),
            'within_1_5cm_fraction':float(np.mean(np.asarray(distances)<=.015)),
            'per_phase_mean_3d_m':{str(p):float(np.mean(errors)) for p,errors in by_phase.items()},
            'per_target_mean_3d_m':{QUERIES[t]:float(np.mean(errors)) if errors else None for t,errors in by_target.items()}}


def train(args):
    torch.manual_seed(42);torch.set_num_threads(2)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ds=GroundingFrames(args.dataset,'train');val=GroundingFrames(args.dataset,'val')
    if not len(ds) or not len(val):raise ValueError('Need valid train/val RGB-D targets')
    cfg={'img_size':ds.size,'feature_dim':64,'latent_dim':64}
    model=Grounding3D(**cfg).to(device);ids=model.head.language.tokenize(QUERIES)
    opt=torch.optim.AdamW(model.parameters(),lr=.0005)
    train_loader=DataLoader(ds,batch_size=args.batch_size,shuffle=True);val_loader=DataLoader(val,batch_size=args.batch_size)
    y,x=torch.meshgrid(torch.arange(ds.size,device=device),torch.arange(ds.size,device=device),indexing='ij')
    out=Path(args.output)
    if out.exists():raise FileExistsError('Use a new output path for this run')
    out.mkdir(parents=True);best=float('inf')
    dataset_hash=file_hash(args.dataset)
    sigma=getattr(args,'heatmap_sigma',3.)
    print(json.dumps({'device':str(device),'train_targets':len(ds),'val_targets':len(val),'visibility':'depth agreement <=8cm heuristic','skipped_train':ds.skipped}),flush=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for sample in train_loader:
            s={k:v.to(device) for k,v in sample.items()}
            heat,offset=model(s['rgb'],ids[s['target']])
            d=(x[None]-s['uv'][:,0,None,None])**2+(y[None]-s['uv'][:,1,None,None])**2
            labels=torch.softmax((-d/(2*sigma*sigma)).flatten(1),dim=1)
            loss=-(labels*F.log_softmax(heat.flatten(1),dim=1)).sum(1).mean()+F.smooth_l1_loss(offset/.05,s['offset']/.05)
            if not torch.isfinite(loss):raise ValueError('Nonfinite grounding loss')
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();losses.append(loss.item())
        model.eval();metric=evaluate_grounder(model,val_loader,ids,device)
        row={'epoch':epoch+1,'loss':float(np.mean(losses)),**metric};print(json.dumps(row),flush=True)
        payload={'schema_version':'omni-grounding-rgbd-v2','model_config':cfg,'model':model.state_dict(),
                 'dataset_sha256':dataset_hash,'train_episodes':ds.episodes,'val_episodes':val.episodes,
                 'training_config':{'epochs':args.epochs,'batch_size':args.batch_size,'seed':42,'learning_rate':.0005,'heatmap_sigma_px':sigma},
                 'instructions':QUERIES,'metric':row,'visibility_heuristic_m':.08}
        atomic_save(payload,out/'last.pt')
        if metric['mean_3d_error_m']<best:best=metric['mean_3d_error_m'];atomic_save(payload,out/'best.pt')
        with (out/'metrics.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
    c=torch.load(out/'best.pt',map_location=device,weights_only=True);model.load_state_dict(c['model']);model.eval()
    report={a:evaluate_grounder(model,val_loader,ids,device,a) for a in ('none','drop-image','swap-text')}
    report['scope']='Visible-by-depth-heuristic validation targets; finite queries, current camera, no OOD/paraphrase guarantee'
    (out/'validation_ablations.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',default='weights/grounding_rgbd_v2')
    p.add_argument('--epochs',type=int,default=40);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--device')
    p.add_argument('--heatmap-sigma',type=float,default=3.,help='Gaussian label sigma in pixels; use 6 at 256 to match 3 at 128 in angular size')
    a=p.parse_args()
    if a.epochs<1 or a.batch_size<1:p.error('epochs and batch-size must be positive')
    if not np.isfinite(a.heatmap_sigma) or a.heatmap_sigma<=0:p.error('heatmap-sigma must be finite and positive')
    train(a)
