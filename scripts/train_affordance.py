"""Finite-language supervised 2D grounding on collected RGB-D episodes.

Labels use simulator object centers. Reported pixel errors do not establish
contact-surface accuracy, occlusion robustness, or end-to-end VLA success.
"""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset,DataLoader
from torch.nn import functional as F
from models.affordance_head import VisualAffordanceHead
from utils.camera import project
from utils.checkpoint import atomic_save

INSTRUCTIONS=('grasp the bottle body','grasp the drawer handle')

class GroundingDataset(Dataset):
    def __init__(self,path,split,stride=8):
        self.path=path;self.index=[]
        with h5py.File(path,'r') as h:
            names=json.loads(h.attrs['splits'])[split]
            for name in names:
                g=h['episodes/'+name]
                if 'rgb_head' not in g['obs']:
                    raise ValueError('Collect with --images before training affordance')
                for t in range(0,len(g['action']),stride):
                    k,transform=g['obs/K_head'][t],g['obs/T_head'][t]
                    objects=g['obs/object'][t]
                    for target in range(2):
                        uv,z=project(objects[target*3:target*3+3],k,transform)
                        size=g['obs/rgb_head'].shape[1]
                        if z>0 and (uv>=0).all() and (uv<size).all():
                            # Reject heavily occluded centers; small tolerance allows object radius.
                            depth=g['obs/depth_head'][t,int(uv[1]),int(uv[0])]
                            if z-depth < .08:
                                self.index.append((name,t,target,uv.astype(np.float32)))
    def __len__(self):return len(self.index)
    def __getitem__(self,i):
        name,t,target,uv=self.index[i]
        with h5py.File(self.path,'r') as h:
            rgb=h[f'episodes/{name}/obs/rgb_head'][t]
        return torch.from_numpy(rgb.transpose(2,0,1).copy()).float()/255,torch.tensor(target),torch.from_numpy(uv)


def train(args):
    torch.set_num_threads(2);torch.manual_seed(42)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    train_ds=GroundingDataset(args.dataset,'train');val_ds=GroundingDataset(args.dataset,'val')
    if not len(train_ds) or not len(val_ds):raise ValueError('Need visible targets in train/val splits')
    with h5py.File(args.dataset,'r') as h:
        first=next(iter(h['episodes'].values()));size=first['obs/rgb_head'].shape[1]
    model=VisualAffordanceHead(img_size=size).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0005)
    instruction_ids=model.language.tokenize(INSTRUCTIONS)
    y,x=torch.meshgrid(torch.arange(size,device=device),torch.arange(size,device=device),indexing='ij')
    train_loader=DataLoader(train_ds,batch_size=args.batch_size,shuffle=True)
    val_loader=DataLoader(val_ds,batch_size=args.batch_size)
    best=float('inf');output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    for epoch in range(args.epochs):
        model.train();losses=[]
        for rgb,target,uv in train_loader:
            rgb,target,uv=rgb.to(device),target.to(device),uv.to(device)
            logits,_,_,_=model.from_rgb(rgb,instruction_ids[target])
            distance=(x[None]-uv[:,0,None,None])**2+(y[None]-uv[:,1,None,None])**2
            distribution=torch.softmax((-distance/18).flatten(1),dim=1)
            loss=-(distribution*F.log_softmax(logits.flatten(1),dim=1)).sum(1).mean()
            optimizer.zero_grad();loss.backward();optimizer.step();losses.append(loss.item())
        model.eval();errors=[]
        with torch.no_grad():
            for rgb,target,uv in val_loader:
                logits,_,_,_=model.from_rgb(rgb.to(device),instruction_ids[target.to(device)])
                peak=logits.flatten(1).argmax(1)
                predictions=torch.stack([peak%size,peak//size],dim=1).float().cpu()
                errors.extend(torch.linalg.vector_norm(predictions-uv,dim=1).tolist())
        metric=float(np.mean(errors));row={'epoch':epoch+1,'loss':float(np.mean(losses)),'val_pixel_error':metric,'visible_train_targets':len(train_ds),'visible_val_targets':len(val_ds)}
        print(json.dumps(row),flush=True)
        payload={'schema_version':'omni-grounding-v1','model':model.state_dict(),'image_size':size,'instructions':INSTRUCTIONS,'metric':row}
        atomic_save(payload,output/'last.pt')
        if metric<best:best=metric;atomic_save(payload,output/'best.pt')
        with (output/'metrics.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True);p.add_argument('--output',default='weights/grounding');p.add_argument('--epochs',type=int,default=30);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--device')
    train(p.parse_args())
