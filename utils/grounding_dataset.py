"""Episode-disjoint sparse RGB-D targets with explicit visibility heuristic."""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.camera import project


class GroundingFrames(Dataset):
    def __init__(self,path,split):
        self.index=[];self.frames={};self.skipped=0
        with h5py.File(path) as h:
            if h.attrs['schema_version']!='omni-grounding-rgbd-v2':raise ValueError('Wrong RGB-D schema')
            splits=json.loads(h.attrs['splits']);names=[n for values in splits.values() for n in values]
            if len(names)!=len(set(names)):raise ValueError('Episode leakage')
            self.episodes=splits[split];self.size=next(iter(h['episodes'].values()))['rgb'].shape[1]
            for name in self.episodes:
                g=h['episodes'][name]
                if not g.attrs['success']:continue
                frame={k:g[k][:] for k in ('rgb','depth','K','T','targets','phase')};self.frames[name]=frame
                if frame['rgb'].shape[1:]!=(self.size,self.size,3) or frame['depth'].shape!=(len(frame['rgb']),self.size,self.size):
                    raise ValueError('Inconsistent RGB/depth frame sizes')
                for step in range(len(frame['rgb'])):
                    for target in range(2):
                        point=frame['targets'][step,target];uv,z=project(point,frame['K'][step],frame['T'][step])
                        if not np.isfinite(uv).all() or z<=0 or not ((uv>=0)&(uv<self.size)).all():self.skipped+=1;continue
                        u,v=np.rint(uv).astype(int).clip(0,self.size-1)
                        surface=frame['depth'][step,v,u];offset=z-surface
                        # Only a depth-based visibility heuristic, not segmentation-certified visibility.
                        if not np.isfinite(surface) or surface<=0 or abs(offset)>.08:self.skipped+=1;continue
                        self.index.append((name,step,target,uv.astype(np.float32),np.float32(offset)))
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        name,step,target,uv,offset=self.index[index];f=self.frames[name]
        return {'rgb':torch.from_numpy(f['rgb'][step].transpose(2,0,1).copy()).float()/255,
                'depth':torch.from_numpy(f['depth'][step]),'K':torch.from_numpy(f['K'][step]),'T':torch.from_numpy(f['T'][step]),
                'point':torch.from_numpy(f['targets'][step,target]),'uv':torch.from_numpy(uv),
                'offset':torch.tensor(offset),'target':torch.tensor(target),'phase':torch.tensor(f['phase'][step])}
