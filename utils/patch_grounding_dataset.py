"""Patch-supervised train targets; evaluation retains every physical target."""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class PatchGroundingFrames(Dataset):
    def __init__(self,path,split):
        self.frames={};self.index=[];self.eligible=0
        with h5py.File(path) as h:
            if h.attrs['schema_version']!='omni-grounding-patch-v3':raise ValueError('Require patch supervision schema')
            self.decoder_config=json.loads(h.attrs['decoder_config'])
            splits=json.loads(h.attrs['splits']);names=[n for group in splits.values() for n in group]
            if len(names)!=len(set(names)):raise ValueError('Episode leakage')
            self.episodes=splits[split];self.size=next(iter(h['episodes'].values()))['rgb'].shape[1]
            self.total_targets=0
            for name in self.episodes:
                g=h['episodes'][name]
                if not g.attrs['success']:raise ValueError('Unexpected failed expert episode')
                keys=('rgb','depth','K','T','targets','phase','patch_uv','patch_offset','patch_label_valid')
                frame={key:g[key][:] for key in keys};self.frames[name]=frame
                if frame['rgb'].shape[1:]!=(self.size,self.size,3):raise ValueError('Image size mismatch')
                for step in range(len(frame['rgb'])):
                    for target in range(2):
                        eligible=bool(frame['patch_label_valid'][step,target]);self.eligible+=int(eligible);self.total_targets+=1
                        if split=='train' and not eligible:continue
                        if eligible and (not np.isfinite(frame['patch_offset'][step,target]) or abs(frame['patch_offset'][step,target])>.08):
                            raise ValueError('Invalid eligible offset label')
                        self.index.append((name,step,target))
    def __len__(self):return len(self.index)
    def __getitem__(self,index):
        name,step,target=self.index[index];f=self.frames[name]
        return {'rgb':torch.from_numpy(f['rgb'][step].transpose(2,0,1).copy()).float()/255,
            'depth':torch.from_numpy(f['depth'][step]),'K':torch.from_numpy(f['K'][step]),'T':torch.from_numpy(f['T'][step]),
            'point':torch.from_numpy(f['targets'][step,target]),'uv':torch.from_numpy(f['patch_uv'][step,target]),
            'offset':torch.tensor(f['patch_offset'][step,target]),'target':torch.tensor(target),
            'phase':torch.tensor(f['phase'][step]),'label_valid':torch.tensor(f['patch_label_valid'][step,target])}
