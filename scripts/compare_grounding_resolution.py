"""Paired-camera trajectory audit and grounding comparison on common held-out targets."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
from models.grounding3d import GroundingRuntime
from utils.grounding_dataset import GroundingFrames
from utils.checkpoint import file_hash
from scripts.train_grounding_rgbd import evaluate_grounder
from simulation.vision import QUERIES


def audit_pair(low_path,high_path):
    with h5py.File(low_path) as low,h5py.File(high_path) as high:
        if low.attrs['scene_sha256']!=high.attrs['scene_sha256']:raise ValueError('Scene mismatch')
        if json.loads(low.attrs['splits'])!=json.loads(high.attrs['splits']):raise ValueError('Episode split mismatch')
        if list(low['episodes'])!=list(high['episodes']):raise ValueError('Episode names mismatch')
        frames=0
        for name in low['episodes']:
            a,b=low['episodes'][name],high['episodes'][name]
            if a.attrs['seed']!=b.attrs['seed'] or a.attrs['success']!=b.attrs['success']:raise ValueError('Seed/outcome mismatch')
            for key in ('targets','phase','step','q','dq','ee','gripper','wrench','action','condition'):
                if not np.array_equal(a[key][:],b[key][:]):raise ValueError(f'Trajectory mismatch {name}/{key}')
            ratio=b['rgb'].shape[1]/a['rgb'].shape[1]
            expected=a['K'][:];expected[:,:2,:]*=ratio
            if not np.allclose(expected,b['K'][:],atol=1e-5):raise ValueError('Intrinsics did not scale with resolution')
            if not np.array_equal(a['T'][:],b['T'][:]):raise ValueError('Extrinsics mismatch')
            frames+=len(a['rgb'])
        return {'episodes':len(low['episodes']),'frames':frames,'trajectory_values_identical':True,
                'same_scene_splits_seeds':True,'scaled_intrinsics_and_identical_extrinsics':True}


def compare(args):
    torch.set_num_threads(2)
    report={'paired_dataset_audit':audit_pair(args.low_dataset,args.high_dataset),'splits':{}}
    for split in ('val','test'):
        datasets=[GroundingFrames(path,split) for path in (args.low_dataset,args.high_dataset)]
        maps=[{(name,step,target):i for i,(name,step,target,*_) in enumerate(ds.index)} for ds in datasets]
        common=sorted(set(maps[0])&set(maps[1]))
        if not common:raise ValueError('No common visible targets')
        row={'common_targets':len(common),'visibility_scope':'Depth agreement <=8cm in both resolutions; excluded targets are not scored',
             'models':{}}
        for label,path,dataset_path,ds,index in zip(('128','256'),(args.low_checkpoint,args.high_checkpoint),
                (args.low_dataset,args.high_dataset),datasets,maps):
            runtime=GroundingRuntime(path)
            if runtime.checkpoint['dataset_sha256']!=file_hash(dataset_path):raise ValueError('Checkpoint/dataset mismatch')
            if ds.size!=runtime.image_size:raise ValueError('Resolution mismatch')
            if set(ds.episodes)&set(runtime.checkpoint['train_episodes']):raise ValueError('Train/held-out leakage')
            if split=='test' and set(ds.episodes)&set(runtime.checkpoint['val_episodes']):raise ValueError('Validation/test leakage')
            ids=runtime.model.head.language.tokenize(QUERIES)
            loader=DataLoader(Subset(ds,[index[key] for key in common]),batch_size=32)
            metrics={a:evaluate_grounder(runtime.model,loader,ids,'cpu',a) for a in ('none','drop-image','swap-text')}
            row['models'][label]={'image_size':ds.size,'available_targets':len(ds),'skipped_targets':ds.skipped,
                 'checkpoint_sha256':file_hash(path),'best_epoch':runtime.checkpoint['metric']['epoch'],
                 'metrics':metrics}
        report['splits'][split]=row
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v['metrics']['none']['mean_3d_error_m'] for k,v in r['models'].items()} for s,r in report['splits'].items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--low-dataset',default='data/grounding_rgbd_v2.h5')
    p.add_argument('--high-dataset',default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--low-checkpoint',default='weights/grounding_rgbd_v2_cpu_pilot/best.pt')
    p.add_argument('--high-checkpoint',default='weights/grounding_rgbd_v2_256_cpu_pilot/best.pt')
    p.add_argument('--output',default='artifacts/vision_language_256_pilot/paired_grounding.json')
    compare(p.parse_args())
