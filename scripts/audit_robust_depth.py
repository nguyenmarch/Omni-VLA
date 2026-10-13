"""Validation-only experiment: robust decoder with legacy offset weights, no deployment claim."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from models.grounding3d import GroundingRuntime
from utils.robust_depth import RobustDepthDecoder
from simulation.vision import QUERIES
from utils.checkpoint import file_hash


@torch.no_grad()
def audit(args):
    torch.set_num_threads(2);runtime=GroundingRuntime(args.checkpoint)
    decoders={name:RobustDepthDecoder(quantile=q) for name,q in [('median',.5),('q20',.2)]}
    ids=runtime.model.head.language.tokenize(QUERIES);rows={name:[] for name in decoders}
    with h5py.File(args.dataset) as h:
        names=json.loads(h.attrs['splits'])['val']
        if set(names)&set(runtime.checkpoint['train_episodes']):raise ValueError('Validation leakage')
        if file_hash(args.dataset)!=runtime.checkpoint['dataset_sha256']:raise ValueError('Dataset/checkpoint mismatch')
        for name in names:
            g=h['episodes'][name]
            for index in range(len(g['rgb'])):
                rgb=torch.from_numpy(g['rgb'][index].transpose(2,0,1).copy()).float()[None].expand(2,-1,-1,-1)/255
                heat,offset=runtime.model(rgb,ids)
                depth=torch.from_numpy(g['depth'][index])[None].expand(2,-1,-1)
                k=torch.from_numpy(g['K'][index])[None].expand(2,-1,-1)
                transform=torch.from_numpy(g['T'][index])[None].expand(2,-1,-1)
                truth=torch.from_numpy(g['targets'][index])
                for label,decoder in decoders.items():
                    decoded=decoder(heat,depth,k,transform,offset)
                    errors=torch.linalg.vector_norm(decoded['points']-truth,dim=1)
                    for target in range(2):
                        valid=bool(decoded['valid'][target])
                        rows[label].append({'episode':name,'frame':index,'phase':int(g['phase'][index]),
                            'target':QUERIES[target],'valid':valid,'error_m':float(errors[target]) if valid else None,
                            'support':int(decoded['support'][target]),'in_workspace':bool(decoded['in_workspace'][target]),
                            'stable_depth':bool(decoded['stable_depth'][target])})
    results={}
    for name,samples in rows.items():
        def summary(selected):
            errors=[r['error_m'] for r in selected if r['valid']]
            return {'targets':len(selected),'valid':len(errors),'coverage':len(errors)/len(selected),
                    'valid_mean_error_m':float(np.mean(errors)) if errors else None,
                    'within_1_5cm_fraction_all_targets':sum(e<=.015 for e in errors)/len(selected)}
        results[name]={**summary(samples),'per_target':{q:summary([r for r in samples if r['target']==q]) for q in QUERIES},
            'samples':samples}
    report={'split':'val','episodes':names,'checkpoint_sha256':file_hash(args.checkpoint),
            'dataset_sha256':file_hash(args.dataset),'decoder':'7x7, min support 6 and fraction .2, IQR <=.10m, workspace rejection',
            'limitations':'Legacy offsets trained on single-pixel labels; recalibration/retraining pending. All val frames including occlusion; rejection coverage must be reported. No self-body mask or wrist fusion yet. Not comparable to visible-only benchmark means.',
            'results':results}
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({name:{k:v for k,v in r.items() if k not in ('samples','per_target')} for name,r in results.items()},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',default='weights/grounding_rgbd_v2_256_cpu_pilot/best.pt')
    p.add_argument('--dataset',default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--output',default='artifacts/advanced_perception/robust_depth_val.json')
    audit(p.parse_args())
