"""Save RGB/depth overlays and calibration/label diagnostics from held-out frames."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
import torch
from models.grounding3d import GroundingRuntime
from simulation.expert import PHASES
from utils.camera_diagnostics import CameraDiagnostics,depth_at
from utils.camera import project
from utils.checkpoint import file_hash


def audit(args):
    torch.set_num_threads(2);runtime=GroundingRuntime(args.grounding)
    monitor=CameraDiagnostics(args.output);rows=[];label_scan=[]
    try:
        with h5py.File(args.dataset) as h:
            if h.attrs['schema_version']!='omni-grounding-rgbd-v2':raise ValueError('Wrong dataset schema')
            names=[args.episode] if args.episode else json.loads(h.attrs['splits'])['test']
            for name in names:
                g=h['episodes'][name];phases=g['phase'][:]
                indices=sorted({int(np.flatnonzero(phases==p)[0]) for p in np.unique(phases)})[:args.max_frames]
                counts={};flagged=[]
                for index in range(len(phases)):
                    depth,k,t=g['depth'][index],g['K'][index],g['T'][index]
                    for target,point in enumerate(g['targets'][index]):
                        uv,z=project(point,k,t);surface=depth_at(depth,uv,z)
                        agrees=surface is not None and abs(float(z)-surface)<=.08
                        status='depth_agreement_within_8cm' if agrees else 'outside_view_invalid_depth_or_depth_disagreement'
                        counts[status]=counts.get(status,0)+1
                        if not agrees:flagged.append({'frame':index,'step':int(g['step'][index]),
                            'phase':PHASES[int(phases[index])],'target':('bottle','handle')[target],
                            'center_depth_m':float(z),'surface_depth_m':surface})
                if flagged and len(indices)<args.max_frames:
                    indices=sorted(set(indices+[flagged[0]['frame']]))
                label_scan.append({'episode':name,'frames':len(phases),'visibility_counts':counts,
                                   'flagged_examples':flagged[:12]})
                for index in indices:
                    frame={key:g[key][index] for key in ('rgb','depth','K','T')}
                    points,metadata=runtime.predict(frame)
                    row=monitor.update(frame,g['targets'][index],points,metadata,int(g.attrs['seed']),
                                       int(g['step'][index]),PHASES[int(phases[index])],force_snapshot=True)
                    rows.append({k:v for k,v in row.items() if not k.endswith('_url')})
            scene_hash=str(h.attrs['scene_sha256'])
        checks=[target for row in rows for target in row['targets']]
        report={'frames':len(rows),'episodes':names,'selection':'First recorded frame of each phase plus first flagged visibility frame if space; diagnostic sample, not aggregate test metric',
                'all_frame_label_visibility_scan':label_scan,
                'dataset_sha256':file_hash(args.dataset),'grounding_sha256':file_hash(args.grounding),
                'dataset_scene_sha256':scene_hash,
                'max_gt_roundtrip_error_m':max(t['gt_roundtrip_error_m'] or 0 for t in checks),
                'max_prediction_reprojection_error_px':max(t['prediction_reprojection_error_px'] for t in checks),
                'visibility_heuristic_counts':{v:sum(t['visibility_heuristic']==v for t in checks) for v in {t['visibility_heuristic'] for t in checks}},
                'samples':rows}
        (Path(args.output)/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='samples'},indent=2))
    finally:monitor.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grounding',default='weights/grounding_rgbd_v2_cpu_pilot/best.pt')
    p.add_argument('--dataset',default='data/grounding_rgbd_v2.h5');p.add_argument('--episode')
    p.add_argument('--max-frames',type=int,default=10);p.add_argument('--output',default='artifacts/camera_label_audit')
    a=p.parse_args()
    if a.max_frames<1:p.error('--max-frames must be positive')
    audit(a)
