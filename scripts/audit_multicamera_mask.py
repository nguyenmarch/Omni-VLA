"""Audit camera sizes and geometric robot mask during an expert rollout."""
import argparse
import base64
import json
from pathlib import Path
import time
import numpy as np
from simulation.bimanual_env import BimanualBottleDrawerEnv
from simulation.multicamera import MultiCameraCapture
from simulation.expert import ScriptedExpert
from utils.camera_diagnostics import png_bytes


def run(args):
    path=Path(args.output)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    env=BimanualBottleDrawerEnv(image_size=256);capture=None;rows=[];timings=[];saved=set()
    try:
        obs,_=env.reset(seed=args.seed);capture=MultiCameraCapture(env.model);expert=ScriptedExpert(env)
        for step in range(args.steps):
            if step%args.stride==0:
                start=time.perf_counter();frames=capture.capture(env.data,episode_id=args.seed,audit=True)
                timings.append(1000*(time.perf_counter()-start))
                for camera,f in frames.items():
                    a=f['excluded'];b=f['audit_robot_mask'];union=int((a|b).sum())
                    if args.snapshots and (camera,expert.phase) not in saved:
                        saved.add((camera,expert.phase));overlay=f['rgb'].copy();overlay[a]=[255,0,0]
                        panels=[f['rgb'],overlay,f['masked_rgb']]
                        encoded=[base64.b64encode(png_bytes(image)).decode() for image in panels]
                        html='<meta charset="utf-8"><h2>'+camera+' phase '+str(expert.phase)+'</h2><p>RGB / robot mask red / masked RGB</p>'+''.join('<img src="data:image/png;base64,'+image+'">' for image in encoded)
                        (path.parent/(path.stem+'_'+camera+'_phase'+str(expert.phase)+'.html')).write_text(html)
                    rows.append({'camera':camera,'step':step,'phase':expert.phase,'simulation_time':f['simulation_time'],
                        'size':list(f['rgb'].shape[:2]),'iou':int((a&b).sum())/union if union else 1.,
                        'false_mask_pixels':int((a&~b).sum()),'missed_robot_pixels':int((~a&b).sum()),'robot_pixels':int(b.sum())})
            expert.update();action=expert.action();obs,_,terminated,truncated,_=env.step(action)
            if terminated or truncated:break
        report={'backend':'MuJoCo MJCF FK robot-only render, not Pinocchio','scope':'Mask/capture audit only; no wrist-trained checkpoint or fusion controller',
            'seed':args.seed,'capture_with_audit_p50_ms':float(np.percentile(timings,50)),
            'capture_with_audit_p95_ms':float(np.percentile(timings,95)),'frames':rows}
        path.write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({k:v for k,v in report.items() if k!='frames'},indent=2))
    finally:
        if capture:capture.close()
        env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--seed',type=int,default=9000);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--snapshots',action='store_true');p.add_argument('--stride',type=int,default=25);p.add_argument('--output',default='artifacts/patch_grounding_v3/multicamera_mask.json')
    args=p.parse_args()
    if args.steps<1 or args.stride<1:p.error('Positive steps/stride required')
    run(args)
