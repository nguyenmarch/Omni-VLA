"""Finite-task modular vision-language BC bridge; no object/phase oracle input."""
import argparse
import json
from pathlib import Path
import time
from collections import deque
from itertools import count
import numpy as np
import torch
from models.grounding3d import GroundingRuntime
from simulation.vision import capture_head,robot_observation,VisualTaskManager,FULL_INSTRUCTION,OPEN_INSTRUCTION
from simulation.expert import PHASES
from utils.config import make_environment
from utils.checkpoint import load_policy,file_hash
from utils.camera_diagnostics import CameraDiagnostics


def evaluate(args):
    torch.set_num_threads(2);grounder=GroundingRuntime(args.grounding);model,c=load_policy(args.policy)
    if c['policy_type']!='bc' or c['model_config']['chunk_size']!=1:raise ValueError('Require single-action BC bridge')
    stats={k:v.numpy() for k,v in c['stats'].items()}
    env=make_environment(images=False,image_size=grounder.image_size,render_mode='human' if args.gui else None)
    if abs(env.dt-c['action_dt'])>1e-9:env.close();raise ValueError('BC timing mismatch')
    loop=getattr(args,'loop',False)
    episodes=deque(maxlen=100 if loop else None)
    perception_ms=deque(maxlen=10000 if loop else None);policy_ms=deque(maxlen=10000 if loop else None)
    perception_errors=deque(maxlen=5000 if loop else None)
    attempted=successes=0;stop_reason='episode_limit'
    monitor=None
    try:
        if getattr(args,'camera_debug',False):
            monitor=CameraDiagnostics(getattr(args,'debug_output','artifacts/camera_debug'),
                                      getattr(args,'debug_port',8765) if args.gui else None)
        for i in count() if loop else range(args.episodes):
            obs,_=env.reset(seed=args.seed+i);manager=None;failure=None;info=env.info();episode_errors=[]
            for step in range(env.max_episode_steps):
                start=time.perf_counter()
                robot=robot_observation(obs)  # strip all simulator object information
                if step%args.perception_every==0:
                    try:
                        t=time.perf_counter();frame=capture_head(env);points,metadata=grounder.predict(frame,args.ablation)
                        perception_ms.append((time.perf_counter()-t)*1000)
                        if manager is None:manager=VisualTaskManager(points,args.instruction)
                        else:manager.observe(points)
                        # Oracle landmarks are used only here for evaluation diagnostics.
                        errors=np.linalg.norm(points-np.asarray(obs['object'])[:6].reshape(2,3),axis=1)
                        diagnostic={'seed':args.seed+i,'step':step,'phase':PHASES[manager.phase],
                                    'bottle_error_m':float(errors[0]),'handle_error_m':float(errors[1])}
                        perception_errors.append(diagnostic);episode_errors.append(errors)
                    except ValueError as error:failure='perception_failed: '+str(error);break
                    if monitor is not None:
                        monitor.update(frame,np.asarray(obs['object'])[:6].reshape(2,3),points,metadata,
                                       args.seed+i,step,PHASES[manager.phase])
                condition=manager.condition(robot)
                t=time.perf_counter();x=torch.from_numpy(((condition-stats['condition_mean'])/stats['condition_std'])[None])
                action=model.sample_euler(x).numpy()[0,0]*stats['action_std']+stats['action_mean']
                action=manager.adapt_action(np.clip(action,-1,1));policy_ms.append((time.perf_counter()-t)*1000)
                obs,_,done,trunc,info=env.step(action);manager.update(robot_observation(obs))
                if args.gui:
                    if env.viewer is not None and not env.viewer.is_running():failure='viewer_closed';break
                    time.sleep(max(0,env.dt-(time.perf_counter()-start)))
                if manager.failed:failure=manager.failure_reason;break
                # Evaluation success flags are consulted only for termination/scoring, not manager/action decisions.
                if done or trunc or manager.completed:break
            if failure=='viewer_closed':stop_reason='viewer_closed';break
            success=bool(info['drawer_position']>.16) if args.instruction==OPEN_INSTRUCTION else bool(info['success'])
            row={'seed':args.seed+i,'success':success,'steps':info['step'],
                 'phase':PHASES[manager.phase] if manager else None,'failure_reason':None if success else failure or 'episode_limit_or_visual_completion',
                 'visual_completed':manager.completed if manager else False,
                 'landmark_mean_error_m':np.mean(episode_errors,axis=0).tolist() if episode_errors else None}
            episodes.append(row);attempted+=1;successes+=int(success);print(json.dumps(row),flush=True)
    except KeyboardInterrupt:
        stop_reason='keyboard_interrupt'
    finally:
        env.close()
        if monitor is not None:monitor.close()
    result={'mode':'modular_finite_language_vision_bc_bridge','instruction':args.instruction,'ablation':args.ablation,
            'grounding_sha256':file_hash(args.grounding),'policy_sha256':file_hash(args.policy),
            'episodes':list(episodes),'successes':successes,'attempted':attempted,
            'success_rate':successes/attempted if attempted else None,
            'loop':loop,'stop_reason':stop_reason,
            'retention':'Loop mode retains last 100 episodes, 5000 perception diagnostics and 10000 timing samples; counts cover completed episodes.' if loop else 'All completed episodes and samples',
            'perception_every_actions':args.perception_every,
            'perception_capture_and_infer_ms_p50':float(np.median(perception_ms)) if perception_ms else None,
            'policy_ms_p50':float(np.median(policy_ms)) if policy_ms else None,
            'perception_capture_and_infer_ms_p95':float(np.percentile(perception_ms,95)) if perception_ms else None,
            'policy_ms_p95':float(np.percentile(policy_ms,95)) if policy_ms else None,
            'perception_diagnostics':list(perception_errors),
            'limitations':'Frozen state BC with estimated landmarks; finite task grammar and known fixture geometry. Not jointly trained VLA. No object or phase oracle is passed to policy/manager.'}
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--grounding',required=True);p.add_argument('--policy',default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--instruction',choices=[FULL_INSTRUCTION,OPEN_INSTRUCTION],default=FULL_INSTRUCTION)
    p.add_argument('--ablation',choices=['none','drop-image','swap-text'],default='none')
    p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=9000);p.add_argument('--perception-every',type=int,default=5)
    p.add_argument('--gui',action='store_true');p.add_argument('--output',default='artifacts/visual_policy_eval.json')
    p.add_argument('--loop',action='store_true',help='Reset and repeat GUI episodes until viewer closes or Ctrl+C; ignores --episodes')
    p.add_argument('--camera-debug',action='store_true',help='RGB/depth overlays and numerical diagnostics; browser live view when --gui')
    p.add_argument('--debug-port',type=int,default=8765)
    p.add_argument('--debug-output',default='artifacts/camera_debug')
    a=p.parse_args()
    if a.episodes<1 or a.perception_every<1:p.error('episodes and perception-every must be positive')
    if a.loop and not a.gui:p.error('--loop requires --gui')
    if not 0<=a.debug_port<=65535:p.error('Invalid debug port')
    evaluate(a)
