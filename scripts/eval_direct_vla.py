"""Closed-loop direct VLA evaluation; no landmarks/phase/task FSM in action path."""
import argparse
import json
from collections import deque
from pathlib import Path
import time
import numpy as np
import torch
from models.direct_vla import DirectVLA, measured_image, proprioception, tokenize, TASK_PHRASES
from simulation.vision import capture_head
from simulation.randomization import DomainRandomizer
from utils.config import make_environment
from utils.checkpoint import file_hash
from utils.reporting import write_json,wilson


def evaluate(args):
    if Path(args.output).exists():raise FileExistsError(args.output)
    torch.set_num_threads(2)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    if c['schema']!='omni-direct-vla-policy-v1':raise ValueError('Wrong policy schema')
    model=DirectVLA(**c['model_config']);model.load_state_dict(c['model']);model.eval()
    instruction=args.instruction or TASK_PHRASES[args.task][0]
    if getattr(args,'ablation','none')=='swap-text':instruction=TASK_PHRASES[1-args.task][0]
    tokens=tokenize([instruction],c['model_config']['vocabulary'])
    env=make_environment(image_size=c['image_size']);randomizer=DomainRandomizer(env)
    if abs(env.dt-c['action_dt'])>1e-9:env.close();raise ValueError('Action timing mismatch')
    stats={k:v.numpy() for k,v in c['robot_stats'].items()};episodes=[];latencies=[]
    try:
        for seed in range(args.seed,args.seed+args.episodes):
            obs,_=env.reset(seed=seed);params=randomizer.apply(seed,args.profile);obs=env._get_observation()
            history=deque(maxlen=c['history']);buffer=deque();info=env.info()
            for step in range(env.max_episode_steps):
                if not buffer:
                    t=time.perf_counter();frame=capture_head(env)
                    image=measured_image(frame)
                    if getattr(args,'ablation','none')=='drop-image':image[:]=0
                    if getattr(args,'ablation','none')=='drop-history':history.clear()
                    history.append((image,(proprioception(obs)-stats['mean'])/stats['std']))
                    images=np.zeros((1,c['history'],7,c['image_size'],c['image_size']),np.float32)
                    robot=np.zeros((1,c['history'],50),np.float32)
                    for i,(image,state) in enumerate(history):images[0,i]=image;robot[0,i]=state
                    mask=torch.arange(c['history'])[None]<len(history)
                    with torch.inference_mode():
                        pred=model(torch.from_numpy(images),torch.from_numpy(robot),tokens,mask)
                    chunk=pred['action'][0].numpy()
                    if not np.isfinite(chunk).all():raise ValueError('Nonfinite action')
                    buffer.extend(np.clip(chunk,-1,1));latencies.append((time.perf_counter()-t)*1000)
                obs,_,done,trunc,info=env.step(buffer.popleft())
                # Oracle evaluator metrics never enter model/action selection.
                success=info['drawer_position']>.16 if args.task==1 else info['success']
                if success or done or trunc:break
            episodes.append({'seed':seed,'success':bool(success),'steps':info['step'],'randomization':params})
            print(json.dumps(episodes[-1]),flush=True)
    finally:env.close()
    n=len(episodes);successes=sum(e['success'] for e in episodes)
    report={'mode':'direct_rgbd_language_recurrent_vla','instruction':instruction,'task_scoring_only':args.task,
            'ablation':getattr(args,'ablation','none'),
            'checkpoint_sha256':file_hash(args.checkpoint),'dataset_sha256':c['dataset_sha256'],
            'smoke_checkpoint':c['smoke_only'],'episodes':episodes,'successes':successes,'attempted':n,
            'wilson_95':wilson(successes,n),'capture_and_policy_ms_p50':float(np.median(latencies)),
            'capture_and_policy_ms_p95':float(np.percentile(latencies,95)),
            'limitations':'Two trained tasks/known vocabulary and objects. Direct image/text/proprio action path; no oracle phase/object inputs. One training seed; no real robot or open-world claim.'}
    write_json(args.output,report);print(report['successes'],report['attempted']);return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--task',type=int,choices=[0,1],default=0,help='Evaluation scoring: 0 stow, 1 open')
    p.add_argument('--instruction');p.add_argument('--episodes',type=int,default=20);p.add_argument('--seed',type=int,default=90000)
    p.add_argument('--ablation',choices=['none','drop-image','swap-text','drop-history'],default='none')
    p.add_argument('--profile',choices=['iid','camera','lighting','friction','position','combined'],default='iid')
    p.add_argument('--output',default='artifacts/direct_vla_eval.json');args=p.parse_args()
    if args.episodes<1:p.error('Positive episode count required')
    evaluate(args)
