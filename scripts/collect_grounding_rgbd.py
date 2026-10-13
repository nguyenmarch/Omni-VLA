"""Sparse RGB-D expert frames; object centers are training/evaluation labels only."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from utils.checkpoint import file_hash
from utils.config import make_environment
from simulation.bimanual_env import ROOT
from simulation.expert import ScriptedExpert,state_condition
from simulation.vision import capture_head,QUERIES,robot_observation


def collect(output,episodes=12,seed=8000,stride=5,size=128):
    p=Path(output)
    if p.exists():raise FileExistsError(p)
    p.parent.mkdir(parents=True,exist_ok=True)
    env=make_environment(images=False,image_size=size);summaries=[]
    try:
        with h5py.File(p,'w') as h:
            h.attrs.update(schema_version='omni-grounding-rgbd-v2',scene_sha256=file_hash(ROOT/'simulation/assets/bottle_drawer_scene.xml'),
                           frame_stride=stride,action_dt=env.dt,image_size=size,instructions=json.dumps(QUERIES),camera='head_cam')
            for i in range(episodes):
                obs,_=env.reset(seed=seed+i);expert=ScriptedExpert(env);g=h.create_group(f'episodes/{i:06d}');g.attrs['seed']=seed+i
                frames=0
                for step in range(env.max_episode_steps):
                    action=expert.action()
                    if step%stride==0:
                        record={**capture_head(env),'targets':obs['object'][:6].reshape(2,3),
                                'phase':np.array(expert.phase,dtype=np.int32),'condition':state_condition(obs,expert.phase),
                                'action':action,'step':np.array(step),**robot_observation(obs)}
                        for key,value in record.items():
                            value=np.asarray(value)
                            if key not in g:g.create_dataset(key,shape=(0,)+value.shape,maxshape=(None,)+value.shape,chunks=(1,)+value.shape,dtype=value.dtype,compression='gzip')
                            ds=g[key];ds.resize(frames+1,axis=0);ds[frames]=value
                        frames+=1
                    obs,_,done,trunc,info=env.step(action);expert.update()
                    if done or trunc or expert.failed:break
                g.attrs.update(success=info['success'],frames=frames)
                row={'seed':seed+i,'frames':frames,'success':info['success']};summaries.append(row);print(json.dumps(row),flush=True)
                h.flush()
            names=list(h['episodes']);np.random.default_rng(42).shuffle(names)
            n=max(1,episodes//5);splits={'train':names[2*n:],'val':names[:n],'test':names[n:2*n]}
            h.attrs['splits']=json.dumps(splits)
        p.with_suffix('.report.json').write_text(json.dumps({'episodes':summaries,'splits':splits},indent=2)+'\n')
    finally:env.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',default='data/grounding_rgbd_v2.h5')
    p.add_argument('--episodes',type=int,default=12);p.add_argument('--seed',type=int,default=8000);p.add_argument('--stride',type=int,default=5)
    p.add_argument('--image-size',type=int,default=128,help='Square RGB-D render size; calibration follows actual renderer size')
    a=p.parse_args()
    if a.episodes<3 or a.stride<1:p.error('Need >=3 episodes and positive stride')
    if a.image_size<32 or a.image_size%16:p.error('image-size must be >=32 and a multiple of 16')
    collect(a.output,a.episodes,a.seed,a.stride,a.image_size)
