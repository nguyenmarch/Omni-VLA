"""Collect real RGB-D/action trajectories for two tasks and randomized scenes."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np
from simulation.expert import ScriptedExpert
from simulation.randomization import DomainRandomizer
from simulation.vision import capture_head
from models.direct_vla import proprioception, TASK_PHRASES
from utils.config import make_environment
from utils.checkpoint import file_hash
from utils.vla_dataset import SCHEMA
from utils.reporting import write_json


def collect(args):
    path = Path(args.output)
    if path.exists(): raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    env = make_environment(image_size=args.image_size)
    randomizer = DomainRandomizer(env)
    rows, splits = [], {'train': [], 'val': [], 'test': []}
    try:
        with h5py.File(path, 'x') as file:
            file.attrs.update(schema=SCHEMA, image_size=args.image_size, chunk_size=args.chunk_size,
                              action_dt=env.dt, instructions=json.dumps(TASK_PHRASES),
                              scene_sha256=file_hash(Path(__file__).resolve().parents[1]/'simulation/assets/bottle_drawer_scene.xml'))
            # Predetermined per-task split; failed teacher attempts stay outside all learning splits.
            for i in range(args.episodes):
                task = i % 2
                part = 'val' if i % 10 in (0, 1) else 'test' if i % 10 in (2, 3) else 'train'
                seed = args.seed+i
                obs, _ = env.reset(seed=seed)
                profile = ('iid', 'camera', 'lighting', 'friction', 'position')[i//2 % 5] if args.randomize else 'iid'
                params = randomizer.apply(seed, profile)
                obs = env._get_observation()
                expert = ScriptedExpert(env)
                frames, labels, phases, robots, points, actions = [], [], [], [], [], []
                for step in range(env.max_episode_steps):
                    action = expert.action()
                    if task == 1:
                        action[:6] = 0.; action[6] = 1.
                    if step % args.chunk_size == 0:
                        frames.append(capture_head(env)); robots.append(proprioception(obs))
                        phases.append(expert.phase); points.append(obs['object'][:6].copy())
                    actions.append(action.copy())
                    obs, _, done, trunc, info = env.step(action)
                    if task == 1:
                        # Teacher-only phase transitions for single-arm opening.
                        expert.age += 1
                        _, target, _, _ = expert.goals()
                        ready = (expert.age >= 20 if expert.phase == 2 else
                                 info['drawer_position'] > .16 if expert.phase == 3 else
                                 np.linalg.norm(obs['ee'][3:]-target) < .012)
                        if ready and expert.phase < 3: expert.phase += 1; expert.age = 0
                        expert.failed = expert.age > 350
                    else: expert.update()
                    success = info['drawer_position'] > .16 if task == 1 else info['success']
                    if success or done or trunc or expert.failed: break
                row = {'seed': seed, 'task': task, 'success': bool(success), 'profile': profile, 'steps': len(actions)}
                rows.append(row); print(json.dumps(row), flush=True)
                if not success: continue
                name = f'{i:06d}'
                group = file.create_group('episodes/'+name)
                group.attrs.update(seed=seed, task=task, randomization=json.dumps(params))
                chunks, masks = [], []
                for j in range(len(frames)):
                    chunk = np.zeros((args.chunk_size, 14), np.float32)
                    selected = actions[j*args.chunk_size:(j+1)*args.chunk_size]
                    chunk[:len(selected)] = selected
                    mask = np.zeros((args.chunk_size, 1), np.float32); mask[:len(selected)] = 1
                    chunks.append(chunk); masks.append(mask)
                record = {key: np.stack([frame[key] for frame in frames]) for key in ('rgb','depth','K','T')}
                record.update(robot=np.stack(robots), action=np.stack(chunks), action_mask=np.stack(masks),
                              phase=np.array(phases), landmarks=np.stack(points))
                for key, value in record.items(): group.create_dataset(key, data=value, compression='gzip')
                splits[part].append(name); file.flush()
            if any(not names for names in splits.values()): raise ValueError('Need successful episodes in every split')
            file.attrs['splits'] = json.dumps(splits)
        write_json(path.with_suffix('.report.json'), {'episodes': rows, 'splits': splits})
    finally: env.close()


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='data/direct_vla_v1.h5');p.add_argument('--episodes',type=int,default=40)
    p.add_argument('--seed',type=int,default=80000);p.add_argument('--image-size',type=int,default=64)
    p.add_argument('--chunk-size',type=int,default=5);p.add_argument('--randomize',action='store_true')
    args=p.parse_args()
    if args.episodes<10 or args.chunk_size<1 or args.image_size<32:p.error('Need >=10 episodes, positive chunk and image size >=32')
    collect(args)
