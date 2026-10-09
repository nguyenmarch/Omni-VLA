"""Collect physics-driven privileged expert episodes into versioned HDF5."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import h5py
import numpy as np
from simulation.bimanual_env import BimanualBottleDrawerEnv, ROOT
from utils.config import make_environment
from simulation.expert import ScriptedExpert, state_condition, PHASES


def collect(args):
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(f'{output} already exists; use a new output path')
    env = make_environment(args.config, images=args.images, render_mode='human' if args.gui else None, image_size=args.image_size)
    summaries = []
    learner = learner_stats = None
    if args.dagger_checkpoint:
        import torch
        from utils.checkpoint import load_policy
        torch.set_num_threads(2)
        learner, ckpt = load_policy(args.dagger_checkpoint)
        learner_stats = {k:v.numpy() for k,v in ckpt['stats'].items()}
        if abs(ckpt['action_dt']-env.dt)>1e-9:
            env.close();raise ValueError('DAgger checkpoint action_dt mismatch')
    started = time.perf_counter()
    cycle = getattr(args, 'dagger_cycle', 8)
    learner_steps = getattr(args, 'dagger_learner_steps', 4)
    learner_phases = getattr(args, 'dagger_phases', None) or list(PHASES)
    try:
        with h5py.File(output, 'w') as h5:
            h5.attrs.update(schema_version='omni-state-v1', action_dt=env.dt, action_dim=14, condition_dim=67,
                            instruction='Open the drawer, stow the bottle, then close the drawer.',
                            phase_names=json.dumps(PHASES), expert='privileged_physics_fsm',
                            scene_sha256=hashlib.sha256((ROOT/'simulation/assets/bottle_drawer_scene.xml').read_bytes()).hexdigest(),
                            action_convention='normalized delta xyz .015m, world rotvec .08rad, gripper [-1,1] -> finger [0,.04]m')
            episodes = h5.create_group('episodes')
            if learner is not None:
                from utils.checkpoint import file_hash
                h5.attrs['dagger_checkpoint_sha256'] = file_hash(args.dagger_checkpoint)
                h5.attrs['dagger_schedule'] = json.dumps({'cycle': cycle, 'learner_steps': learner_steps,
                                                        'phases': learner_phases})
            for episode in range(args.num_episodes):
                seed = args.seed + episode
                obs, _ = env.reset(seed=seed)
                expert = ScriptedExpert(env)
                group = episodes.create_group(f'{episode:06d}')
                group.attrs['seed'] = seed
                steps = 0
                info = env.info()
                while True:
                    tick_started = time.perf_counter()
                    action = expert.action()
                    record = {'condition': state_condition(obs, expert.phase), 'action': action,
                              'phase': np.array(expert.phase, np.int32), **{f'obs/{k}': v for k,v in obs.items()}}
                    for name, value in record.items():
                        value = np.asarray(value)
                        if name not in group:
                            group.create_dataset(name, shape=(0,)+value.shape, maxshape=(None,)+value.shape,
                                                 chunks=(1,)+value.shape, dtype=value.dtype, compression='gzip')
                        ds = group[name]
                        ds.resize(steps+1, axis=0)
                        ds[steps] = value
                    executed = action.copy()
                    use_learner = learner is not None and PHASES[expert.phase] in learner_phases and (steps % cycle) < learner_steps
                    if use_learner:
                        condition = (record['condition']-learner_stats['condition_mean'])/learner_stats['condition_std']
                        chunk = learner.sample_euler(torch.from_numpy(condition[None])).numpy()[0]
                        executed = np.clip(chunk[0]*learner_stats['action_std']+learner_stats['action_mean'],-1,1)
                    if 'executed_action' not in group:
                        group.create_dataset('executed_action',shape=(0,14),maxshape=(None,14),dtype='f4',chunks=(1,14))
                    group['executed_action'].resize(steps+1,axis=0)
                    group['executed_action'][steps] = executed
                    if 'learner_executed' not in group:
                        group.create_dataset('learner_executed',shape=(0,),maxshape=(None,),dtype='?',chunks=(1,))
                    group['learner_executed'].resize(steps+1,axis=0)
                    group['learner_executed'][steps] = use_learner
                    obs, _, done, trunc, info = env.step(executed)
                    steps += 1
                    if args.gui:
                        time.sleep(max(0.,env.dt-(time.perf_counter()-tick_started)))
                    expert.update()
                    if done or trunc or expert.failed:
                        break
                reason = 'success' if info['success'] else expert.failure_reason or 'episode_limit'
                group.attrs.update(success=info['success'], failure_reason=reason, steps=steps, labels_valid=bool(info['success'] or learner is not None),
                                   supervision='dagger_expert_labels' if learner is not None else 'expert_demonstration')
                if not info['success'] and not args.keep_failures and learner is None:
                    del episodes[f'{episode:06d}']
                summary = {'episode': episode, 'seed': seed, 'steps': steps, 'phase': PHASES[expert.phase], **info, 'reason': reason}
                summaries.append(summary)
                print(json.dumps(summary), flush=True)
                h5.flush()
            names = list(episodes)
            rng = np.random.default_rng(42)
            rng.shuffle(names)
            n = len(names)
            n_val = max(1, n//5) if n>=3 else 0
            n_test = max(1, n//5) if n>=3 else 0
            split = {'train': names[n_val+n_test:], 'val': names[:n_val], 'test': names[n_val:n_val+n_test]}
            h5.attrs['splits'] = json.dumps(split)
            h5.attrs['summaries'] = json.dumps(summaries)
    finally:
        env.close()
    report = {'attempted': len(summaries), 'successes': sum(s['success'] for s in summaries),
              'wall_seconds': time.perf_counter()-started, 'episodes': summaries}
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2))
    print('Saved', output, 'successes:', report['successes'], '/', report['attempted'])
    if report['successes'] == 0 and learner is None:
        raise RuntimeError('No successful demonstrations; inspect collection report before training')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',help='Environment YAML')
    parser.add_argument('--num-episodes', '--num_episodes', type=int, default=20)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', default='data/drawer_pilot.h5')
    parser.add_argument('--images', action='store_true',default=None)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--keep-failures', action='store_true')
    parser.add_argument('--dagger-checkpoint', help='Roll out learner 4/8 steps while recording expert corrective labels')
    parser.add_argument('--dagger-cycle', type=int, default=8)
    parser.add_argument('--dagger-learner-steps', type=int, default=4)
    parser.add_argument('--dagger-phases', nargs='+', choices=PHASES, help='Execute learner only in these phases; expert labels recorded throughout')
    args = parser.parse_args()
    if args.num_episodes < 1:
        parser.error('num-episodes must be positive')
    if not 0 < args.dagger_learner_steps < args.dagger_cycle:
        parser.error('Require 0 < dagger-learner-steps < dagger-cycle')
    collect(args)

if __name__ == '__main__':
    main()
