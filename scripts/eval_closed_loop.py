"""Evaluate expert or state-conditioned learned policy on CPU, without teacher actions."""
import argparse
import json
from pathlib import Path
import time
from collections import deque
import numpy as np
from simulation.bimanual_env import BimanualBottleDrawerEnv
from utils.config import make_environment
from simulation.expert import ScriptedExpert, state_condition, PHASES


def evaluate(args):
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    model = checkpoint = session = None
    if args.checkpoint:
        import torch
        from utils.checkpoint import load_policy
        torch.set_num_threads(args.threads)
        torch.manual_seed(args.seed)
        path = Path(args.checkpoint)
        if path.suffix == '.onnx':
            import onnxruntime as ort
            metadata = json.loads(path.with_suffix('.json').read_text())
            checkpoint = metadata
            options=ort.SessionOptions();options.intra_op_num_threads=args.threads;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=['CPUExecutionProvider'])
        else:
            model, checkpoint = load_policy(path)
        if checkpoint['model_config']['cond_dim'] != 67 or checkpoint['model_config']['action_dim'] != 14:
            raise ValueError('Checkpoint observation/timing incompatible with environment')
        stats={k: np.asarray(v, dtype=np.float32) for k,v in checkpoint['stats'].items()}
    elif not args.expert:
        raise ValueError('Use --expert or --checkpoint; no teacher fallback for learned evaluation')
    env=make_environment(args.config,render_mode='human' if args.gui else None)
    if checkpoint is not None and abs(checkpoint['action_dt']-env.dt)>1e-9:
        env.close();raise ValueError('Checkpoint action_dt does not match environment')
    episodes=[];latencies=[];begun=time.perf_counter();sim_seconds=0
    seeds = getattr(args, 'seeds', None) or list(range(args.seed, args.seed+args.episodes))
    speed = getattr(args, 'speed', 1.)
    interrupted = False
    trace = None
    if getattr(args, 'trace', None):
        trace_path = Path(args.trace);trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace = trace_path.open('x')
    try:
        for episode, seed in enumerate(seeds):
            obs,_=env.reset(seed=seed);manager=ScriptedExpert(env);buffer=deque();info=env.info()
            displayed_phase = None
            initial_phase=0
            from simulation.recovery import RecoveryExecutor,RecoveryState
            recovery=RecoveryExecutor(env,detector=getattr(args,'recovery_detector','oracle'))
            injection_done=False
            injection_ticks=0
            while True:
                tick_started = time.perf_counter()
                if args.gui:
                    if env.viewer is not None and not env.viewer.is_running():
                        interrupted = True;break
                    if manager.phase != displayed_phase:
                        print(f'Viewer: seed={seed}, phase={PHASES[manager.phase]}, speed={speed:g}x',flush=True)
                        displayed_phase = manager.phase
                if args.inject_slip and manager.phase==3 and not injection_done and obs['ee'][2]>.67:
                    env.data.qvel[env.model.joint('bottle_free').dofadr[0]:env.model.joint('bottle_free').dofadr[0]+3]=[.7,0,-.7]
                    injection_done=True
                    injection_ticks=18
                recovery_state=recovery.observe(obs,manager.phase) if args.recovery else RecoveryState.RUNNING
                recovering=recovery.active
                if recovering:
                    buffer.clear()
                    action=recovery.action(obs)
                elif args.expert:
                    action=manager.action()
                else:
                    if manager.phase != initial_phase:
                        buffer.clear();initial_phase=manager.phase
                    if not buffer:
                        condition=(state_condition(obs,manager.phase)-stats['condition_mean'])/stats['condition_std']
                        started=time.perf_counter()
                        if model is not None:
                            tensor=torch.from_numpy(condition[None])
                            if checkpoint['policy_type']=='cfm':
                                mc=checkpoint['model_config']
                                noise=np.random.default_rng(seed+info['step']).normal(size=(1,mc['chunk_size'],14)).astype(np.float32)
                                chunk=model.sample_euler(tensor,args.euler_steps,noise=torch.from_numpy(noise)).numpy()[0]
                            else:
                                chunk=model.sample_euler(tensor,args.euler_steps).numpy()[0]
                        else:
                            mc=checkpoint['model_config']
                            if checkpoint['policy_type']=='bc':
                                chunk=session.run(None,{'condition':condition[None]})[0][0]
                            else:
                                x=np.random.default_rng(seed+info['step']).normal(size=(1,mc['chunk_size'],14)).astype(np.float32)
                                for step in range(args.euler_steps):
                                    velocity=session.run(None,{'x_t':x,'t':np.array([step/args.euler_steps],np.float32),'condition':condition[None]})[0]
                                    x += velocity/args.euler_steps
                                chunk=x[0]
                        chunk=chunk*stats['action_std']+stats['action_mean']
                        latencies.append((time.perf_counter()-started)*1000)
                        buffer.extend(np.clip(chunk[:args.execute_horizon],-1,1))
                    action=buffer.popleft()
                if injection_ticks > 0:
                    action=action.copy();action[6]=1.
                    injection_ticks-=1
                if trace is not None:
                    from simulation.diagnostics import action_diagnostics
                    measurement = action_diagnostics(env, manager, action)
                    measurement['seed'] = seed
                    trace.write(json.dumps(measurement)+'\n')
                obs,_,done,trunc,info=env.step(action)
                if not recovering:
                    manager.update()
                if args.gui:
                    time.sleep(max(0., env.dt/speed-(time.perf_counter()-tick_started)))
                if done or trunc or manager.failed or recovery.monitor.state==RecoveryState.FAILED:
                    break
            sim_seconds+=env.data.time
            if interrupted: break
            row={'seed':seed,'steps':info['step'],'phase':PHASES[manager.phase],
                 'failure_reason':None if info['success'] else ('recovery_failed' if recovery.monitor.state==RecoveryState.FAILED else manager.failure_reason or 'episode_limit'),
                 'recovery_attempts':recovery.monitor.retries,'injected_slip':injection_done,**info}
            row['force_detection_final']=recovery.last_detection
            episodes.append(row);print(json.dumps(row),flush=True)
    finally:
        env.close()
        if trace is not None: trace.close()
    wall=time.perf_counter()-begun
    successes=sum(e['success'] for e in episodes);n=len(episodes);p=successes/n if n else None
    interval = None
    if n:
        z=1.96;den=1+z*z/n;center=(p+z*z/(2*n))/den;half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
        interval = [max(0.,center-half),min(1.,center+half)]
    report={'mode':'expert' if args.expert else 'learned_state_oracle_context', 'checkpoint':args.checkpoint,
            'episodes':episodes,'successes':successes,'attempted':n,'success_rate':p,'wilson_95':interval,
            'interrupted':interrupted,'seeds_requested':seeds,'gui_speed':speed if args.gui else None,
            'cfm_noise_protocol':'numpy_default_rng(episode_seed + action_step)' if checkpoint and checkpoint['policy_type']=='cfm' else None,
            'wall_seconds':wall,'simulation_seconds':sim_seconds,'real_time_factor':sim_seconds/wall,
            'sampler_ms_p50':float(np.percentile(latencies,50)) if latencies else None,
            'sampler_ms_p95':float(np.percentile(latencies,95)) if latencies else None,
            'cpu_threads':args.threads,'execute_horizon':args.execute_horizon,'euler_steps':args.euler_steps}
    report['recovery_detector']=getattr(args,'recovery_detector','oracle') if args.recovery else None
    report['recovery_geometry']='oracle_scripted_regrasp' if args.recovery else None
    if args.checkpoint:
        from utils.checkpoint import file_hash
        report['checkpoint_sha256']=file_hash(args.checkpoint)
    from utils.reporting import write_json
    write_json(args.output, report)
    print(json.dumps({k:v for k,v in report.items() if k!='episodes'},indent=2))
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True);group.add_argument('--expert',action='store_true');group.add_argument('--checkpoint')
    p.add_argument('--config',help='Environment YAML')
    p.add_argument('--episodes',type=int,default=50);p.add_argument('--seed',type=int,default=1000)
    p.add_argument('--seeds',nargs='+',type=int,help='Explicit seed sequence, overrides episodes/seed')
    p.add_argument('--speed',type=float,default=1.,help='GUI playback speed: 1 real-time, 0.5 slower; headless runs unpaced')
    p.add_argument('--output',default='artifacts/eval.json');p.add_argument('--gui',action='store_true')
    p.add_argument('--trace',help='New JSONL file: pre-action distances, angles, learner/expert actions; expert labels never executed')
    p.add_argument('--recovery',action='store_true');p.add_argument('--inject-slip',action='store_true')
    p.add_argument('--recovery-detector',choices=['oracle','wrench'],default='oracle',
                   help='Wrench detection uses force/fingers; regrasp still uses oracle geometry')
    p.add_argument('--threads',type=int,default=2);p.add_argument('--euler-steps',type=int,default=6);p.add_argument('--execute-horizon',type=int,default=2)
    args=p.parse_args()
    if min(args.episodes,args.euler_steps,args.execute_horizon,args.threads)<1:
        p.error('episodes, Euler steps, horizon, threads must be positive')
    if not np.isfinite(args.speed) or args.speed<=0:
        p.error('speed must be finite and positive')
    evaluate(args)

if __name__=='__main__':
    main()
