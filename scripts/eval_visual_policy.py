"""Finite-task RGB-D/BC evaluation with bounded async perception and diagnostics."""
import argparse
from collections import deque
from itertools import count
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.grounding3d import load_grounding_runtime
from simulation.expert import PHASES
from simulation.randomization import DomainRandomizer
from simulation.recovery import WrenchSlipMonitor
from simulation.vision import capture_head, robot_observation, VisualTaskManager, FULL_INSTRUCTION, OPEN_INSTRUCTION
from simulation.visual_runtime import LatestWorker, SnapshotPerception, Result, snapshot, usable, hold_action
from utils.checkpoint import load_policy, file_hash
from utils.config import make_environment
from utils.reporting import write_json, wilson
from utils.visual_diagnostics import TrainingEnvelope, trace_row, summarize_trace


def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    trace_path = getattr(args, 'trace', None)
    if trace_path and Path(trace_path).exists():
        raise FileExistsError(trace_path)
    torch.set_num_threads(2)
    grounder = load_grounding_runtime(args.grounding)
    model, checkpoint = load_policy(args.policy)
    if checkpoint['policy_type'] != 'bc' or checkpoint['model_config']['chunk_size'] != 1:
        raise ValueError('Require single-action BC bridge')
    stats = {k: v.numpy() for k, v in checkpoint['stats'].items()}
    envelope = TrainingEnvelope(args.diagnostic_dataset) if getattr(args, 'diagnostic_dataset', None) else None
    env = make_environment(images=False, image_size=grounder.image_size, render_mode='human' if args.gui else None)
    if abs(env.dt-checkpoint['action_dt']) > 1e-9:
        env.close()
        raise ValueError('BC timing mismatch')
    randomizer = DomainRandomizer(env)
    asynchronous = getattr(args, 'async_perception', False)
    paced = args.gui or asynchronous or getattr(args, 'realtime', False)
    loop = getattr(args, 'loop', False)
    max_sim_age = getattr(args, 'max_sim_age', .4)
    max_wall_age = getattr(args, 'max_wall_age', .4)
    max_gap = getattr(args, 'max_perception_gap', 1.)
    max_rejections = getattr(args, 'max_rejections', 3)
    profile = getattr(args, 'profile', 'iid')
    episodes = deque(maxlen=100 if loop else None)
    perception_ms, policy_ms, work_ms, cycle_ms = (deque(maxlen=10000 if loop else None) for _ in range(4))
    perception_errors = deque(maxlen=5000 if loop else None)
    sim_ages, wall_ages = deque(maxlen=10000 if loop else None), deque(maxlen=10000 if loop else None)
    stale_results = rejections = held_steps = deadline_misses = total_ticks = 0
    attempted = successes = 0
    stop_reason = 'episode_limit'
    trace = monitor = worker = None
    begun = time.perf_counter()
    sim_seconds = 0.
    try:
        if trace_path:
            Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
            trace = Path(trace_path).open('x')
        if getattr(args, 'camera_debug', False):
            from utils.camera_diagnostics import CameraDiagnostics
            monitor = CameraDiagnostics(args.debug_output, args.debug_port if args.gui else None)
        seeds = getattr(args, 'seeds', None) or list(range(args.seed, args.seed+args.episodes))
        for episode in count() if loop else range(len(seeds)):
            seed = args.seed+episode if loop else seeds[episode]
            env.reset(seed=seed)
            parameters = randomizer.apply(seed, profile)
            obs = env._get_observation()
            manager = latest = None
            failure = None
            rejection_streak = 0
            last_valid_sim = env.data.time
            info = env.info()
            episode_errors, transitions, slip_events = [], [], []
            slip_monitor = WrenchSlipMonitor() if getattr(args, 'slip_monitor', False) else None
            slip_states = {}
            oracle_at_capture = {}
            if asynchronous:
                processor = SnapshotPerception(env, grounder, args.ablation)
                worker = LatestWorker(processor, processor.close)
                worker.reset(episode)
            for step in range(env.max_episode_steps):
                started = time.perf_counter()
                robot = robot_observation(obs)
                if slip_monitor is not None:
                    robot['wrist_rotation'] = env.data.site('left_wrist').xmat.reshape(3, 3).copy()
                    robot['slip_detection_enabled'] = manager is not None and manager.phase in (3, 4)
                    slip = slip_monitor.update(robot, manager is not None and manager.phase in (2, 3, 4, 5), env.dt)
                    slip_states[slip['state']] = slip_states.get(slip['state'], 0)+1
                    if slip['slip'] and (not slip_events or step-slip_events[-1]['step'] > 10):
                        slip_events.append({'step': step, **slip})
                    if slip['slip'] and getattr(args, 'stop_on_slip', False):
                        env.step(hold_action(robot))
                        info = env.info()
                        failure = 'observable_force_slip_stop'
                        break
                result = None
                if step % args.perception_every == 0:
                    request = snapshot(env, episode, step)
                    # Scoring-only side table. Neither worker nor policy receives it.
                    oracle_at_capture[step] = np.asarray(obs['object'])[:6].reshape(2, 3).copy()
                    if len(oracle_at_capture) > 100:
                        del oracle_at_capture[min(oracle_at_capture)]
                    if asynchronous:
                        worker.submit(request)
                    else:
                        t = time.perf_counter()
                        try:
                            frame = capture_head(env)
                            points, metadata = grounder.predict(frame, args.ablation)
                            result = Result(request, (points, metadata, frame))
                        except ValueError as error:
                            result = Result(request, error=str(error))
                        result.elapsed_ms = (time.perf_counter()-t)*1000
                if asynchronous:
                    result = worker.poll()
                if result is not None:
                    perception_ms.append(result.elapsed_ms)
                    if not usable(result, episode, env.data.time, max_sim_age, max_wall_age):
                        stale_results += 1
                    elif result.error:
                        rejections += 1
                        rejection_streak += 1
                        if rejection_streak >= max_rejections:
                            failure = 'perception_rejected: ' + result.error
                            break
                    else:
                        rejection_streak = 0
                        points, metadata, frame = result.value
                        if manager is None:
                            manager = VisualTaskManager(points, args.instruction,
                                                        transition_tolerance=getattr(args, 'transition_tolerance', .012))
                        else:
                            manager.observe(points)
                        latest = result
                        last_valid_sim = env.data.time
                        truth = oracle_at_capture.get(result.request.step)
                        if truth is not None:
                            errors = np.linalg.norm(points-truth, axis=1)
                            episode_errors.append(errors)
                            perception_errors.append({'seed': seed, 'step': result.request.step,
                                                      'phase': PHASES[manager.phase],
                                                      'bottle_error_m': float(errors[0]), 'handle_error_m': float(errors[1])})
                            if monitor is not None:
                                monitor.update(frame, truth, points, metadata, seed, step, PHASES[manager.phase])
                fresh = usable(latest, episode, env.data.time, max_sim_age, max_wall_age)
                sim_age = env.data.time-latest.request.sim_time if latest else None
                wall_age = time.perf_counter()-latest.request.captured_at if latest else None
                if sim_age is not None:
                    sim_ages.append(sim_age)
                    wall_ages.append(wall_age)
                if env.data.time-last_valid_sim > max_gap:
                    failure = 'perception_unavailable_timeout'
                    break
                if manager is None or not fresh:
                    action = hold_action(robot)
                    held_steps += 1
                else:
                    t = time.perf_counter()
                    condition = manager.condition(robot)
                    x = torch.from_numpy(((condition-stats['condition_mean'])/stats['condition_std'])[None])
                    action = model.sample_euler(x).numpy()[0, 0]*stats['action_std']+stats['action_mean']
                    action = manager.adapt_action(np.clip(action, -1, 1))
                    policy_ms.append((time.perf_counter()-t)*1000)
                    if trace is not None:
                        trace.write(json.dumps(trace_row(manager, robot, action, seed, step, sim_age, wall_age, envelope))+'\n')
                obs, _, done, trunc, info = env.step(action)
                if manager is not None and fresh:
                    previous = manager.phase
                    manager.update(robot_observation(obs))
                    if previous != manager.phase:
                        transitions.append({'step': step+1, 'from': PHASES[previous], 'to': PHASES[manager.phase],
                                            'observation_sim_age_s': sim_age})
                elapsed = time.perf_counter()-started
                work_ms.append(elapsed*1000)
                total_ticks += 1
                deadline_misses += elapsed > env.dt
                if paced:
                    time.sleep(max(0., env.dt-elapsed))
                cycle_ms.append((time.perf_counter()-started)*1000)
                if args.gui and env.viewer is not None and not env.viewer.is_running():
                    failure = 'viewer_closed'
                    break
                if manager is not None and manager.failed:
                    failure = manager.failure_reason
                    break
                if done or trunc or (manager is not None and manager.completed):
                    break
            if worker is not None:
                worker.close()
                worker = None
            sim_seconds += env.data.time
            if failure == 'viewer_closed':
                stop_reason = 'viewer_closed'
                break
            success = bool(info['drawer_position'] > .16) if args.instruction == OPEN_INSTRUCTION else bool(info['success'])
            row = {'seed': seed, 'success': success, 'steps': info['step'],
                   'phase': PHASES[manager.phase] if manager else None,
                   'failure_reason': None if success else failure or 'episode_limit_or_visual_completion',
                   'visual_completed': manager.completed if manager else False,
                   'phase_transitions': transitions, 'randomization': parameters,
                   'wrench_slip_events': slip_events, 'wrench_slip_states': slip_states,
                   'landmark_mean_error_m': np.mean(episode_errors, axis=0).tolist() if episode_errors else None}
            episodes.append(row)
            attempted += 1
            successes += int(success)
            print(json.dumps(row), flush=True)
    except KeyboardInterrupt:
        stop_reason = 'keyboard_interrupt'
    finally:
        if worker is not None:
            worker.close()
        env.close()
        if monitor is not None:
            monitor.close()
        if trace is not None:
            trace.close()
    wall_seconds = time.perf_counter()-begun
    result = {'mode': 'modular_finite_language_vision_bc_bridge', 'instruction': args.instruction,
              'ablation': args.ablation, 'grounding_sha256': file_hash(args.grounding),
              'policy_sha256': file_hash(args.policy), 'episodes': list(episodes),
              'successes': successes, 'attempted': attempted,
              'success_rate': successes/attempted if attempted else None, 'wilson_95': wilson(successes, attempted),
              'loop': loop, 'stop_reason': stop_reason, 'async_perception': asynchronous, 'paced': paced,
              'profile': profile, 'perception_every_actions': args.perception_every,
              'seeds_requested': getattr(args, 'seeds', None) or list(range(args.seed, args.seed+args.episodes)),
              'suite_plan_sha256': getattr(args, 'suite_plan_sha256', None),
              'max_sim_age_s': max_sim_age, 'max_wall_age_s': max_wall_age,
              'max_perception_gap_s': max_gap, 'max_rejections': max_rejections,
              'slip_monitor': getattr(args, 'slip_monitor', False),
              'stop_on_slip': getattr(args, 'stop_on_slip', False),
              'transition_tolerance_m': getattr(args, 'transition_tolerance', .012),
              'held_steps': held_steps, 'stale_results': stale_results, 'perception_rejections': rejections,
              'deadline_misses': deadline_misses, 'control_ticks': total_ticks,
              'deadline_miss_fraction': deadline_misses/total_ticks if total_ticks else None,
              'wall_seconds': wall_seconds, 'simulation_seconds': sim_seconds,
              'real_time_factor': sim_seconds/wall_seconds,
              'perception_diagnostics': list(perception_errors),
              'diagnostic_dataset_sha256': file_hash(args.diagnostic_dataset) if envelope else None,
              'trace_summary': summarize_trace(trace_path) if trace_path else None,
              'retention': 'Bounded rolling samples/episodes in GUI loop; aggregate counts cover completed episodes.' if loop else 'All episodes and timing samples',
              'limitations': 'Scene-specific RGB-D student + frozen state BC, finite task grammar and known fixture geometry. Physics remains synchronous; async moves capture/inference to an independent worker. Timing includes Python/MuJoCo work, excludes ROS2; no hard real-time guarantee.'}
    for name, values in [('perception_capture_and_infer_ms', perception_ms), ('policy_ms', policy_ms),
                         ('control_work_ms', work_ms), ('control_cycle_ms', cycle_ms),
                         ('observation_sim_age_s', sim_ages), ('observation_wall_age_s', wall_ages)]:
        for percentile in (50, 95):
            result[f'{name}_p{percentile}'] = float(np.percentile(values, percentile)) if values else None
    write_json(output, result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grounding', required=True)
    p.add_argument('--policy', default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--instruction', choices=[FULL_INSTRUCTION, OPEN_INSTRUCTION], default=FULL_INSTRUCTION)
    p.add_argument('--ablation', choices=['none', 'drop-image', 'swap-text'], default='none')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--perception-every', type=int, default=5)
    p.add_argument('--async-perception', action='store_true', help='Worker owns snapshot renderer/inference; pace at action_dt')
    p.add_argument('--realtime', action='store_true', help='Pace synchronous headless evaluation for timing comparisons')
    p.add_argument('--max-sim-age', type=float, default=.4)
    p.add_argument('--max-wall-age', type=float, default=.4)
    p.add_argument('--max-perception-gap', type=float, default=1.)
    p.add_argument('--max-rejections', type=int, default=3)
    p.add_argument('--transition-tolerance', type=float, default=.012)
    p.add_argument('--slip-monitor', action='store_true', help='Observable wrist-load/finger detector; no oracle geometry')
    p.add_argument('--stop-on-slip', action='store_true', help='Hold and terminate on detected load loss; requires --slip-monitor')
    p.add_argument('--profile', choices=['iid', 'camera', 'lighting', 'friction', 'position', 'combined'], default='iid')
    p.add_argument('--trace', help='Exclusive JSONL action/phase/age/distribution diagnostics')
    p.add_argument('--diagnostic-dataset', help='Train-only state envelope; scoring diagnostics, never policy input')
    p.add_argument('--gui', action='store_true')
    p.add_argument('--output', default='artifacts/visual_policy_eval.json')
    p.add_argument('--loop', action='store_true')
    p.add_argument('--camera-debug', action='store_true')
    p.add_argument('--debug-port', type=int, default=8765)
    p.add_argument('--debug-output', default='artifacts/camera_debug')
    return p


def main():
    p = parser()
    args = p.parse_args()
    if min(args.episodes, args.perception_every, args.max_rejections) < 1:
        p.error('episodes, perception-every and max-rejections must be positive')
    if any(not np.isfinite(x) or x <= 0 for x in (args.max_sim_age, args.max_wall_age, args.max_perception_gap, args.transition_tolerance)):
        p.error('age/gap/tolerance must be finite and positive')
    if args.seeds and len(args.seeds) != len(set(args.seeds)):
        p.error('Duplicate evaluation seeds')
    if args.loop and not args.gui:
        p.error('--loop requires --gui')
    if args.stop_on_slip and not args.slip_monitor:
        p.error('--stop-on-slip requires --slip-monitor')
    if not 0 <= args.debug_port <= 65535:
        p.error('Invalid debug port')
    evaluate(args)


if __name__ == '__main__':
    main()
