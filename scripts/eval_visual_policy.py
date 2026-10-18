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
from simulation.vision import (capture_head, robot_observation, VisualTaskManager, FULL_INSTRUCTION,
                               OPEN_INSTRUCTION, PLACE_CONTACT_MODES, validate_place_contact_mode)
from simulation.visual_runtime import LatestWorker, SnapshotPerception, Result, snapshot, usable, hold_action
from utils.checkpoint import load_policy, file_hash
from utils.config import make_environment
from utils.reporting import write_json, wilson
from utils.visual_diagnostics import TrainingEnvelope, trace_row, summarize_trace
from utils.visual_protocol import effective_options, object_hash
from utils.visual_scoring import scoring_snapshot, EpisodeAudit
from simulation.transfer import TRANSFER_DEFAULTS
from simulation.placement_contact import PlacementContactObserver, PLACEMENT_DEFAULTS
from simulation.verified_placement import RELEASE_DEFAULTS


def evaluate(args):
    options = effective_options(args)
    startup_hold_steps=getattr(args,'startup_hold_steps',0)
    observation_age_clock=getattr(args,'observation_age_clock','both')
    if observation_age_clock not in ('both','simulation'):
        raise ValueError('Unknown observation age clock')
    if type(startup_hold_steps) is not int or startup_hold_steps<0:
        raise ValueError('startup_hold_steps must be a nonnegative integer')
    place_contact_mode = getattr(args, 'place_contact_mode', 'off')
    validate_place_contact_mode(place_contact_mode)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    trace_path = getattr(args, 'trace', None)
    if trace_path and Path(trace_path).exists():
        raise FileExistsError(trace_path)
    torch.set_num_threads(2)
    grounder = load_grounding_runtime(args.grounding)
    legacy_perception = getattr(args, 'legacy_perception', False)
    arm_mask_enabled = not legacy_perception and getattr(args, 'enable_arm_mask', False)
    temporal_smoothing_enabled = not legacy_perception and getattr(args, 'enable_temporal_smoothing', False)
    depth_max_m = 5.0 if legacy_perception else 3.0
    if hasattr(grounder, 'temporal_fallback_enabled'):
        grounder.temporal_fallback_enabled = not legacy_perception
        grounder.temporal_smoothing_enabled = temporal_smoothing_enabled
        grounder.max_fallback_age_s = args.fallback_max_age_s
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
    wall_guard_budget=max_wall_age if observation_age_clock=='both' else float('inf')
    wall_stale_results=0
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
            # The prior perception worker is joined at episode end.
            if hasattr(grounder, 'reset_episode'):
                grounder.reset_episode()
            parameters = randomizer.apply(seed, profile)
            obs = env._get_observation()
            manager = latest = None
            failure = None
            rejection_streak = 0
            last_valid_sim = env.data.time
            info = env.info()
            episode_errors, transitions, slip_events = [], [], []
            episode_fallbacks = 0
            contact_triggers = 0
            audit = EpisodeAudit() if getattr(args, 'scoring_diagnostics', False) else None
            previous_ee = None
            slip_monitor = WrenchSlipMonitor() if getattr(args, 'slip_monitor', False) else None
            slip_states = {}
            placement_observer = (PlacementContactObserver({key: options['controller_options'][key]
                                   for key in PLACEMENT_DEFAULTS})
                                  if options['controller_options']['placement_observer'] else None)
            placement_states = {}
            oracle_at_capture = {}
            if asynchronous:
                processor = SnapshotPerception(env, grounder, args.ablation,
                                               arm_mask=arm_mask_enabled, depth_max_m=depth_max_m)
                worker = LatestWorker(processor, processor.close)
                worker.reset(episode)
            for step in range(env.max_episode_steps):
                started = time.perf_counter()
                robot = robot_observation(obs)
                robot['wrist_rotations'] = np.stack([env.data.site(side+'_wrist').xmat.reshape(3, 3)
                                                     for side in ('left', 'right')])
                robot['ee_rotations'] = [env.data.site(side+'_ee').xmat.reshape(3, 3).tolist()
                                         for side in ('left', 'right')]
                ee = np.asarray(robot['ee']).reshape(2, 3)
                ee_velocity = np.zeros((2, 3)) if previous_ee is None else (ee-previous_ee)/env.dt
                robot['ee_velocity'] = ee_velocity
                previous_ee = ee.copy()
                stop_for_slip = False
                if slip_monitor is not None:
                    robot['wrist_rotation'] = env.data.site('left_wrist').xmat.reshape(3, 3).copy()
                    robot['slip_detection_enabled'] = manager is not None and manager.phase in (3, 4)
                    slip = slip_monitor.update(robot, manager is not None and manager.phase in (2, 3, 4, 5), env.dt)
                    slip_states[slip['state']] = slip_states.get(slip['state'], 0)+1
                    if slip['slip'] and (not slip_events or step-slip_events[-1]['step'] > 10):
                        slip_events.append({'step': step, **slip})
                    if slip['slip'] and getattr(args, 'stop_on_slip', False):
                        stop_for_slip = True
                result = None
                if not stop_for_slip and step % args.perception_every == 0:
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
                            frame = capture_head(env, arm_mask=arm_mask_enabled, depth_max_m=depth_max_m)
                            points, metadata = grounder.predict(frame, args.ablation)
                            result = Result(request, (points, metadata, frame))
                        except ValueError as error:
                            result = Result(request, error=str(error))
                        result.elapsed_ms = (time.perf_counter()-t)*1000
                if asynchronous:
                    result = worker.poll()
                # Explicit common action prefix for paired replay. Initial
                # renderer compilation can otherwise move a result across the
                # wall-age boundary and change physics before any ablation.
                if episode==0 and step<startup_hold_steps:
                    result=None
                if result is not None:
                    perception_ms.append(result.elapsed_ms)
                    wall_stale_results+=int(time.perf_counter()-result.request.captured_at>max_wall_age)
                    if not usable(result, episode, env.data.time, max_sim_age, wall_guard_budget):
                        stale_results += 1
                    elif result.error:
                        rejections += 1
                        rejection_streak += 1
                        if rejection_streak >= max_rejections:
                            failure = 'perception_rejected: ' + result.error
                    else:
                        rejection_streak = 0
                        points, metadata, frame = result.value
                        episode_fallbacks += int(bool(metadata.get('fallback_used', False)))
                        if manager is None:
                            manager = VisualTaskManager(
                                points, args.instruction,
                                transition_tolerance=getattr(args, 'transition_tolerance', .012),
                                transfer_tolerance=getattr(args, 'transfer_tolerance', .012),
                                place_contact_force_threshold_n=getattr(args, 'place_contact_force_threshold_n', 8.),
                                place_contact_debounce_steps=getattr(args, 'place_contact_debounce_steps', 3),
                                place_contact_mode=place_contact_mode,
                                transfer_options={key: options['controller_options'][key] for key in TRANSFER_DEFAULTS},
                                release_options={key: options['controller_options'][key] for key in RELEASE_DEFAULTS})
                        else:
                            manager.observe(points)
                        manager.placement_observer = placement_observer
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
                fresh = usable(latest, episode, env.data.time, max_sim_age, wall_guard_budget)
                sim_age = env.data.time-latest.request.sim_time if latest else None
                wall_age = time.perf_counter()-latest.request.captured_at if latest else None
                if sim_age is not None:
                    sim_ages.append(sim_age)
                    wall_ages.append(wall_age)
                if env.data.time-last_valid_sim > max_gap:
                    failure = 'perception_unavailable_timeout'
                metadata = latest.value[1] if latest is not None else {}
                source_age = sim_age+metadata.get('fallback_age_s', 0.) if sim_age is not None else None
                if failure:
                    if manager is not None and manager.placement_active:
                        manager.verified_placement.failure_reason='visual_placement_perception_timeout'
                        failure=manager.verified_placement.failure_reason+': '+failure
                    if trace is not None:
                        tick = trace_row(manager, robot, None, seed, step, sim_age, wall_age, envelope,
                                         ee_velocity=ee_velocity, metadata=metadata,
                                         placement=placement_observer.result if placement_observer else None)
                        tick.update(executed=False, stop_reason=failure, transition_reason=None)
                        trace.write(json.dumps(tick, allow_nan=False)+'\n')
                    break
                before_override = None
                held = manager is None or not fresh or stop_for_slip
                controller_held=bool(manager is not None and manager.placement_active and
                    (manager.verified_placement.state in ('collision_hold','release','verify_open') or
                     manager.phase==5 and manager.placement_evidence.get('support')))
                if held or (manager is not None and not manager.requires_policy):
                    action = hold_action(robot)
                    held_steps += int(held)
                    if manager is not None and (manager.placement_active or
                            manager.phase == 4 and manager.terminal_transfer is not None):
                        action = manager.adapt_action(action, robot, env.dt, fresh and not stop_for_slip, source_age)
                else:
                    t = time.perf_counter()
                    condition = manager.condition(robot)
                    x = torch.from_numpy(((condition-stats['condition_mean'])/stats['condition_std'])[None])
                    action = model.sample_euler(x).numpy()[0, 0]*stats['action_std']+stats['action_mean']
                    before_override = np.clip(action, -1, 1)
                    action = manager.adapt_action(before_override, robot, env.dt, fresh, source_age)
                    policy_ms.append((time.perf_counter()-t)*1000)
                phase = manager.phase if manager is not None else None
                tick = trace_row(manager, robot, action, seed, step, sim_age, wall_age, envelope,
                                 before_override=before_override, ee_velocity=ee_velocity,
                                 metadata=metadata, placement=placement_observer.result if placement_observer else None) if trace is not None else None
                # Oracle diagnostics are collected only after selecting the action.
                scoring_before = scoring_snapshot(env) if audit is not None else None
                sim_before, wall_before = float(env.data.time), time.perf_counter()-begun
                obs, _, done, trunc, info = env.step(action)
                # Sample force immediately after physics, before another camera
                # capture/inference. Calibrated H4 consumes it; off stays shadow.
                placement_after = None
                if placement_observer is not None:
                    sensor_robot = robot_observation(obs)
                    sensor_robot['wrist_rotations'] = np.stack([
                        env.data.site(side+'_wrist').xmat.reshape(3, 3).copy() for side in ('left', 'right')])
                    sensor_robot['ee_velocity'] = (np.asarray(obs['ee']).reshape(2, 3)-ee)/env.dt
                    sensor_robot['placement_stage'] = phase if phase is not None else 0
                    sensor_begun = time.perf_counter()
                    placement_after = placement_observer.update(sensor_robot, env.dt)
                    placement_after['update_ms'] = (time.perf_counter()-sensor_begun)*1000
                    placement_states[placement_after['calibration_status']] = placement_states.get(
                        placement_after['calibration_status'], 0)+1
                transition = None
                terminal_active = manager is not None and manager.phase == 4 and manager.terminal_transfer is not None
                placement_active = manager is not None and manager.placement_active
                if manager is not None and (fresh or terminal_active or placement_active) and not stop_for_slip:
                    previous = manager.phase
                    update_robot = robot_observation(obs)
                    update_robot['left_wrist_rotation'] = env.data.site('left_wrist').xmat.reshape(3, 3).copy()
                    if terminal_active or placement_active:
                        update_robot['ee_rotations'] = [env.data.site(side+'_ee').xmat.reshape(3, 3).copy()
                                                        for side in ('left', 'right')]
                        update_robot['ee_velocity'] = (np.asarray(obs['ee']).reshape(2, 3)-ee)/env.dt
                        manager.update(update_robot, env.dt, fresh,
                                       source_age+env.dt if source_age is not None else None)
                    else:
                        manager.update(update_robot)
                    if previous != manager.phase:
                        contact = manager.last_transition_reason in ('wrench_contact','calibrated_support')
                        contact_triggers += int(contact)
                        transition = {'step': step+1, 'from': PHASES[previous], 'to': PHASES[manager.phase],
                                            'observation_sim_age_s': sim_age,
                                            'reason': manager.last_transition_reason or 'legacy_phase_condition',
                                            'place_contact_fz_n': manager.place_contact_fz_n if previous == 5 else None}
                        transitions.append(transition)
                scoring_after = scoring_snapshot(env) if audit is not None else None
                if audit is not None:
                    audit.record(phase, scoring_before, scoring_after, action, robot['gripper'], obs['gripper'],
                                 step+1, sim_before, float(env.data.time), wall_before, time.perf_counter()-begun,
                                 transition=transition, held=held or controller_held)
                if tick is not None:
                    diagnostic_elapsed = time.perf_counter()-started
                    tick.update(executed=True, held=held, sim_time_before_s=sim_before,
                                controller_hold=controller_held,
                                sim_time_after_s=float(env.data.time), transition_reason=transition['reason'] if transition else None,
                                transition=transition, perception_usable=bool(fresh),
                                wall_time_before_s=wall_before, wall_time_after_s=time.perf_counter()-begun,
                                hold_reason=('slip_stop' if stop_for_slip else 'await_perception' if manager is None
                                             else 'stale_perception') if held else None,
                                ee_position_after_step_m=np.asarray(obs['ee']).reshape(2, 3).tolist(),
                                gripper_fingers_after_step_m=np.asarray(obs['gripper']).reshape(2,2).tolist(),
                                wrench_site_after_step=np.asarray(obs['wrench']).reshape(2, 6).tolist(),
                                placement_after_update=placement_after,
                                phase_after=manager.phase if manager is not None else None,
                                contact_streak_after_update=getattr(manager, 'place_contact_streak', 0),
                                transfer_after_update=manager.terminal_transfer.telemetry()
                                    if manager is not None and manager.terminal_transfer is not None else None,
                                release_after_update=manager.verified_placement.telemetry()
                                    if manager is not None and manager.verified_placement is not None else None,
                                scoring_only={'before': scoring_before, 'after': scoring_after} if audit else None,
                                work_ms_before_trace_write=diagnostic_elapsed*1000,
                                deadline_miss_before_trace_write=diagnostic_elapsed > env.dt)
                    trace.write(json.dumps(tick, allow_nan=False)+'\n')
                elapsed = time.perf_counter()-started
                if audit is not None:
                    audit.timing(elapsed, env.dt)
                work_ms.append(elapsed*1000)
                total_ticks += 1
                deadline_misses += elapsed > env.dt
                if paced:
                    time.sleep(max(0., env.dt-elapsed))
                cycle_ms.append((time.perf_counter()-started)*1000)
                if stop_for_slip:
                    failure = 'observable_force_slip_stop'
                    break
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
                   'fallback_predictions': episode_fallbacks,
                   'place_contact_triggers': contact_triggers,
                   'control_audit': audit.summary(env.dt) if audit is not None else None,
                   'wrench_slip_events': slip_events, 'wrench_slip_states': slip_states,
                   'placement_observer_states': placement_states,
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
              'suite_variant': getattr(args, 'suite_variant', None),
              'suite_source_sha256': getattr(args, 'suite_source_sha256', None),
              'startup_hold_steps': startup_hold_steps,
              'observation_age_clock':observation_age_clock,'wall_stale_results':wall_stale_results,
              'effective_options': options, 'options_sha256': object_hash(options),
              'scoring_diagnostics': getattr(args, 'scoring_diagnostics', False),
              'trace_sha256': file_hash(trace_path) if trace_path else None,
              'action_dt_s': env.dt,
              'max_sim_age_s': max_sim_age, 'max_wall_age_s': max_wall_age,
              'max_perception_gap_s': max_gap, 'max_rejections': max_rejections,
              'fallback_predictions': sum(row.get('fallback_predictions', 0) for row in episodes),
              'fallback_episodes': sum(row.get('fallback_predictions', 0) > 0 for row in episodes),
              'perception_invalid_reasons': dict(getattr(grounder, 'failure_reasons', {})),
              'slip_monitor': getattr(args, 'slip_monitor', False),
              'stop_on_slip': getattr(args, 'stop_on_slip', False),
              'legacy_perception': legacy_perception,
              'arm_mask_enabled': arm_mask_enabled,
              'temporal_smoothing_enabled': temporal_smoothing_enabled,
              'fallback_max_age_s': args.fallback_max_age_s,
              'transition_tolerance_m': getattr(args, 'transition_tolerance', .012),
              'transfer_tolerance_m': getattr(args, 'transfer_tolerance', .012),
              'transfer_mode': options['controller_options']['transfer_mode'],
              'placement_observer': options['controller_options']['placement_observer'],
              'place_contact_mode': place_contact_mode,
              'place_contact_force_threshold_n': getattr(args, 'place_contact_force_threshold_n', 8.),
              'place_contact_debounce_steps': getattr(args, 'place_contact_debounce_steps', 3),
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
    p.add_argument('--legacy-perception', action='store_true', help='Run P5 without temporal fallback or projected robot mask (paired benchmark reference)')
    p.add_argument('--enable-arm-mask', action='store_true', help='Development ablation: mask projected robot links where depth agrees with forward kinematics')
    p.add_argument('--enable-temporal-smoothing', action='store_true', help='Development ablation: EMA smooth valid landmarks; can lag moving targets')
    p.add_argument('--fallback-max-age-s', type=float, default=.4, help='Maximum simulated age for a cached landmark')
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--seed', type=int, default=9000)
    p.add_argument('--startup-hold-steps',type=int,default=0,
                   help='Hold the first N actions of the first episode only; explicit paired startup protocol')
    p.add_argument('--observation-age-clock',choices=['both','simulation'],default='both',
                   help='both: enforce sim+wall freshness; simulation: accelerated controller ablation, wall age diagnostic only')
    p.add_argument('--seeds', nargs='+', type=int)
    p.add_argument('--perception-every', type=int, default=5)
    p.add_argument('--async-perception', action='store_true', help='Worker owns snapshot renderer/inference; pace at action_dt')
    p.add_argument('--realtime', action='store_true', help='Pace synchronous headless evaluation for timing comparisons')
    p.add_argument('--max-sim-age', type=float, default=.4)
    p.add_argument('--max-wall-age', type=float, default=.4)
    p.add_argument('--max-perception-gap', type=float, default=1.)
    p.add_argument('--max-rejections', type=int, default=3)
    p.add_argument('--transition-tolerance', type=float, default=.012)
    p.add_argument('--transfer-tolerance', type=float, default=.012,
                   help='EE reach tolerance during transfer; baseline is 12 mm')
    for key, value in TRANSFER_DEFAULTS.items():
        if key == 'transfer_mode':
            p.add_argument('--transfer-mode', choices=['legacy', 'terminal'], default=value)
        else:
            p.add_argument('--'+key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--place-contact-mode', choices=PLACE_CONTACT_MODES, default='off',
                   help='off: legacy release; raw_debug: diagnostic raw force; calibrated: H4 verified release/retreat')
    p.add_argument('--placement-observer', action='store_true',
                   help='H3 observer shadow in off mode; enabled automatically by calibrated H4')
    for key, value in PLACEMENT_DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=type(value), default=value)
    for key, value in RELEASE_DEFAULTS.items():
        p.add_argument('--'+key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--place-contact-force-threshold-n', type=float, default=8.,
                   help='World-vertical wrist force threshold for early placement contact (newtons)')
    p.add_argument('--place-contact-debounce-steps', type=int, default=3,
                   help='Consecutive contact samples required to advance to gripper release')
    p.add_argument('--slip-monitor', action='store_true', help='Observable wrist-load/finger detector; no oracle geometry')
    p.add_argument('--stop-on-slip', action='store_true', help='Hold and terminate on detected load loss; requires --slip-monitor')
    p.add_argument('--profile', choices=['iid', 'camera', 'lighting', 'friction', 'position', 'combined'], default='iid')
    p.add_argument('--trace', help='Exclusive JSONL action/phase/age/distribution diagnostics')
    p.add_argument('--scoring-diagnostics', action='store_true',
                   help='Log oracle contacts/placement only for post-decision scoring; never policy inputs')
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
    try:
        validate_place_contact_mode(args.place_contact_mode)
    except NotImplementedError as error:
        p.error(str(error))
    if min(args.episodes, args.perception_every, args.max_rejections) < 1:
        p.error('episodes, perception-every and max-rejections must be positive')
    if any(not np.isfinite(x) or x <= 0 for x in (args.max_sim_age, args.max_wall_age, args.max_perception_gap,
                                                   args.transition_tolerance, args.transfer_tolerance,
                                                   args.place_contact_force_threshold_n)):
        p.error('age/gap/tolerance must be finite and positive')
    if args.place_contact_debounce_steps < 1:
        p.error('place-contact-debounce-steps must be positive')
    if not np.isfinite(args.fallback_max_age_s) or args.fallback_max_age_s <= 0:
        p.error('fallback-max-age-s must be finite and positive')
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
