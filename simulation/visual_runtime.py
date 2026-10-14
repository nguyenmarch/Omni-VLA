"""Bounded asynchronous perception; physics and rendering never share MjData."""
import copy
from dataclasses import dataclass
from threading import Condition, Thread
import time

import mujoco
import numpy as np

from simulation.vision import capture_head


@dataclass
class Request:
    episode: int
    step: int
    sim_time: float
    captured_at: float
    state: np.ndarray


@dataclass
class Result:
    request: Request
    value: object = None
    error: str = None
    elapsed_ms: float = 0.


def usable(result, episode, sim_time, max_sim_age, max_wall_age, now=None):
    if result is None or result.request.episode != episode:
        return False
    sim_age = sim_time - result.request.sim_time
    wall_age = (time.perf_counter() if now is None else now) - result.request.captured_at
    return 0 <= sim_age <= max_sim_age and 0 <= wall_age <= max_wall_age


class LatestWorker:
    """One running request, one replaceable pending request, one latest result.

    A generation reset drops pending/results; an already running old generation
    may finish but cannot publish into the new episode. Close runs on its owner
    thread, including the renderer's OpenGL context.
    """
    def __init__(self, process, cleanup=lambda: None):
        self.process, self.cleanup = process, cleanup
        self.condition = Condition()
        self.pending = self.result = None
        self.episode = None
        self.stopped = False
        self.replaced = self.discarded = 0
        self.thread = Thread(target=self._run, name='visual-perception', daemon=True)
        self.thread.start()

    def reset(self, episode):
        with self.condition:
            self.episode = episode
            self.pending = self.result = None

    def submit(self, request):
        with self.condition:
            if self.stopped or request.episode != self.episode:
                raise ValueError('Worker stopped or wrong episode')
            self.replaced += self.pending is not None
            self.pending = request
            self.condition.notify()

    def poll(self):
        with self.condition:
            result, self.result = self.result, None
            return result

    def _run(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.stopped or self.pending is not None)
                    if self.stopped:
                        return
                    request, self.pending = self.pending, None
                started = time.perf_counter()
                try:
                    result = Result(request, value=self.process(request))
                except Exception as error:
                    result = Result(request, error=f'{type(error).__name__}: {error}')
                result.elapsed_ms = (time.perf_counter()-started)*1000
                with self.condition:
                    if request.episode == self.episode and not self.stopped:
                        self.result = result
                    else:
                        self.discarded += 1
        finally:
            self.cleanup()

    def close(self):
        with self.condition:
            self.stopped = True
            self.pending = None
            self.condition.notify()
        self.thread.join()


class SnapshotPerception:
    def __init__(self, env, grounder, ablation, arm_mask=True, depth_max_m=3.0):
        self.model = copy.copy(env.model)
        self.data = mujoco.MjData(self.model)
        self.image_size = env.image_size
        self.renderer = None
        self.grounder, self.ablation = grounder, ablation
        self.arm_mask = arm_mask
        self.depth_max_m = depth_max_m

    def _renderer(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, self.image_size, self.image_size)
        return self.renderer

    def __call__(self, request):
        mujoco.mj_setState(self.model, self.data, request.state, mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(self.model, self.data)
        frame = capture_head(self, arm_mask=self.arm_mask, depth_max_m=self.depth_max_m)
        points, metadata = self.grounder.predict(frame, self.ablation)
        return points, metadata, frame

    def reset_episode(self):
        self.grounder.reset_episode()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def snapshot(env, episode, step):
    state = np.empty(mujoco.mj_stateSize(env.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(env.model, env.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    return Request(episode, step, env.data.time, time.perf_counter(), state)


def hold_action(robot):
    action = np.zeros(14, np.float32)
    action[[6, 13]] = np.clip(np.asarray(robot['gripper']).reshape(2, 2).mean(1)/.02-1, -1, 1)
    return action
