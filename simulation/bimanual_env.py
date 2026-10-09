"""CPU-first two-Panda drawer manipulation; oracle state baseline by default."""
from pathlib import Path
import gymnasium as gym
import mujoco
import numpy as np
from utils.camera import calibration
from models.impedance_controller import CartesianImpedanceController, exp_rotation

ROOT = Path(__file__).resolve().parents[1]
SIDES = ('left', 'right')
ACTION_SCALE = np.tile([.015, .015, .015, .08, .08, .08, 1.], 2).astype(np.float32)


class BimanualBottleDrawerEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array', 'human'], 'render_fps': 25}

    def __init__(self, xml_path=None, control_freq=25., render_mode=None, image_size=128, max_episode_steps=1500, images=False):
        path = Path(xml_path) if xml_path else ROOT/'simulation/assets/bottle_drawer_scene.xml'
        if not path.is_absolute():
            path = ROOT/path
        self.model = mujoco.MjModel.from_xml_path(str(path))
        # Upstream collision-only Panda geometry uses group 3, hidden by default viewers.
        self.model.geom_group[self.model.geom_group == 3] = 0
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, image_size)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, image_size)
        self.data = mujoco.MjData(self.model)
        self.control_freq = control_freq
        self.dt = 1/control_freq
        ratio = self.dt/self.model.opt.timestep
        if not np.isclose(ratio, round(ratio)):
            raise ValueError('Action interval must be an integer multiple of physics timestep')
        self.substeps = round(ratio)
        self.max_episode_steps = max_episode_steps
        self.render_mode, self.image_size, self.images = render_mode, image_size, images
        self.renderer = self.viewer = None
        self.controllers = [CartesianImpedanceController(self.model, side) for side in SIDES]
        self.arm_qadr = np.concatenate([c.qadr for c in self.controllers])
        self.arm_dofs = np.concatenate([c.dofs for c in self.controllers])
        self.finger_qadr = np.array([self.model.joint(f'{side}_finger_joint{i}').qposadr[0] for side in SIDES for i in [1, 2]])
        self.grip_act = np.array([self.model.actuator(f'{side}_grip{i}').id for side in SIDES for i in [1, 2]])
        self.bottle_adr = self.model.joint('bottle_free').qposadr[0]
        self.drawer_adr = self.model.joint('drawer_slide').qposadr[0]
        self.action_space = gym.spaces.Box(-np.ones(14, np.float32), np.ones(14, np.float32))
        shapes = {'q': 14, 'dq': 14, 'gripper': 4, 'wrench': 12, 'ee': 6, 'object': 7, 'time': 1}
        spaces = {key: gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32) for key, dim in shapes.items()}
        if images:
            for cam in ['head', 'wrist']:
                spaces[f'rgb_{cam}'] = gym.spaces.Box(0, 255, (image_size, image_size, 3), np.uint8)
                spaces[f'depth_{cam}'] = gym.spaces.Box(0, np.inf, (image_size, image_size), np.float32)
                spaces[f'K_{cam}'] = gym.spaces.Box(-np.inf, np.inf, (3,3), np.float32)
                spaces[f'T_{cam}'] = gym.spaces.Box(-np.inf, np.inf, (4,4), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.reset()

    def site_pos(self, name):
        return self.data.site(name).xpos.copy()

    def _get_observation(self):
        wrench = []
        for side in SIDES:
            for kind in ['force', 'torque']:
                wrench.extend(self.data.sensor(f'{side}_{kind}').data.copy())
        obs = {'q': self.data.qpos[self.arm_qadr].copy(), 'dq': self.data.qvel[self.arm_dofs].copy(),
               'gripper': self.data.qpos[self.finger_qadr].copy(), 'wrench': np.array(wrench),
               'ee': np.concatenate([self.site_pos(f'{side}_ee') for side in SIDES]),
               'object': np.r_[self.site_pos('bottle_site'), self.site_pos('handle_site'), self.data.qpos[self.drawer_adr]],
               'time': np.array([self.data.time])}
        obs = {key: value.astype(np.float32) for key, value in obs.items()}
        if self.images:
            for label, camera in [('head', 'head_cam'), ('wrist', 'right_wrist_cam')]:
                self._renderer().disable_depth_rendering()
                self.renderer.update_scene(self.data, camera=camera)
                obs[f'rgb_{label}'] = self.renderer.render().copy()
                self.renderer.enable_depth_rendering()
                self.renderer.update_scene(self.data, camera=camera)
                obs[f'depth_{label}'] = self.renderer.render().astype(np.float32)
                obs[f'K_{label}'], obs[f'T_{label}'] = calibration(self.model,self.data,camera,self.image_size,self.image_size)
            self.renderer.disable_depth_rendering()
        return obs

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        for controller in self.controllers:
            self.data.qpos[controller.qadr] = controller.home
        self.data.qpos[self.finger_qadr] = .04
        self.data.qpos[self.bottle_adr:self.bottle_adr+2] += self.np_random.uniform(-.015, .015, 2)
        mujoco.mj_forward(self.model, self.data)
        self.targets = [(self.site_pos(f'{side}_ee'), self.data.site(f'{side}_ee').xmat.reshape(3, 3).copy()) for side in SIDES]
        self.step_counter = 0
        self.opened = self.placed = False
        self.initial_bottle = self.site_pos('bottle_site')
        self.max_drawer = 0.
        return self._get_observation(), self.info()

    def info(self):
        opening = float(self.data.qpos[self.drawer_adr])
        self.max_drawer = max(self.max_drawer, opening)
        self.opened |= opening > .16
        bottle = self.site_pos('bottle_site')
        center = self.data.body('drawer').xpos
        inside = abs(bottle[0]-center[0]) < .11 and abs(bottle[1]-center[1]) < .075 and .445 < bottle[2] < .57
        released = self.data.qpos[self.finger_qadr[:2]].mean() > .032
        self.placed |= self.opened and inside and released
        success = self.opened and self.placed and inside and released and opening < .035
        return {'success': bool(success), 'drawer_open': bool(self.opened), 'bottle_placed': bool(self.placed),
                'drawer_position': opening, 'max_drawer': self.max_drawer, 'step': self.step_counter}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError('Expected finite action [14]')
        action = np.clip(action, -1, 1)
        for i, side in enumerate(SIDES):
            a = action[7*i:7*i+7]
            position = self.site_pos(f'{side}_ee') + a[:3]*ACTION_SCALE[:3]
            position = np.clip(position, [-.1, -.75, .44], [.95, .75, 1.5])
            rotation = exp_rotation(a[3:6]*.08) @ self.data.site(f'{side}_ee').xmat.reshape(3, 3)
            self.targets[i] = position, rotation
            self.data.ctrl[self.grip_act[2*i:2*i+2]] = (a[6]+1)*.02
        for sub in range(self.substeps):
            if sub % 2 == 0:
                for controller, target in zip(self.controllers, self.targets):
                    self.data.ctrl[controller.actuators] = controller.compute_torque(self.model, self.data, target)
            mujoco.mj_step(self.model, self.data)
        self.step_counter += 1
        info = self.info()
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        info['physics_valid'] = bool(finite)
        obs = self._get_observation()
        if self.render_mode == 'human':
            self.render()
        return obs, float(info['success']), bool(info['success'] or not finite), self.step_counter >= self.max_episode_steps, info

    def _renderer(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=self.image_size, width=self.image_size)
        return self.renderer

    def render(self):
        if self.render_mode == 'human':
            import mujoco.viewer
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        self._renderer().update_scene(self.data, camera='head_cam')
        return self.renderer.render().copy()

    def close(self):
        if self.renderer:
            self.renderer.close()
        if self.viewer:
            self.viewer.close()
