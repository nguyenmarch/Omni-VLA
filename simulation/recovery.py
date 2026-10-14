"""Conservative slip monitor with debounce and bounded recovery FSM.

Detection is exposed separately until recovery demonstrations are validated.
"""
from collections import deque
from enum import Enum
import numpy as np

class RecoveryState(str,Enum):
    RUNNING='running'
    HOLD='hold'
    REGROUND='reground'
    REGRASP='regrasp'
    FAILED='failed'

class SlipMonitor:
    """Historical oracle-distance monitor, retained for frozen comparisons.

    Use WrenchSlipMonitor for observable force/gripper detection in visual runs.
    """
    def __init__(self,threshold=.10,debounce=3,max_retries=2):
        self.threshold,self.debounce,self.max_retries=threshold,debounce,max_retries
        self.reset()

    def reset(self):
        self.state=RecoveryState.RUNNING
        self.bad_frames=self.retries=0
        self.filtered_wrench=np.zeros(12)

    def update(self,ee_position,object_position,wrench,expected_grasp):
        self.filtered_wrench=.8*self.filtered_wrench+.2*np.asarray(wrench)
        lost=expected_grasp and np.linalg.norm(np.asarray(ee_position)-object_position)>self.threshold
        self.bad_frames=self.bad_frames+1 if lost else 0
        if self.state==RecoveryState.RUNNING and self.bad_frames>=self.debounce:
            self.retries+=1
            self.state=RecoveryState.HOLD if self.retries<=self.max_retries else RecoveryState.FAILED
        return self.state

    def advance(self,grounding_ready=False,grasp_confirmed=False):
        if self.state==RecoveryState.HOLD:
            self.state=RecoveryState.REGROUND
        elif self.state==RecoveryState.REGROUND and grounding_ready:
            self.state=RecoveryState.REGRASP
        elif self.state==RecoveryState.REGRASP and grasp_confirmed:
            self.state=RecoveryState.RUNNING
            self.bad_frames=0
        return self.state


class WrenchSlipMonitor:
    """World-frame wrist load loss corroborated by observable finger motion.

    A rolling median rejects contact impulses. Open-gripper force calibrates
    the unloaded reference; positive world-Z support during grasp calibrates
    payload weight for this scene's vertical gravity. Carrying can lose load;
    intentional placement unloading is disabled via slip_detection_enabled.
    No object position enters this detector. Calibration failure stays unknown.
    This detects opening/collapse with load loss, not every form of slip.
    """
    def __init__(self, minimum_load_n=.3, loss_fraction=.55, debounce=3,
                 steady_speed_m_s=.2, calibration_frames=5, filter_frames=15):
        if (not np.isfinite([minimum_load_n, loss_fraction, steady_speed_m_s]).all()
                or minimum_load_n <= 0 or not 0 < loss_fraction < 1
                or steady_speed_m_s <= 0 or min(debounce, calibration_frames) < 1
                or filter_frames < calibration_frames):
            raise ValueError('Invalid load, debounce or calibration parameters')
        self.minimum_load_n = minimum_load_n
        self.loss_fraction = loss_fraction
        self.debounce = debounce
        self.steady_speed_m_s = steady_speed_m_s
        self.calibration_frames = calibration_frames
        self.filter_frames = filter_frames
        self.reset()

    def reset(self):
        self.forces = deque(maxlen=self.filter_frames)
        self.filtered = self.unloaded = self.reference = self.previous_ee = None
        self.reference_width = None
        self.steady_frames = self.bad_frames = 0
        self.expected = False

    def update(self, robot, expected_grasp, dt):
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError('Positive finite dt required')
        wrench = np.asarray(robot['wrench'], dtype=float)
        ee = np.asarray(robot['ee'], dtype=float)
        gripper = np.asarray(robot['gripper'], dtype=float)
        if wrench.shape != (12,) or ee.shape != (6,) or gripper.shape != (4,):
            raise ValueError('Expected wrench[12], ee[6], gripper[4]')
        force, ee, width = wrench[:3], ee[:3], float(gripper[:2].mean())
        rotation = np.asarray(robot.get('wrist_rotation', np.eye(3)))
        if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
            raise ValueError('Finite wrist rotation [3,3] required')
        force = rotation @ force
        if not np.isfinite(np.r_[force, ee, width]).all():
            raise ValueError('Finite force/proprioception required')
        speed = 0. if self.previous_ee is None else float(np.linalg.norm(ee-self.previous_ee)/dt)
        self.previous_ee = ee.copy()
        if expected_grasp != self.expected:
            self.forces.clear()
            self.steady_frames = self.bad_frames = 0
            self.expected = expected_grasp
        steady = speed <= self.steady_speed_m_s
        self.steady_frames = self.steady_frames+1 if steady else 0
        self.forces.append(force)
        self.filtered = np.median(self.forces, axis=0)
        if not expected_grasp:
            self.reference = self.reference_width = None
            self.bad_frames = 0
            if (width >= .032 and self.steady_frames >= self.calibration_frames
                    and len(self.forces) >= self.calibration_frames):
                self.unloaded = self.filtered.copy()
            return {'state': 'unloaded' if self.unloaded is not None else 'uncalibrated',
                    'slip': False, 'speed_m_s': speed, 'load_n': None}
        load = None if self.unloaded is None else self.filtered-self.unloaded
        if (self.reference is None and load is not None
                and self.steady_frames >= self.calibration_frames
                and len(self.forces) >= self.filter_frames):
            if load[2] >= self.minimum_load_n and .001 < width < .032:
                self.reference = np.array([0., 0., load[2]])
                self.reference_width = width
        if self.reference is None:
            return {'state': 'uncalibrated', 'slip': False, 'speed_m_s': speed,
                    'load_n': float(np.linalg.norm(load)) if load is not None else None}
        reference_n, retained = float(self.reference[2]), float(load[2])
        # Require both relative load loss and low absolute support: a bottle
        # rotating into a stable grasp can change load without being lost.
        force_loss = retained < min(self.minimum_load_n, reference_n*(1-self.loss_fraction))
        collapsed = width < max(.0005, self.reference_width*.2)
        widened = width > self.reference_width+.006
        suspect = (robot.get('slip_detection_enabled', True) and steady
                   and force_loss and (collapsed or widened))
        self.bad_frames = self.bad_frames+1 if suspect else 0
        slip = self.bad_frames >= self.debounce
        return {'state': 'slip' if slip else 'loaded', 'slip': slip, 'speed_m_s': speed,
                'load_n': float(np.linalg.norm(load)), 'reference_load_n': reference_n,
                'retained_load_n': retained, 'finger_collapse': collapsed,
                'finger_opening': widened, 'bad_frames': self.bad_frames}


class RecoveryExecutor:
    """Privileged scripted bottle regrasp; holds right wrist and preempts chunks."""
    def __init__(self,env,monitor=None,detector='oracle'):
        if detector not in ('oracle', 'wrench'):
            raise ValueError('Recovery detector must be oracle or wrench')
        self.env=env
        self.monitor=monitor or SlipMonitor()
        self.detector=detector
        self.force_monitor=WrenchSlipMonitor() if detector=='wrench' else None
        self.last_detection=None
        self.stage='idle'
        self.age=0
        self.target=None
        self.grasp_goal=None
        self.rotation=np.diag([1.,-1.,-1.])
        self.resume_phase=None

    @property
    def active(self):
        return self.stage!='idle'

    def observe(self,obs,phase):
        if self.force_monitor is None:
            state=self.monitor.update(obs['ee'][:3],obs['object'][:3],obs['wrench'],phase in [3,4,5])
        else:
            robot={key:obs[key] for key in ('ee','gripper','wrench')}
            robot['wrist_rotation']=self.env.data.site('left_wrist').xmat.reshape(3,3).copy()
            robot['slip_detection_enabled']=phase in (3,4) and not self.active
            expected=phase in (2,3,4,5) if not self.active else self.stage in ('grasp','lift')
            self.last_detection=self.force_monitor.update(robot,expected,self.env.dt)
            if self.last_detection['slip'] and self.monitor.state==RecoveryState.RUNNING:
                self.monitor.retries+=1
                self.monitor.state=(RecoveryState.HOLD if self.monitor.retries<=self.monitor.max_retries
                                    else RecoveryState.FAILED)
            state=self.monitor.state
        if state==RecoveryState.HOLD and not self.active:
            self.stage='hold';self.age=0;self.resume_phase=phase
        return state

    def action(self,obs):
        from models.impedance_controller import rotation_error
        self.age+=1
        action=np.r_[np.zeros(6),-1.,np.zeros(6),-1.].astype(np.float32)
        if self.stage=='hold':
            if self.age>=5:
                self.monitor.advance();self.target=obs['object'][:3].copy()
                self.grasp_goal=self.target+[0,0,.025]
                if abs(self.env.data.body('bottle').xmat[8]) < .7:
                    self.grasp_goal[2]=max(self.target[2]+.008,.44)
                    axis=-self.env.data.body('bottle').xmat.reshape(3,3)[:,2].copy()
                    axis[2]=0.;axis/=np.linalg.norm(axis)
                    tool_z=(axis+[0,0,-1])/np.sqrt(2)
                    tool_y=np.cross(axis,[0,0,1])
                    tool_x=np.cross(tool_y,tool_z)
                    self.rotation=np.column_stack([tool_x,tool_y,tool_z])
                self.stage='approach';self.age=0
            return action
        # The right gripper remains closed on the drawer while left hand recovers.
        goal=self.target+[0,0,.18] if self.stage=='approach' else self.grasp_goal.copy()
        if self.stage=='lift':goal=self.target+[0,0,.24]
        action[:3]=np.clip((goal-obs['ee'][:3])/.015,-1,1)
        rotation=self.env.data.site('left_ee').xmat.reshape(3,3)
        action[3:6]=np.clip(rotation_error(self.rotation,rotation)/.08,-1,1)
        action[6]=1. if self.stage in ['approach','descend'] else -1.
        reached=np.linalg.norm(goal-obs['ee'][:3])<.014
        if self.stage=='approach' and reached:
            self.monitor.advance(grounding_ready=True);self.stage='descend';self.age=0
        elif self.stage=='descend' and reached:
            self.stage='grasp';self.age=0
        elif self.stage=='grasp' and self.age>=25:
            self.stage='lift';self.age=0
        elif self.stage=='lift' and reached:
            confirmed=np.linalg.norm(obs['ee'][:3]-obs['object'][:3])<.08 and obs['object'][2]>self.target[2]+.12
            if confirmed:
                self.monitor.advance(grasp_confirmed=True);self.stage='idle';self.age=0
            else:
                self.monitor.state=RecoveryState.FAILED
        if self.age>350:
            self.monitor.state=RecoveryState.FAILED
        return action
