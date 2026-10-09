"""Conservative slip monitor with debounce and bounded recovery FSM.

Detection is exposed separately until recovery demonstrations are validated.
"""
from enum import Enum
import numpy as np

class RecoveryState(str,Enum):
    RUNNING='running'
    HOLD='hold'
    REGROUND='reground'
    REGRASP='regrasp'
    FAILED='failed'

class SlipMonitor:
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


class RecoveryExecutor:
    """Privileged scripted bottle regrasp; holds right wrist and preempts chunks."""
    def __init__(self,env,monitor=None):
        self.env=env
        self.monitor=monitor or SlipMonitor()
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
        state=self.monitor.update(obs['ee'][:3],obs['object'][:3],obs['wrench'],phase in [3,4,5])
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
