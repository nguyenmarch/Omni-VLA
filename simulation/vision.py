"""RGB-D capture and finite-language phase management without object ground truth."""
import numpy as np
from utils.camera import calibration
from simulation.expert import PHASES

ROBOT_KEYS=('q','dq','gripper','wrench','ee')
FULL_INSTRUCTION='stow the bottle in the drawer then close it'
OPEN_INSTRUCTION='open the drawer'
QUERIES=('grasp the bottle body','grasp the drawer handle')


def capture_head(env):
    renderer=env._renderer()
    renderer.disable_depth_rendering();renderer.update_scene(env.data,camera='head_cam')
    rgb=renderer.render().copy()
    renderer.enable_depth_rendering();renderer.update_scene(env.data,camera='head_cam')
    depth=renderer.render().copy();renderer.disable_depth_rendering()
    k,t=calibration(env.model,env.data,'head_cam',env.image_size,env.image_size)
    return {'rgb':rgb,'depth':depth,'K':k,'T':t}


def robot_observation(obs):
    return {k:np.asarray(obs[k]).copy() for k in ROBOT_KEYS}


class VisualTaskManager:
    """Known drawer geometry, robot proprioception and perceived landmarks only.

    Finite task grammar; no expert actions, simulator object sites or success
    flags. Opening is relative to the perceived handle at episode start.
    """
    def __init__(self,points,instruction=FULL_INSTRUCTION,transition_tolerance=.012):
        if instruction not in (FULL_INSTRUCTION,OPEN_INSTRUCTION):
            raise ValueError('Only the two documented English task instructions are supported')
        self.open_only=instruction==OPEN_INSTRUCTION
        if not np.isfinite(transition_tolerance) or transition_tolerance <= 0:
            raise ValueError('Transition tolerance must be finite and positive')
        self.transition_tolerance=transition_tolerance
        self.initial=np.asarray(points,dtype=np.float32).copy()
        if self.initial.shape!=(2,3) or not np.isfinite(self.initial).all():
            raise ValueError('Expected two finite perceived 3D points')
        self.points=self.initial.copy();self.phase=0;self.age=0
        self.failed=False;self.completed=False;self.failure_reason=''
        self.opened=False;self.placed=False

    @property
    def opening(self):
        return float(np.clip(self.initial[1,0]-self.points[1,0],0,.22))

    def observe(self,points):
        p=np.asarray(points,dtype=np.float32)
        if p.shape!=(2,3) or not np.isfinite(p).all():raise ValueError('Expected two finite perceived 3D points')
        self.points=p.copy()

    def condition(self,robot):
        objects=np.r_[self.points.flatten(),self.opening]
        return np.concatenate([np.asarray(robot[k]) for k in ROBOT_KEYS]+[objects,np.eye(10,dtype=np.float32)[self.phase]]).astype(np.float32)

    def goals(self):
        bottle,handle=self.initial
        center=self.points[1]+[.19,0,-.09]  # calibrated fixture geometry
        left=np.r_[bottle[:2],.72];right=handle+[0,0,.18]
        if self.phase>=1:left=bottle+[0,0,.025];right=handle.copy()
        if self.phase>=3:left=np.r_[bottle[:2],.72];right=handle+[-.205,0,0]
        if self.phase>=4:left=np.r_[center[:2],.72]
        if self.phase>=5:left=center+[0,0,.12]
        if self.phase>=7:left=np.r_[center[:2],.78]
        if self.phase>=8:right=handle+[.005,0,0]
        return left,right

    def update(self,robot):
        self.age+=1;lp,rp=np.asarray(robot['ee']).reshape(2,3);left,right=self.goals()
        reached=(self.open_only or np.linalg.norm(lp-left)<self.transition_tolerance) and np.linalg.norm(rp-right)<self.transition_tolerance
        self.opened|=self.opening>.16
        center=self.points[1]+[.19,0,-.09];bottle=self.points[0]
        inside=abs(bottle[0]-center[0])<.11 and abs(bottle[1]-center[1])<.075 and .445<bottle[2]<.57
        released=np.asarray(robot['gripper'])[:2].mean()>.032
        self.placed|=self.opened and inside and released
        if self.open_only and self.phase==3 and self.opened:self.completed=True;return
        if self.phase in (2,6):ready=self.age>=20
        elif self.phase==3:ready=self.opened and (self.open_only or lp[2]>.69)
        elif self.phase==5:ready=np.linalg.norm(lp-left)<.025 and np.linalg.norm(rp-right)<.02
        elif self.phase==8:ready=self.opening<.03
        elif self.phase==9:ready=self.placed and inside and released and self.opening<.035
        else:ready=reached
        if ready:
            if self.phase==9:self.completed=True
            else:self.phase+=1;self.age=0
        if self.age>350:self.failed=True;self.failure_reason='visual_timeout_'+PHASES[self.phase]

    def adapt_action(self,action):
        a=np.asarray(action).copy()
        if self.open_only:a[:6]=0.;a[6]=1.  # explicit finite-task arm selection
        return a
