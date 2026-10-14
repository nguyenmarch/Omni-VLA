"""RGB-D capture and finite-language phase management without object ground truth."""
import numpy as np
from utils.camera import calibration
from simulation.expert import PHASES

ROBOT_KEYS=('q','dq','gripper','wrench','ee')
FULL_INSTRUCTION='stow the bottle in the drawer then close it'
OPEN_INSTRUCTION='open the drawer'
QUERIES=('grasp the bottle body','grasp the drawer handle')


def capture_head(env, arm_mask=False, depth_max_m=5.0):
    renderer=env._renderer()
    renderer.disable_depth_rendering();renderer.update_scene(env.data,camera='head_cam')
    rgb=renderer.render().copy()
    renderer.enable_depth_rendering();renderer.update_scene(env.data,camera='head_cam')
    depth=renderer.render().copy();renderer.disable_depth_rendering()
    k,t=calibration(env.model,env.data,'head_cam',env.image_size,env.image_size)
    frame={'rgb':rgb,'depth':depth,'K':k,'T':t,
           'depth_max_m':float(depth_max_m),
           'sim_time':float(env.data.time)}
    if arm_mask:
        frame['robot_mask']=projected_arm_mask(env.model,env.data,k,t,depth,env.image_size)
    return frame


def projected_arm_mask(model,data,intrinsic,world_from_camera,depth,size):
    """Mask robot link pixels only when measured depth agrees with link FK."""
    mask=np.zeros((size,size),dtype=bool)
    depth=np.asarray(depth,dtype=np.float32)
    rotation=world_from_camera[:3,:3]
    translation=world_from_camera[:3,3]
    fx=float(intrinsic[0,0])
    for side in ('left','right'):
        ids=[]
        for index in range(model.nbody):
            name=model.body(index).name
            if name.startswith(f'{side}_link') or name==f'{side}_hand':
                ids.append((name,index))
        ids.sort(key=lambda row: (0 if row[0].endswith('link0') else 1,
                                  int(row[0].split('link')[-1]) if 'link' in row[0] else 99))
        camera_points=[]
        for _,body_id in ids:
            world=np.asarray(data.xpos[body_id])
            camera=(world-translation)@rotation
            if camera[2]>.05:
                pixel=intrinsic@camera
                camera_points.append((pixel[:2]/pixel[2],float(camera[2])))
            else:
                camera_points.append((None,float(camera[2])))
        for (p0,z0),(p1,z1) in zip(camera_points,camera_points[1:]):
            if p0 is None or p1 is None:
                continue
            length=float(np.linalg.norm(p1-p0))
            samples=max(1,int(np.ceil(length*1.5)))
            for alpha in np.linspace(0.,1.,samples+1):
                uv=p0*(1-alpha)+p1*alpha
                z=z0*(1-alpha)+z1*alpha
                radius=int(np.clip(np.ceil(fx*.045/max(z,.05)),2,18))
                cx,cy=np.rint(uv).astype(int)
                x0,x1=max(0,cx-radius),min(size,cx+radius+1)
                y0,y1=max(0,cy-radius),min(size,cy+radius+1)
                if x0>=x1 or y0>=y1:
                    continue
                yy,xx=np.ogrid[y0:y1,x0:x1]
                disk=(xx-cx)**2+(yy-cy)**2<=radius**2
                observed=depth[y0:y1,x0:x1]
                on_robot=(np.isfinite(observed) & (observed>.05) & (observed<3.)
                          & (np.abs(observed-z)<=.09))
                mask[y0:y1,x0:x1] |= disk & on_robot
    return mask


def robot_observation(obs):
    return {k:np.asarray(obs[k]).copy() for k in ROBOT_KEYS}


class VisualTaskManager:
    """Known drawer geometry, robot proprioception and perceived landmarks only.

    Finite task grammar; no expert actions, simulator object sites or success
    flags. Opening is relative to the perceived handle at episode start.
    """
    def __init__(self,points,instruction=FULL_INSTRUCTION,transition_tolerance=.012,
                 transfer_tolerance=.025,place_contact_force_threshold_n=8.,
                 place_contact_debounce_steps=3):
        if instruction not in (FULL_INSTRUCTION,OPEN_INSTRUCTION):
            raise ValueError('Only the two documented English task instructions are supported')
        self.open_only=instruction==OPEN_INSTRUCTION
        if not np.isfinite(transition_tolerance) or transition_tolerance <= 0:
            raise ValueError('Transition tolerance must be finite and positive')
        self.transition_tolerance=transition_tolerance
        if not np.isfinite(transfer_tolerance) or transfer_tolerance <= 0:
            raise ValueError('Transfer tolerance must be finite and positive')
        if not np.isfinite(place_contact_force_threshold_n) or place_contact_force_threshold_n <= 0:
            raise ValueError('Place contact force threshold must be finite and positive')
        if int(place_contact_debounce_steps) < 1:
            raise ValueError('Place contact debounce must be positive')
        self.transfer_tolerance=float(transfer_tolerance)
        self.place_contact_force_threshold_n=float(place_contact_force_threshold_n)
        self.place_contact_debounce_steps=int(place_contact_debounce_steps)
        self.place_contact_streak=0
        self.place_contact_fz_n=0.
        self.last_transition_reason=''
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
        self.last_transition_reason=''
        tolerance=self.transfer_tolerance if self.phase==4 else self.transition_tolerance
        reached=(self.open_only or np.linalg.norm(lp-left)<tolerance) and np.linalg.norm(rp-right)<tolerance
        self.opened|=self.opening>.16
        center=self.points[1]+[.19,0,-.09];bottle=self.points[0]
        inside=abs(bottle[0]-center[0])<.11 and abs(bottle[1]-center[1])<.075 and .445<bottle[2]<.57
        released=np.asarray(robot['gripper'])[:2].mean()>.032
        self.placed|=self.opened and inside and released
        if self.open_only and self.phase==3 and self.opened:self.completed=True;return
        if self.phase in (2,6):ready=self.age>=20
        elif self.phase==3:ready=self.opened and (self.open_only or lp[2]>.69)
        elif self.phase==5:
            wrench=np.asarray(robot['wrench'],dtype=float)
            if wrench.shape!=(12,) or not np.isfinite(wrench).all():
                raise ValueError('Expected finite wrench[12] for place contact trigger')
            # The left wrist force sensor reports in its site frame. Rotate to
            # world coordinates so Fz means vertical contact in the drawer.
            wrist_rotation=np.asarray(robot.get('left_wrist_rotation',np.eye(3)),dtype=float)
            if wrist_rotation.shape!=(3,3) or not np.isfinite(wrist_rotation).all():
                raise ValueError('Expected finite left wrist rotation[3,3]')
            self.place_contact_fz_n=float((wrist_rotation@wrench[:3])[2])
            drawer_center=self.points[1]+[.19,0,-.09]
            near_drawer=(abs(lp[0]-drawer_center[0])<.13 and abs(lp[1]-drawer_center[1])<.10
                         and np.linalg.norm(lp-left)<.10)
            contact=near_drawer and abs(self.place_contact_fz_n)>=self.place_contact_force_threshold_n
            self.place_contact_streak=self.place_contact_streak+1 if contact else 0
            reached_place=np.linalg.norm(lp-left)<.025 and np.linalg.norm(rp-right)<.02
            ready=reached_place or self.place_contact_streak>=self.place_contact_debounce_steps
            if ready:
                self.last_transition_reason=('wrench_contact' if self.place_contact_streak>=self.place_contact_debounce_steps
                                             else 'place_goal_reached')
        elif self.phase==8:ready=self.opening<.03
        elif self.phase==9:ready=self.placed and inside and released and self.opening<.035
        else:ready=reached
        if ready:
            if self.phase==4:self.last_transition_reason='transfer_goal_reached'
            if self.phase==9:self.completed=True
            else:self.phase+=1;self.age=0
        if self.age>350:self.failed=True;self.failure_reason='visual_timeout_'+PHASES[self.phase]

    def adapt_action(self,action):
        a=np.asarray(action).copy()
        if self.open_only:a[:6]=0.;a[6]=1.  # explicit finite-task arm selection
        return a
