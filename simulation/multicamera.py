"""Synchronous stamped RGB-D and self-body prior using MuJoCo MJCF FK.

This simulator backend is not Pinocchio. Robot-only depth uses joint telemetry
and robot geometry; scene segmentation is available solely for mask auditing.
No object landmarks enter the mask. Call from the physics owner thread.
"""
import copy
import time
import mujoco
import numpy as np
from utils.camera import calibration


class MultiCameraCapture:
    SIZES={'head_cam':256,'right_wrist_cam':224}
    def __init__(self,model):
        self.robot_geoms=np.array([model.body(int(b)).name.startswith(('left_','right_')) for b in model.geom_bodyid])
        robot_joints=[j for j,b in enumerate(model.jnt_bodyid) if model.body(int(b)).name.startswith(('left_','right_'))]
        if any(int(model.jnt_type[j]) not in (int(mujoco.mjtJoint.mjJNT_HINGE),int(mujoco.mjtJoint.mjJNT_SLIDE)) for j in robot_joints):
            raise ValueError('This fixed-base FK backend requires scalar robot joints')
        self.robot_qadr=np.asarray([model.jnt_qposadr[j] for j in robot_joints],dtype=int)
        self.robot_model=copy.copy(model)
        self.robot_model.geom_group[:]=np.where(self.robot_geoms,0,5)
        self.robot_data=mujoco.MjData(self.robot_model)
        self.model=model;self.renderers={};self.robot_renderers={};self.sequence=0
        self.option=mujoco.MjvOption();self.option.geomgroup[:]=[1,0,0,0,0,0];self.option.sitegroup[:]=0
        # Each context and MjData stays on this caller thread.
        for camera,size in self.SIZES.items():
            self.renderers[camera]=mujoco.Renderer(model,height=size,width=size)
            self.robot_renderers[camera]=mujoco.Renderer(self.robot_model,height=size,width=size)

    def capture(self,data,episode_id=0,audit=False):
        self.robot_data.qpos[self.robot_qadr]=data.qpos[self.robot_qadr]
        mujoco.mj_forward(self.robot_model,self.robot_data)
        self.sequence+=1;frames={}
        for camera,size in self.SIZES.items():
            renderer=self.renderers[camera];renderer.disable_depth_rendering();renderer.disable_segmentation_rendering()
            renderer.update_scene(data,camera=camera);rgb=renderer.render().copy()
            renderer.enable_depth_rendering();renderer.update_scene(data,camera=camera);depth=renderer.render().copy()
            renderer.disable_depth_rendering()
            rr=self.robot_renderers[camera];rr.enable_depth_rendering()
            rr.update_scene(self.robot_data,camera=camera,scene_option=self.option);robot_depth=rr.render().copy()
            # Preserve foreground objects covering a robot link. Missing depth is not mask evidence.
            mask=np.isfinite(depth)&np.isfinite(robot_depth)&(depth>0)&(depth<5)&(robot_depth>0)&(robot_depth<5)&(np.abs(depth-robot_depth)<=.01)
            masked=rgb.copy();masked[mask]=0
            k,t=calibration(self.model,data,camera,size,size)
            frame={'rgb':rgb,'masked_rgb':masked,'depth':depth,'excluded':mask,'robot_depth':robot_depth,'K':k,'T':t,
                'camera':camera,'episode_id':int(episode_id),'sequence':self.sequence,'simulation_time':float(data.time),
                'wall_time_ns':time.monotonic_ns(),'mask_backend':'mujoco_mjcf_fk'}
            if audit:
                renderer.enable_segmentation_rendering();renderer.update_scene(data,camera=camera);seg=renderer.render().copy()
                renderer.disable_segmentation_rendering()
                ids=seg[:,:,0];is_geom=seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM)
                frame['audit_robot_mask']=is_geom&(ids>=0)&self.robot_geoms[ids.clip(0,len(self.robot_geoms)-1)]
            frames[camera]=frame
        return frames

    def close(self):
        for renderer in (*self.renderers.values(),*self.robot_renderers.values()):renderer.close()
