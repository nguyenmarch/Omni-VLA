"""MuJoCo camera (-Z forward, +Y up) ↔ CV (+Z forward, +Y down)."""
import numpy as np


def calibration(model, data, camera, width, height):
    cam_id=model.camera(camera).id
    focal=.5*height/np.tan(np.deg2rad(model.cam_fovy[cam_id])*.5)
    intrinsic=np.array([[focal,0,width/2],[0,focal,height/2],[0,0,1]],np.float32)
    world_from_camera=np.eye(4,dtype=np.float32)
    world_from_camera[:3,:3]=data.cam_xmat[cam_id].reshape(3,3)@np.diag([1,-1,-1])
    world_from_camera[:3,3]=data.cam_xpos[cam_id]
    return intrinsic,world_from_camera


def project(point_world, intrinsic, world_from_camera):
    point=np.linalg.solve(world_from_camera,np.r_[point_world,1])[:3]
    uv=intrinsic@point
    return uv[:2]/uv[2],point[2]


def unproject(uv, depth, intrinsic, world_from_camera):
    point_camera=depth*np.linalg.solve(intrinsic,np.r_[uv,1])
    return (world_from_camera@np.r_[point_camera,1])[:3]
