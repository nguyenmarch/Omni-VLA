import os
os.environ.setdefault('MUJOCO_GL','egl')
import numpy as np
from simulation.bimanual_env import BimanualBottleDrawerEnv
from simulation.multicamera import MultiCameraCapture


def test_camera_sizes_stamps_and_fk_mask_preserve_physics():
    env=BimanualBottleDrawerEnv(image_size=256);capture=None
    try:
        env.reset(seed=9000);before=env.data.qpos.copy();capture=MultiCameraCapture(env.model)
        frames=capture.capture(env.data,episode_id=9000,audit=True)
        np.testing.assert_array_equal(before,env.data.qpos)
        # Object motion must not alter the robot-only FK render.
        env.data.qpos[env.bottle_adr]+=0.1
        more=capture.capture(env.data,episode_id=9000)
        for cam in capture.SIZES:
            np.testing.assert_array_equal(frames[cam]['robot_depth'],more[cam]['robot_depth'])
        env.data.qpos[:]=before
        for cam,size in capture.SIZES.items():
            f=frames[cam];assert f['rgb'].shape==(size,size,3)
            assert f['depth'].shape==f['excluded'].shape==(size,size)
            assert f['episode_id']==9000 and f['sequence']==1 and f['simulation_time']==env.data.time
            assert np.isfinite(f['K']).all() and np.isfinite(f['T']).all()
            predicted=f['excluded'];truth=f['audit_robot_mask']
            assert (predicted&truth).sum()/(predicted|truth).sum()>.9
            assert np.all(f['masked_rgb'][predicted]==0)
            np.testing.assert_array_equal(f['masked_rgb'][~predicted],f['rgb'][~predicted])
    finally:
        if capture:capture.close()
        env.close()
