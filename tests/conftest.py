"""Select offscreen rendering before pytest imports MuJoCo modules."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
