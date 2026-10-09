"""Physics spike: 6 mm/revolution thread constraint, release at two turns.

This fixture is an articulated approximation, not detached free-body uncapping.
"""
from pathlib import Path
import json
import mujoco
import numpy as np


def check():
    path=Path(__file__).resolve().parents[1]/'simulation/assets/screw_fixture.xml'
    model=mujoco.MjModel.from_xml_path(str(path));data=mujoco.MjData(model)
    released=False;max_error=0.;release_jump=None
    for _ in range(30000):
        if not released:
            data.ctrl[0]=.012
            max_error=max(max_error,abs(data.qpos[1]-.006/(2*np.pi)*data.qpos[0]))
            if data.qpos[0]>=4*np.pi:
                before=data.qpos.copy();data.eq_active[model.equality('thread').id]=0
                mujoco.mj_forward(model,data)
                release_jump=float(np.linalg.norm(data.qpos-before));released=True
                data.ctrl[0]=0.;data.ctrl[1]=.02
        mujoco.mj_step(model,data)
        if released and data.qpos[1]>.025:
            break
    report={'released':released,'turns':float(data.qpos[0]/(2*np.pi)),'height':float(data.qpos[1]),
            'max_thread_error_m':float(max_error),'release_pose_jump':release_jump,
            'limitation':'hinge+slide fixture; no free-body detach or bimanual integration'}
    if not released or max_error>.001 or release_jump!=0:
        raise RuntimeError(report)
    print(json.dumps(report,indent=2));return report

if __name__=='__main__':
    check()
