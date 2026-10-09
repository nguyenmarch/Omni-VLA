"""Torque Cartesian impedance for Panda arms using MuJoCo Jacobians."""
import mujoco
import numpy as np


def rotation_error(target, current):
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (target @ current.T).reshape(-1))
    if quat[0] < 0:
        quat *= -1
    norm = np.linalg.norm(quat[1:])
    return quat[1:] * (2 * np.arctan2(norm, quat[0]) / max(norm, 1e-9))


def exp_rotation(vector):
    angle = np.linalg.norm(vector)
    if angle < 1e-9:
        return np.eye(3)
    x, y, z = vector / angle
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * cross @ cross


class CartesianImpedanceController:
    def __init__(self, model, side, kp=700., kd=55.):
        self.side = side
        self.site = model.site(f'{side}_ee').id
        self.qadr = np.array([model.joint(f'{side}_joint{i}').qposadr[0] for i in range(1, 8)])
        self.dofs = np.array([model.joint(f'{side}_joint{i}').dofadr[0] for i in range(1, 8)])
        self.actuators = np.array([model.actuator(f'{side}_motor{i}').id for i in range(1, 8)])
        self.kp = np.array([kp]*3 + [50.]*3)
        self.kd = np.array([kd]*3 + [12.]*3)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self.home = np.array([0., 0., 0., -1.57079, 0., 1.57079, .7854])

    def compute_torque(self, model, data, target_pose):
        position, rotation = target_pose
        mujoco.mj_jacSite(model, data, self.jp, self.jr, self.site)
        jac = np.vstack([self.jp[:, self.dofs], self.jr[:, self.dofs]])
        error = np.r_[position - data.site_xpos[self.site], rotation_error(rotation, data.site_xmat[self.site].reshape(3, 3))]
        velocity = jac @ data.qvel[self.dofs]
        torque = jac.T @ (self.kp * error - self.kd * velocity)
        # Damped task nullspace, plus full MuJoCo bias (gravity/Coriolis).
        projector = np.eye(7) - jac.T @ np.linalg.solve(jac @ jac.T + .001*np.eye(6), jac)
        torque += projector @ (8*(self.home-data.qpos[self.qadr]) - 2*data.qvel[self.dofs])
        torque += data.qfrc_bias[self.dofs]
        return np.clip(torque, model.actuator_ctrlrange[self.actuators, 0], model.actuator_ctrlrange[self.actuators, 1])
