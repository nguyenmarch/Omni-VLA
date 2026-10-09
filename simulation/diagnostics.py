"""Read-only measurements taken before executing a policy action."""
import numpy as np
from models.impedance_controller import rotation_error
from simulation.expert import PHASES


def action_diagnostics(env, expert, action):
    left, right, _, _ = expert.goals()
    reference = expert.action()
    distances, angles = [], []
    for side, goal in [('left', left), ('right', right)]:
        distances.append(float(np.linalg.norm(goal - env.site_pos(f'{side}_ee'))))
        rotation = env.data.site(f'{side}_ee').xmat.reshape(3, 3)
        angles.append(float(np.linalg.norm(rotation_error(expert.rotation, rotation))))
    return {'step': env.step_counter, 'time': float(env.data.time),
            'phase': PHASES[expert.phase], 'position_error_m': distances,
            'rotation_error_rad': angles, 'action': np.asarray(action).tolist(),
            'expert_action': reference.tolist(),
            'action_mae': float(np.mean(np.abs(action - reference)))}
