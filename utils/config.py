"""Resolve environment config without depending on the current directory."""
from pathlib import Path
import yaml

ROOT=Path(__file__).resolve().parents[1]


def make_environment(config_path=None,**overrides):
    from simulation.bimanual_env import BimanualBottleDrawerEnv
    path=Path(config_path) if config_path else ROOT/'configs/env_config.yaml'
    if not path.is_absolute() and not path.exists():path=ROOT/path
    config=yaml.safe_load(path.read_text())['env']
    expected_controller=1/(2*config['physics_dt'])
    if config['controller_hz']!=expected_controller:
        raise ValueError('Current controller updates every 2 physics steps; controller_hz must match physics_dt')
    args={'xml_path':config['xml_path'],'control_freq':config['action_hz'],'max_episode_steps':config['max_episode_steps'],
          'images':config.get('images',False),'image_size':config.get('image_size',128)}
    args.update({key:value for key,value in overrides.items() if value is not None})
    env=BimanualBottleDrawerEnv(**args)
    if abs(env.model.opt.timestep-config['physics_dt'])>1e-9:
        env.close();raise ValueError('Config physics_dt does not match MJCF timestep')
    return env
