"""Explicit, serializable options shared by visual evaluation and paired plans."""
import hashlib
import json
import math

from simulation.vision import FULL_INSTRUCTION, OPEN_INSTRUCTION, validate_place_contact_mode
from simulation.transfer import TRANSFER_DEFAULTS, validate_transfer_options
from simulation.placement_contact import PLACEMENT_DEFAULTS, validate_placement_options
from simulation.verified_placement import RELEASE_DEFAULTS, validate_release_options

PERCEPTION_DEFAULTS = {
    'legacy_perception': False, 'enable_arm_mask': False, 'enable_temporal_smoothing': False,
    'fallback_max_age_s': .4, 'perception_every': 5, 'ablation': 'none',
    'async_perception': False, 'realtime': False, 'max_sim_age': .4, 'max_wall_age': .4,
    'max_perception_gap': 1., 'max_rejections': 3,
}
CONTROLLER_DEFAULTS = {
    'transition_tolerance': .012, 'transfer_tolerance': .012, 'place_contact_mode': 'off',
    'place_contact_force_threshold_n': 8., 'place_contact_debounce_steps': 3,
    'slip_monitor': False, 'stop_on_slip': False,
    'placement_observer': False, **PLACEMENT_DEFAULTS,
    **RELEASE_DEFAULTS,
    **TRANSFER_DEFAULTS,
}


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def normalize_options(perception=None, controller=None):
    result = {}
    for name, provided, defaults in [('perception_options', perception, PERCEPTION_DEFAULTS),
                                      ('controller_options', controller, CONTROLLER_DEFAULTS)]:
        provided = {} if provided is None else provided
        if not isinstance(provided, dict) or set(provided)-set(defaults):
            raise ValueError(f'Unknown or malformed {name}')
        values = {**defaults, **provided}
        for key, default in defaults.items():
            value = values[key]
            if isinstance(default, bool):
                if type(value) is not bool:
                    raise ValueError(f'{key} must be boolean')
            elif isinstance(default, (float, int)):
                if (isinstance(value, bool) or not isinstance(value, (float, int))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f'{key} must be finite and positive')
                if isinstance(default, int) and int(value) != value:
                    raise ValueError(f'{key} must be an integer')
                values[key] = int(value) if isinstance(default, int) else float(value)
        result[name] = values
    p, c = result['perception_options'], result['controller_options']
    if p['ablation'] not in ('none', 'drop-image', 'swap-text'):
        raise ValueError('Unknown perception ablation')
    if p['legacy_perception'] and (p['enable_arm_mask'] or p['enable_temporal_smoothing']):
        raise ValueError('Legacy perception cannot enable mask or smoothing')
    if c['stop_on_slip'] and not c['slip_monitor']:
        raise ValueError('stop_on_slip requires slip_monitor')
    validate_place_contact_mode(c['place_contact_mode'])
    validate_transfer_options(c)
    validate_placement_options(c)
    validate_release_options(c)
    if c['place_contact_mode']=='calibrated':
        c['placement_observer']=True
    return result


def effective_options(args):
    return normalize_options(
        {key: getattr(args, key, value) for key, value in PERCEPTION_DEFAULTS.items()},
        {key: getattr(args, key, value) for key, value in CONTROLLER_DEFAULTS.items()})


def option_report_fields(options):
    p, c = options['perception_options'], options['controller_options']
    return {'legacy_perception': p['legacy_perception'], 'arm_mask_enabled': p['enable_arm_mask'],
            'temporal_smoothing_enabled': p['enable_temporal_smoothing'],
            'fallback_max_age_s': p['fallback_max_age_s'], 'ablation': p['ablation'],
            'perception_every_actions': p['perception_every'], 'async_perception': p['async_perception'],
            'max_sim_age_s': p['max_sim_age'], 'max_wall_age_s': p['max_wall_age'],
            'max_perception_gap_s': p['max_perception_gap'], 'max_rejections': p['max_rejections'],
            'transition_tolerance_m': c['transition_tolerance'], 'transfer_tolerance_m': c['transfer_tolerance'],
            'transfer_mode': c['transfer_mode'],
            'placement_observer': c['placement_observer'],
            'place_contact_mode': c['place_contact_mode'],
            'place_contact_force_threshold_n': c['place_contact_force_threshold_n'],
            'place_contact_debounce_steps': c['place_contact_debounce_steps'],
            'slip_monitor': c['slip_monitor'], 'stop_on_slip': c['stop_on_slip']}


def check_instruction(instruction):
    if instruction not in (FULL_INSTRUCTION, OPEN_INSTRUCTION):
        raise ValueError('Unsupported finite-task instruction')
    return instruction
