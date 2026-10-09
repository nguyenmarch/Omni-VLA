from pathlib import Path
import hashlib
import torch
from models.bc_policy import BehaviorCloningPolicy
from models.flow_matching_policy import FlowMatchingActionPolicy


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_policy(kind, config):
    cls = {'bc': BehaviorCloningPolicy, 'cfm': FlowMatchingActionPolicy}[kind]
    kwargs = {key: config[key] for key in ('action_dim','chunk_size','cond_dim','hidden_dim')}
    if kind == 'cfm':
        kwargs['direct_skip'] = config.get('direct_skip',False)
    else:
        kwargs['phase_heads'] = config.get('phase_heads',False)
    return cls(**kwargs)


def load_policy(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get('schema_version') != 'omni-state-v1':
        raise ValueError('Checkpoint schema incompatible with drawer state baseline')
    model = make_policy(checkpoint['policy_type'], checkpoint['model_config']).to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model, checkpoint


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
