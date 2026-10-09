"""Append corrective episodes to training only, preserve base held-out split."""
import argparse
import json
from pathlib import Path
import shutil
import h5py
from utils.checkpoint import file_hash


def merge(base, correction, output, state_scene_audit=None):
    if Path(output).exists():
        raise FileExistsError(output)
    audit = json.loads(Path(state_scene_audit).read_text()) if state_scene_audit else None
    with h5py.File(base) as dst, h5py.File(correction) as src:
        for key in ('schema_version', 'action_dt', 'condition_dim', 'action_dim', 'action_convention'):
            if dst.attrs[key] != src.attrs[key]:
                raise ValueError(f'Incompatible {key}')
        changed_scene = dst.attrs['scene_sha256'] != src.attrs['scene_sha256']
        if changed_scene:
            if (not audit or not audit.get('passed')
                or audit['dataset_sha256'] != file_hash(base)
                or audit['legacy_scene_sha256'] != dst.attrs['scene_sha256']
                or audit['current_scene_sha256'] != src.attrs['scene_sha256']):
                raise ValueError('Incompatible scene_sha256; require matching state-only replay audit')
            for h in (dst, src):
                for g in h['episodes'].values():
                    invalid = []
                    g['obs'].visititems(lambda k, v: invalid.append(k) if isinstance(v, h5py.Dataset) and v.ndim > 2 else None)
                    if invalid:
                        raise ValueError('Scene migration accepts state-only data')
        heldout = {int(dst['episodes'][name].attrs['seed'])
                   for split in ('val', 'test') for name in json.loads(dst.attrs['splits'])[split]}
        if any(int(g.attrs['seed']) in heldout for g in src['episodes'].values()):
            raise ValueError('Correction seed overlaps held-out episodes')
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(output) + '.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    shutil.copyfile(base, temporary)
    try:
        with h5py.File(temporary, 'r+') as dst, h5py.File(correction) as src:
            splits = json.loads(dst.attrs['splits'])
            for name, g in src['episodes'].items():
                new = 'correction_' + name
                suffix = 1
                while new in dst['episodes']:
                    new = f'correction_{suffix}_{name}'
                    suffix += 1
                src.copy(g, dst['episodes'], name=new)
                splits['train'].append(new)
            dst.attrs['splits'] = json.dumps(splits)
            dst.attrs['correction_source'] = str(correction)
            dst.attrs['base_dataset_sha256'] = file_hash(base)
            dst.attrs['correction_dataset_sha256'] = file_hash(correction)
            if changed_scene:
                dst.attrs['legacy_scene_sha256'] = dst.attrs['scene_sha256']
                dst.attrs['scene_sha256'] = src.attrs['scene_sha256']
                dst.attrs['state_scene_audit'] = json.dumps(audit)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', required=True)
    p.add_argument('--correction', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--state-scene-audit', help='Passed replay audit for explicit state-only scene migration')
    a = p.parse_args()
    merge(a.base, a.correction, a.output, a.state_scene_audit)
