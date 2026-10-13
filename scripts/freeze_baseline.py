"""Freeze and verify the local drawer/grounding benchmark inputs and reports.

The manifest records hashes for large local HDF5/checkpoint files without
copying them into Git. Source code itself is frozen by the Git commit.
"""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import torch


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs' / 'REPRO_BASELINE.json'
INPUTS = (
    'data/drawer_dagger_v2.h5',
    'data/grounding_rgbd_v2_256.h5',
    'data/grounding_patch_v3_256.h5',
    'weights/bc_kaggle_v2/best.pt',
    'weights/cfm_kaggle_v2/best.pt',
    'weights/cfm_h1_kaggle_v2/best.pt',
    'weights/grounding_patch_v3_cpu_pilot/selected.pt',
    'configs/train_bc_dagger_v2.yaml',
    'configs/train_cfm_dagger_v2.yaml',
    'configs/train_cfm_h1_v2.yaml',
)
REPORTS = (
    'artifacts/expert_100.json',
    'artifacts/bc_kaggle_v2_same.json',
    'artifacts/bc_kaggle_v2_fresh.json',
    'artifacts/bc_kaggle_v2_comparison.json',
    'artifacts/cfm_kaggle_v2/comparison.json',
    'artifacts/cfm_h1_kaggle_v2/comparison.json',
    'artifacts/patch_grounding_v3/selected_val.json',
    'artifacts/patch_grounding_v3/selected_test.json',
    'artifacts/patch_grounding_v3/closed_loop_selected.json',
    'artifacts/bc_kaggle_v2_import.json',
    'artifacts/cfm_kaggle_v2_import.json',
    'artifacts/cfm_h1_kaggle_v2_import.json',
    'data/grounding_patch_v3_256.report.json',
) + tuple(
    f'artifacts/cfm_{prefix}kaggle_v2/euler_{step}_{label}.json'
    for prefix in ('', 'h1_') for step in (4, 6, 8) for label in ('same', 'fresh')
)
DATASETS = ('data/drawer_dagger_v2.h5', 'data/grounding_patch_v3_256.h5')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads((ROOT / path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dataset_splits(path):
    result = {}
    with h5py.File(ROOT / path) as dataset:
        splits = json.loads(dataset.attrs['splits'])
        names = [name for group in splits.values() for name in group]
        require(len(names) == len(set(names)), f'Overlapping episode splits: {path}')
        all_seeds = [int(dataset['episodes'][name].attrs['seed']) for name in names]
        require(len(all_seeds) == len(set(all_seeds)), f'Overlapping seed splits: {path}')
        for split, episode_names in splits.items():
            seeds = [int(dataset['episodes'][name].attrs['seed']) for name in episode_names]
            require(len(seeds) == len(set(seeds)), f'Duplicate seeds in {path}:{split}')
            result[split] = {'episodes': episode_names, 'seeds': seeds}
    return result


def checked_rollout(path, expected_seeds, expected_hash=None):
    report = read_json(path)
    seeds = [row['seed'] for row in report['episodes']]
    require(seeds == expected_seeds, f'Seed sequence mismatch: {path}')
    require(report['attempted'] == len(expected_seeds), f'Incomplete report: {path}')
    require(report['successes'] == sum(bool(row['success']) for row in report['episodes']),
            f'Success count mismatch: {path}')
    if expected_hash is not None:
        require(report.get('checkpoint_sha256') == expected_hash, f'Checkpoint mismatch: {path}')
    return {'report': path, 'successes': report['successes'], 'attempted': report['attempted']}


def build_manifest():
    paths = INPUTS + REPORTS
    hashes = {name: sha256(ROOT / name) for name in paths}
    splits = {name: dataset_splits(name) for name in DATASETS}
    drawer_train = {seed for part in splits['data/drawer_dagger_v2.h5'].values() for seed in part['seeds']}
    grounding_train = {seed for part in splits['data/grounding_patch_v3_256.h5'].values() for seed in part['seeds']}
    sets = {'reused_development': list(range(5000, 5050)), 'fresh_at_bc_v2': list(range(7000, 7050))}
    require(all(not drawer_train.intersection(seeds) for seeds in sets.values()),
            'Drawer evaluation seed overlaps a dataset episode')
    require(not grounding_train.intersection(range(9000, 9005)),
            'Visual rollout seed overlaps a grounding dataset episode')

    bc_hash = hashes['weights/bc_kaggle_v2/best.pt']
    drawer_hash = hashes['data/drawer_dagger_v2.h5']
    patch_hash = hashes['weights/grounding_patch_v3_cpu_pilot/selected.pt']
    patch_data_hash = hashes['data/grounding_patch_v3_256.h5']
    for path, expected_dataset, expected_type in (
        ('weights/bc_kaggle_v2/best.pt', drawer_hash, 'bc'),
        ('weights/cfm_kaggle_v2/best.pt', drawer_hash, 'cfm'),
        ('weights/cfm_h1_kaggle_v2/best.pt', drawer_hash, 'cfm'),
        ('weights/grounding_patch_v3_cpu_pilot/selected.pt', patch_data_hash, None),
    ):
        checkpoint = torch.load(ROOT / path, map_location='cpu', weights_only=True)
        require(checkpoint['dataset_sha256'] == expected_dataset,
                f'Checkpoint trained on different data: {path}')
        if expected_type:
            require(checkpoint['policy_type'] == expected_type
                    and checkpoint['splits'] == {k: v['episodes'] for k, v in splits['data/drawer_dagger_v2.h5'].items()},
                    f'Checkpoint policy or split mismatch: {path}')
        else:
            require(checkpoint['train_episodes'] == splits['data/grounding_patch_v3_256.h5']['train']['episodes']
                    and checkpoint['val_episodes'] == splits['data/grounding_patch_v3_256.h5']['val']['episodes'],
                    f'Patch checkpoint split mismatch: {path}')
    patch_lineage = read_json('data/grounding_patch_v3_256.report.json')
    require(patch_lineage['dataset_sha256'] == patch_data_hash
            and patch_lineage['source_sha256'] == hashes['data/grounding_rgbd_v2_256.h5'],
            'Patch dataset lineage mismatch')
    expert = checked_rollout('artifacts/expert_100.json', list(range(3000, 3100)))
    bc = {
        label: checked_rollout(f'artifacts/bc_kaggle_v2_{report_label}.json', seeds)
        for label, report_label, seeds in (
            ('reused_development', 'same', sets['reused_development']),
            ('fresh_at_bc_v2', 'fresh', sets['fresh_at_bc_v2']),
        )
    }
    bc_comparison = read_json('artifacts/bc_kaggle_v2_comparison.json')
    require(all(read_json(row['report']).get('checkpoint') == 'weights/bc_kaggle_v2/best.pt'
                for row in bc.values()), 'BC rollout checkpoint path mismatch')
    require(bc_comparison['checkpoint_sha256'] == bc_hash and bc_comparison['dataset_sha256'] == drawer_hash,
            'BC comparison input hash mismatch')
    require(all(bc_comparison['reports'][name]['successes'] == bc[label]['successes']
                for name, label in (('same', 'reused_development'), ('fresh', 'fresh_at_bc_v2'))),
            'BC comparison count mismatch')

    cfm = {}
    for label, checkpoint in (
        ('h16', 'weights/cfm_kaggle_v2/best.pt'),
        ('h1', 'weights/cfm_h1_kaggle_v2/best.pt'),
    ):
        path = f'artifacts/cfm_{"h1_" if label == "h1" else ""}kaggle_v2/comparison.json'
        report = read_json(path)
        require(report['benchmark_complete'] and report['checkpoint_sha256'] == hashes[checkpoint],
                f'Incomplete or mismatched CFM comparison: {path}')
        require(report['dataset_sha256'] == drawer_hash and report['baseline_checkpoint_sha256'] == bc_hash,
                f'CFM comparison input mismatch: {path}')
        require({(row['euler_steps'], row['set']) for row in report['cfm']} ==
                {(step, name) for step in (4, 6, 8) for name in ('same', 'fresh')},
                f'Missing CFM setting: {path}')
        rows = []
        for row in report['cfm']:
            raw = f'artifacts/cfm_{"h1_" if label == "h1" else ""}kaggle_v2/euler_{row["euler_steps"]}_{row["set"]}.json'
            checked = checked_rollout(raw, sets['reused_development' if row['set'] == 'same' else 'fresh_at_bc_v2'], hashes[checkpoint])
            require(checked['successes'] == row['successes'], f'CFM summary count mismatch: {raw}')
            rows.append({'euler_steps': row['euler_steps'], 'set': row['set'], **checked})
        cfm[label] = {'summary': path, 'rollouts': rows}

    patch = {}
    for split in ('val', 'test'):
        path = f'artifacts/patch_grounding_v3/selected_{split}.json'
        report = read_json(path)
        require(report['split'] == split and report['checkpoint_sha256'] == patch_hash
                and report['dataset_sha256'] == patch_data_hash,
                f'Patch report input mismatch: {path}')
        require(report['episodes'] == splits['data/grounding_patch_v3_256.h5'][split]['episodes'],
                f'Patch report split mismatch: {path}')
        metric = report['ablations']['none']
        patch[split] = {'report': path, **{key: metric[key] for key in
                       ('targets', 'valid_predictions', 'within_1_5cm_count',
                        'valid_mean_error_m', 'valid_p95_error_m')}}
    visual = checked_rollout('artifacts/patch_grounding_v3/closed_loop_selected.json', list(range(9000, 9005)))
    visual_report = read_json(visual['report'])
    require(visual_report['grounding_sha256'] == patch_hash and visual_report['policy_sha256'] == bc_hash,
            'Visual rollout input mismatch')

    return {
        'schema_version': 1,
        'scope': 'Local baseline freeze; full simulation benchmark outcomes are recorded, not rerun by this check.',
        'files_sha256': hashes,
        'dataset_splits': splits,
        'evaluation_sets': sets,
        'benchmarks': {'expert_drawer': expert, 'bc_v2': bc, 'cfm': cfm,
                       'patch_grounding': patch, 'visual_closed_loop': visual},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write', action='store_true', help='Create/update the frozen manifest')
    args = parser.parse_args()
    current = build_manifest()
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=2, ensure_ascii=False) + '\n')
        print(f'Wrote {MANIFEST.relative_to(ROOT)}')
    else:
        require(MANIFEST.exists(), 'Manifest missing; use --write for the initial freeze')
        require(read_json(str(MANIFEST.relative_to(ROOT))) == current,
                'Frozen manifest differs from local files, report contents, or seed protocol')
        print('Baseline manifest, hashes, reports, and seed protocol verified')


if __name__ == '__main__':
    main()
