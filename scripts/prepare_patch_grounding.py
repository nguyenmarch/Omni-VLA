"""Version supervision using the identical patch decoder used at inference."""

import argparse
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch

from utils.camera import project
from utils.checkpoint import file_hash
from utils.patch_grounding_dataset import DEFAULT_DECODER_CONFIG, PATCH_OFFSET_LIMIT_M
from utils.robust_depth import RobustDepthDecoder


LABEL_FIELDS = (
    'patch_uv', 'patch_surface', 'patch_offset', 'patch_label_valid',
    'patch_support', 'patch_iqr',
)


def frame_labels(episode, index, decoder):
    """Project both targets, then apply the inference patch contract at GT pixels."""
    height, width = episode['rgb'].shape[1:3]
    if height != width or episode['depth'].shape[1:] != (height, width):
        raise ValueError('Expected square RGB and matching depth frames')

    intrinsic = episode['K'][index]
    transform = episode['T'][index]
    projections = [project(point, intrinsic, transform) for point in episode['targets'][index]]
    uv = np.stack([pixel for pixel, _ in projections])
    target_depth = torch.tensor([depth for _, depth in projections], dtype=torch.float32)
    in_view = (np.isfinite(uv).all(axis=1)
               & (uv >= 0).all(axis=1)
               & (uv < width).all(axis=1)
               & (target_depth.numpy() > 0))

    heatmap = torch.full((2, height, width), -torch.inf)
    for target in range(2):
        if in_view[target]:
            u, v = np.rint(uv[target]).astype(int).clip(0, width - 1)
            heatmap[target, v, u] = 1

    depth = torch.from_numpy(episode['depth'][index])[None].expand(2, -1, -1)
    k = torch.from_numpy(intrinsic)[None].expand(2, -1, -1)
    t = torch.from_numpy(transform)[None].expand(2, -1, -1)
    decoded = decoder(heatmap, depth, k, t)
    offset = target_depth - decoded['surface_depth']
    corrected = decoder(heatmap, depth, k, t, offset)
    label_valid = (corrected['valid'] & torch.isfinite(offset)
                   & (offset.abs() <= PATCH_OFFSET_LIMIT_M)
                   & torch.from_numpy(in_view))

    return {
        'patch_uv': uv.astype(np.float32),
        'patch_surface': decoded['surface_depth'].numpy(),
        'patch_offset': offset.numpy(),
        'patch_label_valid': label_valid.numpy(),
        'patch_support': decoded['support'].numpy(),
        'patch_iqr': decoded['iqr'].numpy(),
    }


def prepare(args):
    torch.set_num_threads(2)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)

    config = {**DEFAULT_DECODER_CONFIG, 'quantile': args.quantile}
    decoder = RobustDepthDecoder(**config)
    source_hash = file_hash(args.source)
    shutil.copyfile(args.source, temporary)
    report = {}
    total = eligible = 0
    try:
        with h5py.File(temporary, 'a') as dataset:
            if dataset.attrs['schema_version'] != 'omni-grounding-rgbd-v2':
                raise ValueError('Require raw v2 RGB-D dataset')
            dataset.attrs.update(
                schema_version='omni-grounding-patch-v3',
                source_sha256=source_hash,
                decoder_config=json.dumps(config),
                offset_limit_m=PATCH_OFFSET_LIMIT_M,
            )
            for name, episode in dataset['episodes'].items():
                records = {field: [] for field in LABEL_FIELDS}
                for index in range(len(episode['rgb'])):
                    labels = frame_labels(episode, index, decoder)
                    for field in LABEL_FIELDS:
                        records[field].append(labels[field])
                for field, values in records.items():
                    episode.create_dataset(field, data=np.asarray(values), compression='gzip')
                count = int(np.sum(records['patch_label_valid']))
                targets = 2 * len(episode['rgb'])
                report[name] = {'targets': targets, 'eligible': count, 'coverage': count / targets}
                total += targets
                eligible += count
            dataset.attrs['supervision_report'] = json.dumps(report)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    result = {
        'source_sha256': source_hash,
        'dataset_sha256': file_hash(output),
        'decoder_config': config,
        'targets': total,
        'eligible': eligible,
        'coverage': eligible / total,
        'episodes': report,
    }
    output.with_suffix('.report.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'episodes'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='data/grounding_rgbd_v2_256.h5')
    parser.add_argument('--output', default='data/grounding_patch_v3_256.h5')
    parser.add_argument('--quantile', type=float, default=0.5)
    arguments = parser.parse_args()
    if not np.isfinite(arguments.quantile) or not 0 <= arguments.quantile <= 1:
        parser.error('Invalid quantile')
    prepare(arguments)
