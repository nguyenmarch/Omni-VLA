"""Replay saved RGB-D episodes to obtain segmentation/occlusion teacher labels.

Privileged geom IDs are used only for supervision and evaluation, never runtime.
The original images, calibration, actions and episode split must replay exactly.
"""
import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from simulation.bimanual_env import ROOT
from simulation.expert import ScriptedExpert
from simulation.vision import capture_head
from utils.checkpoint import file_hash
from utils.config import make_environment


def prepare(source, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp')
    if temporary.exists():
        raise FileExistsError(temporary)
    env = None
    rows = []
    try:
        with h5py.File(source) as src, h5py.File(temporary, 'w') as dest:
            if src.attrs['schema_version'] != 'omni-grounding-rgbd-v2':
                raise ValueError('Require raw RGB-D v2 data')
            if src.attrs['scene_sha256'] != file_hash(ROOT / 'simulation/assets/bottle_drawer_scene.xml'):
                raise ValueError('Scene changed: recollect or explicitly audit migration')
            env = make_environment(image_size=int(src.attrs['image_size']))
            geoms = [env.model.geom(name).id for name in ('bottle_geom', 'handle')]
            groups = env.model.geom_group.copy()
            option = mujoco.MjvOption()
            option.sitegroup[:] = 0
            option.geomgroup[:] = [0, 0, 0, 0, 1, 0]
            dest.attrs.update(schema_version='omni-p5-labels-v1', source_sha256=file_hash(source),
                              splits=src.attrs['splits'], image_size=src.attrs['image_size'],
                              target_semantics='bottle_site and handle_site centers; compatible with frozen BC',
                              visibility='visible pixels >=6 and visible/amodal >=0.25; segmentation teacher only')
            for name, episode in src['episodes'].items():
                obs, _ = env.reset(seed=int(episode.attrs['seed']))
                expert = ScriptedExpert(env)
                steps = {int(step): i for i, step in enumerate(episode['step'][:])}
                out = dest.create_group('episodes/' + name)
                masks, amodal, visible = [], [], []
                max_error = 0.
                for step in range(max(steps) + 1):
                    if step in steps:
                        index = steps[step]
                        comparisons = [(obs['q'], episode['q'][index]),
                                       (obs['object'][:6].reshape(2, 3), episode['targets'][index]),
                                       (expert.action(), episode['action'][index])]
                        error = max(float(np.max(np.abs(a - b))) for a, b in comparisons)
                        max_error = max(max_error, error)
                        if error > 1e-6 or not np.isfinite(error):
                            raise ValueError(f'Replay mismatch {name}:{step}: {error}')
                        frame = capture_head(env)
                        for key in ('K', 'T', 'depth'):
                            if not np.allclose(frame[key], episode[key][index], atol=1e-6, rtol=0):
                                raise ValueError(f'Camera replay mismatch {name}:{step}:{key}')
                        if not np.array_equal(frame['rgb'], episode['rgb'][index]):
                            raise ValueError(f'RGB replay mismatch {name}:{step}')
                        renderer = env._renderer()
                        renderer.enable_segmentation_rendering()
                        renderer.update_scene(env.data, camera='head_cam')
                        seg = renderer.render().copy()
                        semantic = np.zeros(seg.shape[:2], np.uint8)
                        for target, geom in enumerate(geoms, 1):
                            semantic[(seg[:, :, 0] == geom) & (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))] = target
                        counts = []
                        try:
                            for geom in geoms:
                                env.model.geom_group[:] = 5
                                env.model.geom_group[geom] = 4
                                renderer.update_scene(env.data, camera='head_cam', scene_option=option)
                                isolated = renderer.render().copy()
                                counts.append(int(((isolated[:, :, 0] == geom) &
                                                   (isolated[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM))).sum()))
                        finally:
                            env.model.geom_group[:] = groups
                            renderer.disable_segmentation_rendering()
                        masks.append(semantic)
                        visible.append([(semantic == target).sum() for target in (1, 2)])
                        amodal.append(counts)
                    if step < max(steps):
                        obs, _, _, _, _ = env.step(expert.action())
                        expert.update()
                out.create_dataset('semantic', data=np.asarray(masks), compression='gzip')
                out.create_dataset('visible_pixels', data=np.asarray(visible, np.int32))
                out.create_dataset('amodal_pixels', data=np.asarray(amodal, np.int32))
                out.attrs['seed'] = int(episode.attrs['seed'])
                row = {'episode': name, 'frames': len(masks), 'max_replay_error': max_error}
                rows.append(row)
                print(json.dumps(row), flush=True)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        if env is not None:
            env.close()
    report = {'source_sha256': file_hash(source), 'labels_sha256': file_hash(output), 'episodes': rows}
    output.with_suffix('.report.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--output', default='data/grounding_p5_labels.h5')
    args = p.parse_args()
    prepare(args.source, args.output)
