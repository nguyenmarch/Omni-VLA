"""Read immutable RGB-D frames with separately replay-certified teacher labels."""
import json
import h5py
import numpy as np
from utils.checkpoint import file_hash


class P5Frames:
    def __init__(self, source, labels, split):
        if split not in ('train', 'val', 'test'):
            raise ValueError(split)
        self.source, self.labels, self.split = str(source), str(labels), split
        with h5py.File(source) as src, h5py.File(labels) as teacher:
            if teacher.attrs['schema_version'] != 'omni-p5-labels-v1':
                raise ValueError('Wrong label schema')
            if teacher.attrs['source_sha256'] != file_hash(source):
                raise ValueError('Teacher/source hash mismatch')
            splits = json.loads(src.attrs['splits'])
            if splits != json.loads(teacher.attrs['splits']):
                raise ValueError('Teacher split mismatch')
            names = [name for group in splits.values() for name in group]
            if len(names) != len(set(names)):
                raise ValueError('Episode leakage')
            seeds = [int(src['episodes'][name].attrs['seed']) for name in names]
            if len(seeds) != len(set(seeds)):
                raise ValueError('Seed leakage')
            self.episodes = splits[split]
            self.size = int(src.attrs['image_size'])
            self.count = sum(len(src['episodes'][name]['rgb']) for name in self.episodes)

    def __iter__(self):
        with h5py.File(self.source) as src, h5py.File(self.labels) as labels:
            for name in self.episodes:
                episode, teacher = src['episodes'][name], labels['episodes'][name]
                if len(episode['rgb']) != len(teacher['semantic']):
                    raise ValueError('Frame count mismatch')
                for i in range(len(episode['rgb'])):
                    frame = {key: episode[key][i] for key in ('rgb', 'depth', 'K', 'T')}
                    count, amodal = teacher['visible_pixels'][i], teacher['amodal_pixels'][i]
                    ratio = count / np.maximum(amodal, 1)
                    yield frame, {'points': episode['targets'][i], 'semantic': teacher['semantic'][i],
                                  'visible': (count >= 6) & (ratio >= .25), 'visible_fraction': ratio,
                                  'visible_pixels': count, 'episode': name, 'frame': i,
                                  'seed': int(episode.attrs['seed']), 'phase': int(episode['phase'][i])}
