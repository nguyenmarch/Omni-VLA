"""Episode-disjoint temporal RGB-D action windows with train-only robot stats."""
import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from models.direct_vla import measured_image, proprioception, tokenize, TASK_PHRASES

SCHEMA = 'omni-direct-vla-v1'


class VLADataset(Dataset):
    def __init__(self, path, split, history=8, stats=None):
        if history < 1 or split not in ('train', 'val', 'test'):
            raise ValueError('Positive history and documented split required')
        self.path, self.history, self.split = str(path), history, split
        with h5py.File(path) as file:
            if file.attrs['schema'] != SCHEMA:
                raise ValueError('Wrong VLA dataset schema')
            self.splits = json.loads(file.attrs['splits'])
            names = [n for part in self.splits.values() for n in part]
            seeds = [int(file['episodes'][n].attrs['seed']) for n in names]
            if len(set(names)) != len(names) or len(set(seeds)) != len(seeds):
                raise ValueError('Episode/seed leakage')
            self.size, self.chunk = int(file.attrs['image_size']), int(file.attrs['chunk_size'])
            self.action_dt = float(file.attrs['action_dt'])
            self.index = [(name, i) for name in self.splits[split] for i in range(len(file['episodes'][name]['rgb']))]
            if stats is None:
                values = np.concatenate([file['episodes'][name]['robot'][:] for name in self.splits['train']])
                stats = {'mean': values.mean(0).astype(np.float32), 'std': np.maximum(values.std(0), .05).astype(np.float32)}
        self.stats = stats

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        name, step = self.index[index]
        with h5py.File(self.path) as file:
            episode = file['episodes'][name]
            start = max(0, step-self.history+1)
            length = step-start+1
            image = np.zeros((self.history, 7, self.size, self.size), np.float32)
            robot = np.zeros((self.history, 50), np.float32)
            for j, i in enumerate(range(start, step+1)):
                image[j] = measured_image({key: episode[key][i] for key in ('rgb', 'depth', 'K', 'T')})
            robot[:length] = (episode['robot'][start:step+1]-self.stats['mean'])/self.stats['std']
            task = int(episode.attrs['task'])
            phrases = TASK_PHRASES[task]
            text = phrases[(step+index) % len(phrases)] if self.split == 'train' else phrases[0]
            return {'images': torch.from_numpy(image), 'robot': torch.from_numpy(robot), 'tokens': tokenize([text])[0],
                    'history_mask': torch.arange(self.history) < length,
                    'action': torch.from_numpy(episode['action'][step]), 'action_mask': torch.from_numpy(episode['action_mask'][step]),
                    'phase': torch.tensor(int(episode['phase'][step])),
                    'landmarks': torch.from_numpy(episode['landmarks'][step])}
