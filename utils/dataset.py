"""Episode-disjoint action chunks with train-only normalization and tail masks."""
import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

SCHEMA = 'omni-state-v1'


def phase_sample_weights(dataset, min_count=1):
    """Balance phases within this split; cap upweighting of tiny phase samples."""
    if min_count < 1: raise ValueError('min_count must be positive')
    with h5py.File(dataset.path) as h:
        phases = {name: h[f'episodes/{name}/phase'][:] for name in dataset.episodes}
    labels = np.array([phases[name][step] for name, step in dataset.index], dtype=np.int64)
    counts = np.bincount(labels)
    return torch.as_tensor(1. / np.maximum(counts[labels], min_count), dtype=torch.double), counts.tolist()


def fit_stats(path, episode_names):
    if not episode_names:
        raise ValueError('Training split is empty')
    accum = {key: [0, None, None] for key in ('condition', 'action')}
    with h5py.File(path, 'r') as h5:
        for name in episode_names:
            group = h5[f'episodes/{name}']
            if not group.attrs.get('labels_valid',group.attrs.get('success',False)):
                continue
            for key in accum:
                # State/actions only, never RGB dataset; bounded memory per episode.
                values = group[key][:].astype(np.float64)
                count, total, squares = accum[key]
                accum[key] = [count+len(values), values.sum(0) if total is None else total+values.sum(0),
                              (values**2).sum(0) if squares is None else squares+(values**2).sum(0)]
    stats = {}
    for key, (count, total, squares) in accum.items():
        if count == 0:
            raise ValueError('No valid training labels')
        mean = total/count
        std = np.sqrt(np.maximum(squares/count-mean**2, 0))
        stats[key+'_mean'] = mean.astype(np.float32)
        stats[key+'_std'] = np.maximum(std, .05).astype(np.float32)
    return stats


class BimanualFlowDataset(Dataset):
    def __init__(self, h5_path, chunk_size=16, split='train', stats=None, cache_state=False):
        self.path = str(h5_path)
        self.chunk_size = chunk_size
        self._h5 = None
        self._pid = None
        self.cache = {}
        with h5py.File(self.path, 'r') as h5:
            if h5.attrs.get('schema_version') != SCHEMA:
                raise ValueError('Unsupported dataset schema')
            self.splits = json.loads(h5.attrs['splits'])
            flat = [name for names in self.splits.values() for name in names]
            if len(flat) != len(set(flat)):
                raise ValueError('Episode leakage across splits')
            self.episodes = self.splits[split]
            self.index = [(name, t) for name in self.episodes if h5[f'episodes/{name}'].attrs.get('labels_valid', h5[f'episodes/{name}'].attrs['success'])
                          for t in range(len(h5[f'episodes/{name}/action']))]
        if cache_state:
            with h5py.File(self.path,'r') as h5:
                for name in self.episodes:
                    group=h5[f'episodes/{name}']
                    self.cache[name]={key:group[key][:] for key in ('condition','action')}
        self.stats = stats if stats is not None else fit_stats(self.path, self.splits['train'])

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        if not self.cache and self._pid != os.getpid():
            self.close()
            self._h5 = h5py.File(self.path, 'r')
            self._pid = os.getpid()
        name, t = self.index[idx]
        group = self.cache[name] if self.cache else self._h5[f'episodes/{name}']
        action = group['action'][t:t+self.chunk_size]
        length = len(action)
        padded = np.zeros((self.chunk_size, action.shape[1]), np.float32)
        padded[:length] = (action-self.stats['action_mean'])/self.stats['action_std']
        mask = np.zeros((self.chunk_size, 1), np.float32)
        mask[:length] = 1
        condition = (group['condition'][t]-self.stats['condition_mean'])/self.stats['condition_std']
        return {'condition': torch.from_numpy(condition.astype(np.float32)), 'action': torch.from_numpy(padded), 'mask': torch.from_numpy(mask)}

    def close(self):
        if self._h5 is not None:
            self._h5.close()
            self._h5 = None
        self._pid = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = state['_pid'] = None
        return state
