"""Train a RGB-D segmentation student and center regressor using validation only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from models.rgbd_student import RGBDStudent, TRAIN_PHRASES, image_features, semantic_clouds
from utils.p5_dataset import P5Frames
from utils.checkpoint import file_hash, atomic_save


def sample_pixels(dataset, rng):
    features, labels = [], []
    for frame, truth in dataset:
        x, _, valid = image_features(frame)
        semantic = truth['semantic'].reshape(-1)
        for target, limit in ((0, 96), (1, 32), (2, 32)):
            indices = np.flatnonzero((semantic == target) & valid.reshape(-1))
            if len(indices):
                indices = rng.choice(indices, min(limit, len(indices)), replace=False)
                features.append(x[indices])
                labels.append(torch.from_numpy(semantic[indices].astype(np.int64)))
    if not features:
        raise ValueError('No valid pixel samples')
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def center_samples(model, dataset, threshold):
    samples = [[] for _ in range(2)]
    totals = [0, 0]
    for frame, truth in dataset:
        clouds = semantic_clouds(model, frame, threshold)
        for target, cloud in enumerate(clouds):
            totals[target] += 1
            if 'features' in cloud:
                samples[target].append((cloud['features'], (truth['points'][target] - cloud['origin']) / .1,
                                        bool(truth['visible'][target])))
    tensors = []
    for rows in samples:
        if not rows:
            raise ValueError('Segmenter provides no foreground samples')
        tensors.append((torch.from_numpy(np.stack([r[0] for r in rows])),
                        torch.from_numpy(np.stack([r[1] for r in rows])),
                        torch.tensor([r[2] for r in rows], dtype=torch.bool)))
    return tensors, sum(totals)


def train(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Use a new training directory')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train_data, val_data = [P5Frames(args.source, args.labels, split) for split in ('train', 'val')]
    model = RGBDStudent()
    x, y = sample_pixels(train_data, rng)
    vx, vy = sample_pixels(val_data, np.random.default_rng(0))
    output.mkdir(parents=True)
    print(json.dumps({'train_pixels': len(y), 'val_pixels': len(vy), 'train_frames': train_data.count}), flush=True)
    optimizer = torch.optim.AdamW(model.segmenter.parameters(), lr=.003, weight_decay=.0001)
    best_iou, best_segmenter = -1, None
    for epoch in range(args.segmentation_epochs):
        order = torch.randperm(len(y))
        model.train()
        for indices in order.split(8192):
            batch = x[indices].clone()
            gain = .85 + .3 * torch.rand(len(indices), 1)
            batch[:, :3] = (batch[:, :3] * gain).clamp(0, 1)
            batch[:, 7:] = (batch[:, 7:] * gain).clamp(0, 1)
            batch[:, 4:7] += torch.randn_like(batch[:, 4:7]) * .01
            loss = F.cross_entropy(model.segmenter(batch), y[indices])
            optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            prediction = model.segmenter(vx).argmax(-1)
            ious = [float(((prediction == k) & (vy == k)).sum() / ((prediction == k) | (vy == k)).sum().clamp_min(1)) for k in (1, 2)]
        score = float(np.mean(ious))
        if score > best_iou:
            best_iou = score
            best_segmenter = {k: v.clone() for k, v in model.segmenter.state_dict().items()}
        if (epoch + 1) % 5 == 0:
            print(json.dumps({'segmentation_epoch': epoch + 1, 'sampled_val_iou': ious}), flush=True)
    model.segmenter.load_state_dict(best_segmenter)
    texts = [text for group in TRAIN_PHRASES for text in group]
    labels = torch.tensor([i for i, group in enumerate(TRAIN_PHRASES) for _ in group])
    ids = model.language.tokenize(texts)
    optimizer = torch.optim.AdamW(model.language.parameters(), lr=.03)
    for _ in range(200):
        loss = F.cross_entropy(model.language(ids), labels)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    model.eval()
    print('Preparing predicted-surface center regression samples', flush=True)
    samples, _ = center_samples(model, train_data, args.threshold)
    validation, total = center_samples(model, val_data, args.threshold)
    optimizer = torch.optim.AdamW(model.centers.parameters(), lr=.002, weight_decay=.0001)
    best, log = None, []
    source_hash, labels_hash = file_hash(args.source), file_hash(args.labels)
    for epoch in range(args.center_epochs):
        model.train()
        loss = sum(F.smooth_l1_loss(model.centers[i](features), target)
                   for i, (features, target, _) in enumerate(samples))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        model.eval()
        with torch.no_grad():
            errors = torch.cat([torch.linalg.vector_norm(model.centers[i](features) - target, dim=1) * .1
                                for i, (features, target, _) in enumerate(validation)])
            visible = torch.cat([row[2] for row in validation])
        row = {'epoch': epoch + 1, 'loss': float(loss.detach()),
               'val_accepted': len(errors), 'val_targets': total,
               'val_accurate': int((errors <= .015).sum()),
               'val_visible_mean_m': float(errors[visible].mean()),
               'val_accepted_mean_m': float(errors.mean())}
        score = (-row['val_accurate'], row['val_accepted_mean_m'])
        payload = {'schema_version': 'omni-p5-student-v1', 'model_config': {'width': 48, 'vocabulary': model.language.vocabulary},
                   'model': model.state_dict(), 'image_size': train_data.size, 'threshold': args.threshold,
                   'source_sha256': source_hash, 'labels_sha256': labels_hash,
                   'train_episodes': train_data.episodes, 'val_episodes': val_data.episodes,
                   'training': vars(args), 'selection': 'validation accurate count, then accepted mean error',
                   'metric': row, 'train_phrases': TRAIN_PHRASES, 'sampled_val_segmentation_iou': best_iou}
        if best is None or score < best:
            best = score
            atomic_save(payload, output / 'best.pt')
        log.append(row)
        if (epoch + 1) % 50 == 0:
            print(json.dumps(row), flush=True)
    atomic_save(payload, output / 'last.pt')
    (output / 'metrics.json').write_text(json.dumps(log, indent=2) + '\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--labels', default='data/grounding_p5_labels.h5')
    p.add_argument('--output', default='weights/p5_student')
    p.add_argument('--segmentation-epochs', type=int, default=40)
    p.add_argument('--center-epochs', type=int, default=400)
    p.add_argument('--threshold', type=float, default=.7)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threads', type=int, default=2)
    args = p.parse_args()
    if min(args.segmentation_epochs, args.center_epochs, args.threads) < 1 or not 0 < args.threshold < 1:
        p.error('Invalid epochs/threads/threshold')
    train(args)
