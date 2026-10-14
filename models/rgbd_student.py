"""CPU RGB-D semantic student, surface-cloud center regression, finite text encoder.

Only RGB, measured depth, calibration and text enter inference. Simulator masks
and center labels supervise training/evaluation. This is a scene-specific student.
"""
import re

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

QUERIES = ('grasp the bottle body', 'grasp the drawer handle')
BOTTLE_RADIUS_M = .025
TRAIN_PHRASES = (
    ('grasp the bottle body', 'locate the bottle', 'find the bottle body',
     'pick up the bottle', 'where is the bottle', 'point to the bottle',
     'show the bottle body', 'identify the bottle', 'find the green bottle',
     'locate the bottle for grasping'),
    ('grasp the drawer handle', 'locate the drawer handle', 'find the handle',
     'pull the drawer handle', 'where is the handle', 'point to the handle',
     'show the drawer handle', 'identify the handle', 'find the dark handle',
     'locate the handle for grasping'),
)
# Held out at sentence level, using known vocabulary; no zero-shot synonym claim.
HELDOUT_PHRASES = (
    ('identify the green bottle body', 'show the bottle for grasping',
     'where is the green bottle', 'point to the bottle body'),
    ('identify the dark drawer handle', 'show the handle for grasping',
     'where is the dark handle', 'point to the drawer handle'),
)


def words(text):
    return re.findall(r'[a-z]+', text.lower())


class TargetTextEncoder(nn.Module):
    def __init__(self, vocabulary=None):
        super().__init__()
        self.vocabulary = vocabulary or sorted({word for group in TRAIN_PHRASES for text in group for word in words(text)})
        self.indices = {word: i + 1 for i, word in enumerate(self.vocabulary)}
        self.embedding = nn.Embedding(len(self.vocabulary) + 1, 24, padding_idx=0)
        self.classifier = nn.Linear(24, 2)

    def tokenize(self, texts):
        tokens = [words(text) for text in texts]
        if any(not row or any(word not in self.indices for word in row) for row in tokens):
            raise ValueError('Unsupported wording outside the finite training vocabulary')
        ids = torch.zeros(len(tokens), max(map(len, tokens)), dtype=torch.long, device=self.embedding.weight.device)
        for i, row in enumerate(tokens):
            ids[i, :len(row)] = torch.tensor([self.indices[word] for word in row], device=ids.device)
        return ids

    def forward(self, ids):
        return self.classifier(self.embedding(ids).sum(1) / ids.ne(0).sum(1, keepdim=True).clamp_min(1))


class RGBDStudent(nn.Module):
    def __init__(self, width=48, vocabulary=None):
        super().__init__()
        self.segmenter = nn.Sequential(nn.Linear(10, width), nn.SiLU(),
                                       nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
        self.centers = nn.ModuleList([
            nn.Sequential(nn.Linear(27, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(), nn.Linear(32, 3))
            for _ in range(2)
        ])
        self.language = TargetTextEncoder(vocabulary)


def image_features(frame, ablation='none'):
    """Return per-pixel learned inputs and a calibrated measured surface cloud."""
    rgb = np.asarray(frame['rgb'])
    depth = np.asarray(frame['depth'], dtype=np.float32).copy()
    k = np.asarray(frame['K'], dtype=np.float32)
    transform = np.asarray(frame['T'], dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.shape[:2] != depth.shape:
        raise ValueError('RGB/depth shape mismatch')
    if not np.isfinite(k).all() or not np.isfinite(transform).all() or k.shape != (3, 3) or transform.shape != (4, 4):
        raise ValueError('Invalid calibration')
    if abs(np.linalg.det(k)) < 1e-8:
        raise ValueError('Singular calibration')
    if ablation not in ('none', 'drop-rgb', 'drop-depth', 'drop-rgbd', 'swap-text'):
        raise ValueError(ablation)
    if ablation in ('drop-depth', 'drop-rgbd'):
        depth[:] = 0
    # Reject far-plane/background values before unprojection. P5 consumes a
    # surface cloud, so there is no single peak pixel to trust for depth.
    depth_max = float(frame.get('depth_max_m', 5.0))
    valid = np.isfinite(depth) & (depth > .05) & (depth < depth_max)
    robot_mask = frame.get('robot_mask')
    if robot_mask is not None:
        robot_mask = np.asarray(robot_mask, dtype=bool)
        if robot_mask.shape != depth.shape:
            raise ValueError('Robot occlusion mask shape differs from depth image')
        valid &= ~robot_mask
    depth[~valid] = 0
    h, w = depth.shape
    v, u = np.mgrid[:h, :w]
    pixels = np.stack((u, v, np.ones_like(u)), -1).astype(np.float32)
    rays = pixels @ np.linalg.inv(k).T
    points = (rays * depth[:, :, None]) @ transform[:3, :3].T + transform[:3, 3]
    color = torch.from_numpy(rgb.copy()).float() / 255
    if ablation in ('drop-rgb', 'drop-rgbd'):
        color.zero_()
    local = F.avg_pool2d(color.permute(2, 0, 1)[None], 3, 1, 1)[0].permute(1, 2, 0)
    xyz = torch.from_numpy((points - np.array([.4, 0, .8], np.float32)).clip(-2, 2))
    features = torch.cat((color, torch.from_numpy(depth[:, :, None]) / 3, xyz, local), -1)
    return features.reshape(-1, 10), points, valid


def largest_component(mask):
    """Eight-connected foreground; reject isolated semantic false positives."""
    remaining = set(zip(*np.nonzero(mask)))
    best = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        component, pending = [start], [start]
        while pending:
            y, x = pending.pop()
            for dy, dx in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                neighbor = y + dy, x + dx
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    pending.append(neighbor)
                    component.append(neighbor)
        if len(component) > len(best):
            best = component
    output = np.zeros_like(mask)
    if best:
        ys, xs = zip(*best)
        output[ys, xs] = True
    return output


def cloud_features(points, camera_position):
    """Translation-invariant shape statistics and viewing direction, in meters."""
    quantiles = np.quantile(points, [.1, .25, .5, .75, .9], axis=0).astype(np.float32)
    origin = quantiles[2]
    centered = points - origin
    covariance = centered.T @ centered / len(points)
    direction = origin - camera_position
    direction /= max(float(np.linalg.norm(direction)), 1e-8)
    features = np.r_[(quantiles - origin).flatten() / .1,
                     points.std(0) / .1, covariance[np.triu_indices(3)] / .01, direction]
    return features.astype(np.float32), origin


def refine_bottle_height(point, origin, axis_z_alignment):
    """Use the visible upper cylinder surface when the bottle lies sideways.

    The fixed scene's cylinder radius is known at inference. The median surface
    is roughly 0.68 radii above its center in the near-horizontal view. This
    branch was selected on validation; no pose, phase, or target label is read.
    """
    if axis_z_alignment >= .4:
        return point
    revised = point.copy()
    revised[2] = origin[2] - .68 * BOTTLE_RADIUS_M
    return revised


def semantic_clouds(model, frame, threshold=.7, ablation='none'):
    features, points, valid = image_features(frame, ablation)
    device = next(model.parameters()).device
    probability = model.segmenter(features.to(device)).softmax(-1).detach().cpu().numpy().reshape(*valid.shape, 3)
    workspace = ((points >= [-.1, -.75, .4]) & (points <= [.95, .75, 1.5])).all(-1)
    outputs = []
    for target in range(2):
        mask = largest_component((probability[:, :, target + 1] >= threshold) & valid & workspace)
        count = int(mask.sum())
        row = {'mask': mask, 'support': count, 'confidence': float(probability[:, :, target + 1][mask].mean()) if count else 0.}
        if count >= 6:
            cloud = points[mask]
            row['features'], row['origin'] = cloud_features(cloud, frame['T'][:3, 3])
            if target == 0:
                axis = np.linalg.eigh(np.cov(cloud.T))[1][:, -1]
                row['axis_z_alignment'] = float(abs(axis[2]))
        outputs.append(row)
    return outputs


class RGBDStudentRuntime:
    def __init__(self, checkpoint):
        self.checkpoint = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if self.checkpoint['schema_version'] != 'omni-p5-student-v1':
            raise ValueError('Unsupported student schema')
        self.model = RGBDStudent(**self.checkpoint['model_config'])
        self.model.load_state_dict(self.checkpoint['model'])
        self.model.eval()
        self.image_size = self.checkpoint['image_size']
        self.last_valid_prediction = {}
        self.consecutive_failures = 0
        self.max_fallback_steps = 15
        self.max_fallback_age_s = .4
        self.temporal_fallback_enabled = True
        self.temporal_smoothing_enabled = True
        self.failure_reasons = {}
        self.fallback_count = 0

    def reset_episode(self):
        """Clear temporal state so landmarks can never leak between episodes."""
        self.last_valid_prediction = {}
        self.consecutive_failures = 0

    def _invalid_reason(self, rows):
        if any(row['support'] < 6 for row in rows):
            return 'insufficient_support'
        if any(row['text_confidence'] < .8 for row in rows):
            return 'text_confidence'
        return 'invalid_sample_or_workspace'

    @torch.inference_mode()
    def predict_detailed(self, frame, queries=QUERIES, ablation='none'):
        if np.asarray(frame['rgb']).shape != (self.image_size, self.image_size, 3):
            raise ValueError('Image size differs from training camera contract')
        queries = queries[::-1] if ablation == 'swap-text' else queries
        language = self.model.language(self.model.language.tokenize(queries)).softmax(-1)
        confidence, ids = language.max(-1)
        clouds = semantic_clouds(self.model, frame, self.checkpoint['threshold'], ablation)
        outputs = []
        for target, text_confidence in zip(ids.tolist(), confidence.tolist()):
            cloud = clouds[target]
            point = None
            if 'features' in cloud and text_confidence >= .8:
                offset = self.model.centers[target](torch.from_numpy(cloud['features'])[None])[0].numpy() * .1
                point = cloud['origin'] + offset
                if target == 0:
                    point = refine_bottle_height(point, cloud['origin'], cloud['axis_z_alignment'])
            valid = point is not None and np.isfinite(point).all() and ((point >= [-.1, -.75, .4]) & (point <= [.95, .75, 1.5])).all()
            outputs.append({**cloud, 'point': point, 'valid': bool(valid), 'text_confidence': text_confidence,
                            'semantic_target': target})
        return outputs

    def predict(self, frame, ablation='none'):
        # Keep the visual runner's historical RGB-only ablation spelling.
        rows = self.predict_detailed(frame, ablation='drop-rgb' if ablation == 'drop-image' else ablation)
        valid = all(row['valid'] for row in rows)
        fallback_used = False
        if valid:
            points = np.stack([row['point'] for row in rows])
            previous = self.last_valid_prediction.get('points')
            if previous is not None and self.temporal_smoothing_enabled:
                points = .7 * points + .3 * previous
            self.last_valid_prediction = {
                'points': points.copy(),
                'sim_time': float(frame.get('sim_time', 0.)),
            }
            self.consecutive_failures = 0
        else:
            reason = self._invalid_reason(rows)
            self.failure_reasons[reason] = self.failure_reasons.get(reason, 0) + 1
            if not self.temporal_fallback_enabled:
                raise ValueError(f'Invalid RGB-D student prediction: {reason}; no fallback')
            cached = self.last_valid_prediction
            self.consecutive_failures += 1
            age = float(frame.get('sim_time', 0.)) - cached.get('sim_time', 0.) if cached else float('inf')
            if (not cached or self.consecutive_failures > self.max_fallback_steps
                    or age < 0 or age > self.max_fallback_age_s):
                raise ValueError(f'Invalid RGB-D student prediction: {reason}; fallback unavailable (age={age:.3f}s, streak={self.consecutive_failures})')
            points = cached['points'].copy()
            fallback_used = True
            self.fallback_count += 1
        camera = (points - frame['T'][:3, 3]) @ frame['T'][:3, :3]
        pixels = camera @ frame['K'].T
        return points, {'pixels': (pixels[:, :2] / pixels[:, 2:]).tolist(),
                        'center_depth_m': camera[:, 2].tolist(),
                        'support': [row['support'] for row in rows],
                        'confidence': [row['confidence'] for row in rows],
                        'fallback_used': fallback_used,
                        'fallback_age_s': max(0., float(frame.get('sim_time', 0.)) - self.last_valid_prediction['sim_time']),
                        'fallback_streak': self.consecutive_failures,
                        'invalid_reason': None if valid else reason}
