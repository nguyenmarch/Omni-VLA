import json

import h5py
import numpy as np
import pytest
import torch

from models.rgbd_student import (RGBDStudent, TargetTextEncoder, TRAIN_PHRASES, HELDOUT_PHRASES,
                                image_features, cloud_features, semantic_clouds, largest_component,
                                refine_bottle_height)
from scripts.eval_p5 import summarize
from utils.checkpoint import file_hash
from utils.p5_dataset import P5Frames


def test_calibrated_cloud_and_missing_depth_rejection():
    torch.set_num_threads(2)
    t = np.eye(4, dtype=np.float32)
    t[:3, 3] = [.2, -.1, .5]
    frame = {'rgb': np.zeros((4, 4, 3), np.uint8), 'depth': np.ones((4, 4), np.float32) * .2,
             'K': np.array([[10, 0, 2], [0, 10, 2], [0, 0, 1]], np.float32), 'T': t}
    features, points, valid = image_features(frame)
    np.testing.assert_allclose(points[2, 2], [.2, -.1, .7], atol=1e-7)
    assert features.shape == (16, 10) and valid.all()
    model = RGBDStudent()
    rows = semantic_clouds(model, frame, threshold=.01, ablation='drop-depth')
    assert all(row['support'] == 0 and 'features' not in row for row in rows)
    frame['K'][:] = 0
    with pytest.raises(ValueError, match='Singular'):
        image_features(frame)


def test_cloud_shape_features_are_translation_invariant():
    cloud = np.random.default_rng(12).normal(size=(50, 3)).astype(np.float32) * .02
    camera = np.array([1, 1, 1], np.float32)
    shift = np.array([.5, -.3, .8], np.float32)
    features, origin = cloud_features(cloud, camera)
    translated, new_origin = cloud_features(cloud + shift, camera + shift)
    np.testing.assert_allclose(features, translated, atol=2e-6)
    np.testing.assert_allclose(origin + shift, new_origin, atol=1e-7)
    assert len(features) == 27


def test_language_holdout_and_unknown_words():
    train = {p for group in TRAIN_PHRASES for p in group}
    test = {p for group in HELDOUT_PHRASES for p in group}
    assert train.isdisjoint(test)
    encoder = TargetTextEncoder()
    assert encoder.tokenize(sorted(test)).shape[0] == len(test)
    with pytest.raises(ValueError, match='Unsupported'):
        encoder.tokenize(['grasp the banana'])


def test_component_and_metrics_do_not_hide_rejections():
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:4, 1:4] = True
    mask[7, 7] = True
    largest = largest_component(mask)
    assert largest.sum() == 9 and not largest[7, 7]
    result = summarize([{'valid': True, 'error_m': .01}, {'valid': False, 'error_m': None}])
    assert result['coverage'] == .5 and result['within_1_5cm_all_fraction'] == .5
    assert summarize([{'valid': False, 'error_m': None}])['mean_error_m'] is None


def test_horizontal_cylinder_refinement_uses_measured_surface():
    point = np.array([.4, .1, .41], np.float32)
    origin = np.array([.42, .12, .475], np.float32)
    np.testing.assert_allclose(refine_bottle_height(point, origin, .1), [.4, .1, .458], atol=1e-6)
    np.testing.assert_array_equal(refine_bottle_height(point, origin, .9), point)


def test_teacher_source_and_episode_seed_contract(tmp_path):
    source, labels = tmp_path / 'source.h5', tmp_path / 'labels.h5'
    splits = json.dumps({'train': ['a'], 'val': ['b'], 'test': ['c']})
    with h5py.File(source, 'w') as file:
        file.attrs['splits'] = splits
        file.attrs['image_size'] = 4
        for name, seed in [('a', 1), ('b', 1), ('c', 3)]:
            episode = file.create_group('episodes/' + name)
            episode.attrs['seed'] = seed
            episode.create_dataset('rgb', data=np.zeros((1, 4, 4, 3), np.uint8))
    with h5py.File(labels, 'w') as file:
        file.attrs['schema_version'] = 'omni-p5-labels-v1'
        file.attrs['source_sha256'] = file_hash(source)
        file.attrs['splits'] = splits
    with pytest.raises(ValueError, match='Seed leakage'):
        P5Frames(source, labels, 'test')
    with h5py.File(source, 'a') as file:
        file['episodes/b'].attrs['seed'] = 2
    with pytest.raises(ValueError, match='hash mismatch'):
        P5Frames(source, labels, 'test')
