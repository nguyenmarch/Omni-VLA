"""P5 all-target/visible/occluded metrics, held-out language and image ablations."""
import argparse
import base64
import html
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.rgbd_student import RGBDStudentRuntime, QUERIES, HELDOUT_PHRASES
from models.grounding3d import GroundingRuntime
from utils.camera import project, unproject
from utils.camera_diagnostics import png_bytes, overlay
from utils.p5_dataset import P5Frames
from utils.checkpoint import file_hash


def summarize(rows):
    accepted = [r for r in rows if r['valid']]
    errors = [r['error_m'] for r in accepted]
    return {'targets': len(rows), 'accepted': len(accepted),
            'coverage': len(accepted) / len(rows) if rows else None,
            'mean_error_m': float(np.mean(errors)) if errors else None,
            'p95_error_m': float(np.percentile(errors, 95)) if errors else None,
            'within_1_5cm_count': sum(r['error_m'] <= .015 for r in accepted),
            'within_1_5cm_all_fraction': sum(r['error_m'] <= .015 for r in accepted) / len(rows) if rows else None}


def grouped(rows):
    return {'all': summarize(rows),
            'visible': summarize([r for r in rows if r['visible']]),
            'occluded': summarize([r for r in rows if not r['visible']]),
            'per_target': {name: summarize([r for r in rows if r['target'] == i]) for i, name in enumerate(('bottle', 'handle'))},
            'per_phase': {str(p): summarize([r for r in rows if r['phase'] == p]) for p in sorted({r['phase'] for r in rows})}}


@torch.inference_mode()
def legacy_predict(runtime, frame):
    image = torch.from_numpy(frame['rgb'].transpose(2, 0, 1).copy()).float()[None].expand(2, -1, -1, -1) / 255
    heat, offset = runtime.model(image, runtime.model.head.language.tokenize(QUERIES))
    depth = torch.from_numpy(frame['depth'])[None].expand(2, -1, -1)
    k = torch.from_numpy(frame['K'])[None].expand(2, -1, -1)
    t = torch.from_numpy(frame['T'])[None].expand(2, -1, -1)
    result = runtime.model.decode_detailed(heat, offset, depth, k, t)
    return [{'point': result['points'][i].numpy(), 'valid': bool(result['valid'][i]),
             'pixel': result['pixels'][i].numpy(), 'center_depth': float(result['center_depth'][i]),
             'support': int(result['support'][i]), 'confidence': None} for i in range(2)]


def records(predictions, truth, frame):
    rows = []
    for target, prediction in enumerate(predictions):
        point = prediction['point']
        valid = prediction['valid']
        uv, z = project(truth['points'][target], frame['K'], frame['T'])
        pred_uv, pred_z = project(point, frame['K'], frame['T']) if point is not None and np.isfinite(point).all() else (None, None)
        error = float(np.linalg.norm(point - truth['points'][target])) if valid else None
        row = {k: truth[k] for k in ('episode', 'frame', 'seed', 'phase')}
        row.update(target=target, visible=bool(truth['visible'][target]),
                   visible_fraction=float(truth['visible_fraction'][target]),
                   visible_pixels=int(truth['visible_pixels'][target]), valid=valid, error_m=error,
                   point=point.tolist() if point is not None and np.isfinite(point).all() else None,
                   truth=truth['points'][target].tolist(), gt_uv=uv.tolist(),
                   pred_uv=pred_uv.tolist() if pred_uv is not None and np.isfinite(pred_uv).all() else None,
                   support=prediction['support'], confidence=prediction['confidence'])
        if pred_uv is not None and np.isfinite(pred_uv).all():
            row['ray_error_at_true_depth_m'] = float(np.linalg.norm(unproject(pred_uv, z, frame['K'], frame['T']) - truth['points'][target]))
            row['depth_error_m'] = float(abs(pred_z - z))
        if 'mask' in prediction:
            target_mask = truth['semantic'] == target + 1
            intersection = (target_mask & prediction['mask']).sum()
            union = (target_mask | prediction['mask']).sum()
            row['mask_iou'] = float(intersection / union) if union else None
        rows.append(row)
    return rows


def gallery(dataset, rows, output):
    selected = sorted(rows, key=lambda r: (r['valid'], -(r['error_m'] or 0)))[:16]
    lookup = {}
    for row in selected:
        lookup.setdefault((row['episode'], row['frame']), []).append(row)
    cards = []
    for frame, truth in dataset:
        key = (truth['episode'], truth['frame'])
        if key not in lookup:
            continue
        for row in lookup[key]:
            markers = [{'gt_uv': None, 'pred_uv': None} for _ in range(2)]
            markers[row['target']] = {'gt_uv': row['gt_uv'], 'pred_uv': row['pred_uv']}
            png = png_bytes(overlay(frame['rgb'], markers))
            data = base64.b64encode(png).decode()
            caption = {k: row[k] for k in ('episode', 'frame', 'phase', 'target', 'visible_fraction', 'valid', 'error_m')}
            cards.append(f'<article><img src="data:image/png;base64,{data}"><pre>{html.escape(json.dumps(caption, indent=2))}</pre></article>')
    output.write_text('<!doctype html><meta charset="utf-8"><title>P5 error atlas</title>'
                      '<style>body{font:16px system-ui;background:#111827;color:#eee}main{display:flex;flex-wrap:wrap;gap:20px}img{width:384px;image-rendering:pixelated}article{max-width:400px}</style>'
                      '<h1>P5 error atlas</h1><p>Rejected targets first, then largest accepted error. Circle: true center; cross: prediction.</p><main>' + ''.join(cards) + '</main>')


@torch.inference_mode()
def evaluate(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    torch.set_num_threads(args.threads)
    dataset = P5Frames(args.source, args.labels, args.split)
    runtime = RGBDStudentRuntime(args.checkpoint)
    checkpoint = runtime.checkpoint
    if checkpoint['source_sha256'] != file_hash(args.source) or checkpoint['labels_sha256'] != file_hash(args.labels):
        raise ValueError('Checkpoint/source/teacher contract mismatch')
    if set(checkpoint['train_episodes']) & set(dataset.episodes):
        raise ValueError('Training episode leakage')
    first_frame = next(iter(dataset))[0]
    for _ in range(3):
        runtime.predict_detailed(first_frame)
    modes = ('none', 'drop-rgb', 'drop-depth', 'drop-rgbd', 'swap-text')
    result = {'schema_version': 1, 'checkpoint_sha256': file_hash(args.checkpoint),
              'source_sha256': file_hash(args.source), 'labels_sha256': file_hash(args.labels),
              'split': args.split, 'episodes': dataset.episodes, 'cpu_threads': args.threads,
              'visibility': 'teacher visible pixels >=6 and visible/amodal >=0.25',
              'scope': 'Fixed head camera, two object centers, finite known-vocabulary phrases; no unseen-object or synonym claim',
              'postprocess': 'Bottle PCA axis alignment <0.4: center z = measured cloud median z - 0.68 * known 0.025m cylinder radius; selected on validation',
              'ablations': {}}
    normal_rows = []
    for mode in modes:
        rows, times = [], []
        for frame, truth in dataset:
            started = time.perf_counter()
            predictions = runtime.predict_detailed(frame, ablation=mode)
            times.append((time.perf_counter() - started) * 1000)
            rows.extend(records(predictions, truth, frame))
        result['ablations'][mode] = grouped(rows)
        if mode == 'none':
            normal_rows = rows
            result['inference_ms'] = {f'p{p}': float(np.percentile(times, p)) for p in (50, 95, 99)}
            result['inference_ms']['scope'] = 'Two queries incl. preprocessing, segmentation, geometry, text; excludes HDF5 IO and rendering'
        print(json.dumps({'ablation': mode, **result['ablations'][mode]['all']}), flush=True)
    # Shared visual output is identical across phrases; reuse it as a cached backbone.
    by_frame = {}
    for row in normal_rows:
        by_frame.setdefault((row['episode'], row['frame']), {})[row['target']] = row
    paraphrases, phrase_rows = [], []
    for target, phrases in enumerate(HELDOUT_PHRASES):
        for phrase in phrases:
            scores = runtime.model.language(runtime.model.language.tokenize([phrase])).softmax(-1)[0]
            choice = int(scores.argmax())
            confidence = float(scores[choice])
            paraphrases.append({'phrase': phrase, 'expected_target': target, 'predicted_target': choice, 'confidence': confidence})
            for pair in by_frame.values():
                base, selected = pair[target], pair[choice]
                row = dict(base)
                row['valid'] = selected['valid'] and confidence >= .8
                row['error_m'] = float(np.linalg.norm(np.array(selected['point']) - base['truth'])) if row['valid'] else None
                phrase_rows.append(row)
    result['heldout_paraphrases'] = {'scope': 'Disjoint sentences; known vocabulary; cached shared visual features',
                                    'language_accuracy': sum(p['expected_target'] == p['predicted_target'] and p['confidence'] >= .8 for p in paraphrases) / len(paraphrases),
                                    'phrases': paraphrases, 'metrics': grouped(phrase_rows)}
    if args.baseline:
        baseline = GroundingRuntime(args.baseline)
        baseline_rows = []
        for frame, truth in dataset:
            baseline_rows.extend(records(legacy_predict(baseline, frame), truth, frame))
        result['frozen_patch_v3'] = {'checkpoint_sha256': file_hash(args.baseline), **grouped(baseline_rows)}
        (output / 'baseline_rows.json').write_text(json.dumps(baseline_rows, indent=2, allow_nan=False) + '\n')
    visible = result['ablations']['none']['visible']
    result['gates'] = {'visible_mean_le_1_5cm': visible['mean_error_m'] is not None and visible['mean_error_m'] <= .015,
                       'visible_coverage_ge_90pct': visible['coverage'] is not None and visible['coverage'] >= .9,
                       'advanced_visible_p95_le_1_5cm': visible['p95_error_m'] is not None and visible['p95_error_m'] <= .015}
    (output / 'report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (output / 'rows.json').write_text(json.dumps(normal_rows, indent=2, allow_nan=False) + '\n')
    gallery(dataset, normal_rows, output / 'error_atlas.html')
    print(json.dumps({'gates': result['gates'], 'visible': visible, 'latency': result['inference_ms']}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--labels', default='data/grounding_p5_labels.h5')
    p.add_argument('--checkpoint', default='weights/p5_student/best.pt')
    p.add_argument('--baseline', default='weights/grounding_patch_v3_cpu_pilot/selected.pt')
    p.add_argument('--split', choices=('val', 'test'), default='test')
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--output', default='artifacts/p5_student/test')
    args = p.parse_args()
    if args.threads < 1:
        p.error('threads must be positive')
    evaluate(args)
