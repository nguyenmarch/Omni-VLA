"""Measure complete head RGB-D capture + two-query inference on fresh resets."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from models.rgbd_student import RGBDStudentRuntime
from simulation.vision import capture_head
from utils.config import make_environment
from utils.checkpoint import file_hash


def benchmark(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(args.threads)
    runtime = RGBDStudentRuntime(args.checkpoint)
    env = make_environment(image_size=runtime.image_size)
    elapsed, accepted = [], 0
    try:
        for index in range(args.frames + 3):
            env.reset(seed=args.seed + index)
            start = time.perf_counter()
            frame = capture_head(env)
            rows = runtime.predict_detailed(frame)
            duration = (time.perf_counter() - start) * 1000
            if index >= 3:
                elapsed.append(duration)
                accepted += int(all(row['valid'] for row in rows))
    finally:
        env.close()
    result = {'frames': args.frames, 'warmup_frames': 3, 'seed_start': args.seed,
              'accepted_pairs': accepted, 'threads': args.threads,
              'checkpoint_sha256': file_hash(args.checkpoint),
              'capture_and_infer_ms': {f'p{p}': float(np.percentile(elapsed, p)) for p in (50, 95, 99)},
              'sustained_hz': 1000 / float(np.mean(elapsed)),
              'scope': 'Head256 RGB-D, calibration, segmentation, center regression and two text queries; reset poses, no physics/policy concurrency; not hard real time'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', default='weights/p5_student/best.pt')
    p.add_argument('--frames', type=int, default=30)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--seed', type=int, default=11000)
    p.add_argument('--output', default='artifacts/p5_student/cpu_throughput.json')
    args = p.parse_args()
    if min(args.frames, args.threads) < 1:
        p.error('frames and threads must be positive')
    benchmark(args)
