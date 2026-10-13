"""Evaluate a frozen grounding checkpoint on an episode-held-out split."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.grounding3d import GroundingRuntime
from utils.grounding_dataset import GroundingFrames
from utils.checkpoint import file_hash
from simulation.vision import QUERIES
from scripts.train_grounding_rgbd import evaluate_grounder


def evaluate(args):
    torch.set_num_threads(2)
    runtime=GroundingRuntime(args.checkpoint)
    if runtime.checkpoint['dataset_sha256']!=file_hash(args.dataset):
        raise ValueError('Dataset differs from training dataset')
    ds=GroundingFrames(args.dataset,args.split)
    if args.split=='test' and set(ds.episodes)&set(runtime.checkpoint['train_episodes']+runtime.checkpoint['val_episodes']):
        raise ValueError('Test episode leakage')
    ids=runtime.model.head.language.tokenize(QUERIES)
    report={a:evaluate_grounder(runtime.model,DataLoader(ds,batch_size=32),ids,'cpu',a)
            for a in ('none','drop-image','swap-text')}
    report.update(split=args.split,episodes=ds.episodes,skipped_targets=ds.skipped,
                  checkpoint_sha256=file_hash(args.checkpoint),dataset_sha256=file_hash(args.dataset),
                  scope='Current scene/camera; depth-visibility heuristic <=8cm; excluded targets are not evaluated. Drop-image zeros RGB only; depth/calibration remain available.')
    path=Path(args.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--dataset',required=True)
    p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--output',default='artifacts/grounding_rgbd_test.json')
    evaluate(p.parse_args())
