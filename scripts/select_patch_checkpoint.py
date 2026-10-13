"""Choose among retained checkpoints using validation accurate coverage, never test."""
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.grounding3d import Grounding3D
from utils.patch_grounding_dataset import PatchGroundingFrames
from utils.checkpoint import atomic_save,file_hash
from scripts.train_patch_grounding import evaluate, selection_key
from simulation.vision import QUERIES


def main(args):
    torch.set_num_threads(2);dataset=PatchGroundingFrames(args.dataset,'val');loader=DataLoader(dataset,batch_size=32)
    candidates=[];best=None;score=None;digest=file_hash(args.dataset)
    for path in args.checkpoints:
        c=torch.load(path,map_location='cpu',weights_only=True)
        if c['dataset_sha256']!=digest or c['model_config']['decoder_config']!=dataset.decoder_config:raise ValueError('Data contract mismatch')
        if set(c['train_episodes'])&set(dataset.episodes):raise ValueError('Validation leakage')
        m=Grounding3D(**c['model_config']);m.load_state_dict(c['model']);m.eval()
        metric=evaluate(m,loader,m.head.language.tokenize(QUERIES),torch.device('cpu'))
        key=selection_key(metric)
        candidates.append({'checkpoint_sha256':file_hash(path),'epoch':c['metric']['epoch'],'validation':metric})
        if score is None or key<score:score=key;best=c
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    best['selection']='maximize valid within 1.5cm count on all validation targets, then minimize invalid=1m penalized error; retained candidates only'
    best['selection_candidates']=candidates;atomic_save(best,out)
    print('Selected epoch',best['metric']['epoch'],'among',[c['epoch'] for c in candidates])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='data/grounding_patch_v3_256.h5')
    p.add_argument('--checkpoints',nargs='+',required=True);p.add_argument('--output',required=True);main(p.parse_args())
