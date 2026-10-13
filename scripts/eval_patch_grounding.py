"""Evaluate all held-out targets using a shared patch decoder, including rejection coverage."""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from models.grounding3d import Grounding3D
from utils.patch_grounding_dataset import PatchGroundingFrames
from utils.checkpoint import file_hash
from scripts.train_patch_grounding import evaluate
from simulation.vision import QUERIES


def run(args):
    torch.set_num_threads(2);ds=PatchGroundingFrames(args.dataset,args.split)
    c=torch.load(args.checkpoint,map_location='cpu',weights_only=True);cfg=dict(c['model_config'])
    if c['schema_version']=='omni-grounding-rgbd-v2':
        cfg['decoder_config']=ds.decoder_config
        model=Grounding3D(**cfg)
        missing,extra=model.load_state_dict(c['model'],strict=False)
        if extra or any(not k.startswith('robust_decoder.') for k in missing):raise ValueError('Unexpected legacy state mismatch')
        scope='Legacy offsets evaluated with patch decoder; diagnostic baseline, not deployment checkpoint'
    elif c['schema_version']=='omni-grounding-patch-v3':
        if cfg.get('decoder_config')!=ds.decoder_config or c['dataset_sha256']!=file_hash(args.dataset):raise ValueError('Checkpoint/data contract mismatch')
        if set(c['train_episodes']) & set(ds.episodes) and args.split!='train':raise ValueError('Training episode leakage')
        model=Grounding3D(**cfg);model.load_state_dict(c['model']);scope='Patch-trained checkpoint'
    else:raise ValueError('Unsupported schema')
    if model.img_size!=ds.size:raise ValueError('Image size mismatch')
    model.eval();ids=model.head.language.tokenize(QUERIES);loader=DataLoader(ds,batch_size=32)
    report={'scope':scope,'split':args.split,'checkpoint_sha256':file_hash(args.checkpoint),'dataset_sha256':file_hash(args.dataset),
        'decoder_config':ds.decoder_config,'selection_penalty_m':1.,'episodes':ds.episodes,
        'ablations':{a:evaluate(model,loader,ids,torch.device('cpu'),a) for a in ('none','drop-image','swap-text')}}
    out=Path(args.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report['ablations']['none'],indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',default='data/grounding_patch_v3_256.h5')
    p.add_argument('--checkpoint',required=True);p.add_argument('--split',choices=('val','test'),default='test');p.add_argument('--output',required=True)
    run(p.parse_args())
