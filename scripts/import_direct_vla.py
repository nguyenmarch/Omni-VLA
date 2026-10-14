"""Validate a Kaggle direct-VLA archive before importing any files."""
import argparse
import io
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from models.direct_vla import DirectVLA, tokenize, TASK_PHRASES
from utils.checkpoint import file_hash
from utils.vla_dataset import VLADataset
from utils.reporting import write_json


def import_candidate(archive_path, dataset, output, report):
    required={'best.pt','last.pt','metrics.jsonl'}
    with zipfile.ZipFile(archive_path) as archive:
        if set(archive.namelist()) != required or len(archive.infolist())!=3:
            raise ValueError('Expected exactly best.pt, last.pt and metrics.jsonl at archive root')
        if any(info.file_size>100*1024*1024 for info in archive.infolist()):
            raise ValueError('Unexpected archive member size')
        raw={name:archive.read(name) for name in required}
    checkpoints={name:torch.load(io.BytesIO(raw[name]),map_location='cpu',weights_only=True) for name in ('best.pt','last.pt')}
    rows=[json.loads(line) for line in raw['metrics.jsonl'].decode().splitlines()]
    if not rows or [row['epoch'] for row in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('Incomplete or duplicate training epochs')
    data=VLADataset(dataset,'train');dataset_hash=file_hash(dataset)
    for name,c in checkpoints.items():
        if (c['schema']!='omni-direct-vla-policy-v1' or c['dataset_sha256']!=dataset_hash
                or c['splits']!=data.splits or c['smoke_only'] or c['args'].get('max_batches') is not None):
            raise ValueError('Checkpoint dataset/splits/schema mismatch or smoke training')
        if c['image_size']!=data.size or c['model_config']['chunk_size']!=data.chunk or abs(c['action_dt']-data.action_dt)>1e-9:
            raise ValueError('Image/chunk/timing contract mismatch')
        for key,value in data.stats.items():
            np.testing.assert_allclose(c['robot_stats'][key].numpy(),value,rtol=1e-6,atol=1e-6)
        if not np.isfinite([r['val_action_mse'] for r in rows]).all():raise ValueError('Nonfinite metrics')
        if abs(rows[c['epoch']-1]['val_action_mse']-c['validation_action_mse'])>1e-10:
            raise ValueError('Checkpoint metric does not match log')
        model=DirectVLA(**c['model_config']);model.load_state_dict(c['model']);model.eval()
        with torch.inference_mode():
            pred=model(torch.zeros(1,1,7,data.size,data.size),torch.zeros(1,1,50),
                       tokenize([TASK_PHRASES[0][0]],c['model_config']['vocabulary']),torch.ones(1,1,dtype=torch.bool))
        if pred['action'].shape!=(1,data.chunk,14) or not torch.isfinite(pred['action']).all():
            raise ValueError('CPU forward contract failed')
    best,last=checkpoints['best.pt'],checkpoints['last.pt']
    if last['epoch']!=len(rows) or last['epoch']!=last['args']['epochs']:
        raise ValueError('Full planned training not completed')
    if best['validation_action_mse']!=min(row['val_action_mse'] for row in rows):
        raise ValueError('Best checkpoint was not selected by validation metric')
    output=Path(output)
    if output.exists():raise FileExistsError(output)
    if Path(report).exists():raise FileExistsError(report)
    output.mkdir(parents=True)
    for name,content in raw.items():
        with (output/name).open('xb') as stream:stream.write(content)
    result={'archive_sha256':file_hash(archive_path),'dataset_sha256':dataset_hash,
            'checkpoint_sha256':file_hash(output/'best.pt'),'output':str(output),'training_epochs':last['epoch'],
            'best_epoch':best['epoch'],'validation_action_mse':best['validation_action_mse'],
            'train_action_mse_final':rows[-1]['train_action_mse'],'training_torch':best['torch_version'],
            'CPU_torch':str(torch.__version__),'CPU_forward':'passed','normalization_and_splits':'verified',
            'metrics':rows,'limitation':'Archive/source equivalence cannot be proved from checkpoint metadata; dataset and architecture contracts checked. Validation MSE is not rollout success.'}
    write_json(report,result);print(json.dumps({k:v for k,v in result.items() if k!='metrics'},indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--archive',default='direct_vla_outputs.zip')
    p.add_argument('--dataset',default='data/direct_vla_v1.h5');p.add_argument('--output',default='weights/direct_vla_kaggle')
    p.add_argument('--report',default='artifacts/direct_vla_kaggle/import.json');args=p.parse_args()
    torch.set_num_threads(2);import_candidate(args.archive,args.dataset,args.output,args.report)
