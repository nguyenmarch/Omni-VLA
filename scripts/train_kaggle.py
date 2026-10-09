"""Train BC/CFM on Kaggle GPU or CPU smoke runs, with resumable checkpoints."""
import argparse
import json
import random
from pathlib import Path
import time
import numpy as np
import h5py
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
from utils.dataset import BimanualFlowDataset, phase_sample_weights
from utils.checkpoint import make_policy, atomic_save, file_hash


def masked_loss(prediction, target, mask):
    return ((prediction-target).square()*mask).sum()/(mask.sum()*prediction.shape[-1]).clamp_min(1)


def train(args):
    config = yaml.safe_load(Path(args.config).read_text())
    tc, mc = config['train'], config['model']
    tc['first_action_weight'] = args.first_action_weight
    seed = tc['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.set_num_threads(args.threads)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out = Path(args.output);out.mkdir(parents=True, exist_ok=True)
    dataset_hash = file_hash(args.dataset)
    with h5py.File(args.dataset,'r') as h5:
        action_dt=float(h5.attrs['action_dt'])
        if h5.attrs['condition_dim']!=mc['cond_dim'] or h5.attrs['action_dim']!=mc['action_dim']:
            raise ValueError('Dataset/model dimensions do not match')
    train_ds = BimanualFlowDataset(args.dataset, mc['chunk_size'], 'train',cache_state=True)
    val_ds = BimanualFlowDataset(args.dataset, mc['chunk_size'], 'val', train_ds.stats,cache_state=True)
    if not len(train_ds) or not len(val_ds):
        raise ValueError('Need successful train and validation episodes; collect at least 3')
    workers = tc.get('num_workers', 0)
    batch = args.batch_size or tc['batch_size']
    sampler = None
    if tc.get('phase_balanced', False):
        weights, counts = phase_sample_weights(train_ds, tc.get('phase_sampling_min_count',1))
        sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)
        print(json.dumps({'train_phase_counts':counts,'sampling':'phase_balanced',
                          'phase_sampling_min_count':tc.get('phase_sampling_min_count',1)}),flush=True)
    loader = DataLoader(train_ds, batch_size=batch, shuffle=sampler is None, sampler=sampler,
                        num_workers=workers, pin_memory=device.type=='cuda')
    val_loader = DataLoader(val_ds, batch_size=batch, num_workers=workers)
    model = make_policy(args.policy, mc).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc['learning_rate'], weight_decay=tc['weight_decay'])
    epochs = args.epochs or tc['epochs']
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    amp = device.type=='cuda' and tc.get('mixed_precision', True)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    start, best = 0, float('inf')
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        if ckpt['dataset_sha256'] != dataset_hash or ckpt['policy_type'] != args.policy or ckpt['model_config'] != mc:
            raise ValueError('Resume config/dataset mismatch')
        if ckpt['scheduler']['T_max'] != epochs or ckpt['train_config'].get('first_action_weight',1.) != args.first_action_weight:
            raise ValueError('Resume requires identical total epochs and first-action weight')
        if ckpt['train_config'].get('phase_balanced',False) != tc.get('phase_balanced',False):
            raise ValueError('Resume requires identical phase sampling')
        if ckpt['train_config'].get('phase_sampling_min_count',1) != tc.get('phase_sampling_min_count',1):
            raise ValueError('Resume requires identical phase sampling floor')
        model.load_state_dict(ckpt['model']);optimizer.load_state_dict(ckpt['optimizer']);scheduler.load_state_dict(ckpt['scheduler']);scaler.load_state_dict(ckpt['scaler'])
        torch.set_rng_state(ckpt['rng'].cpu())
        if device.type=='cuda' and ckpt.get('cuda_rng'):
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt['cuda_rng']])
        start, best = ckpt['epoch']+1, ckpt['best_val']
    fixed_gen = torch.Generator(device=device).manual_seed(seed+1000)
    fixed_noise = torch.randn(len(val_ds), mc['chunk_size'], mc['action_dim'], generator=fixed_gen, device=device)
    print(json.dumps({'device': str(device), 'gpu': torch.cuda.get_device_name() if device.type=='cuda' else None,
                      'train_samples': len(train_ds), 'val_samples': len(val_ds), 'policy': args.policy}), flush=True)
    history = []
    stop = min(epochs,start+args.max_epochs_per_run) if args.max_epochs_per_run else epochs
    for epoch in range(start, stop):
        begun = time.perf_counter(); model.train(); total = count = 0
        for sample in loader:
            condition, action, mask = (sample[k].to(device) for k in ('condition', 'action', 'mask'))
            mask = mask.clone();mask[:,0] *= args.first_action_weight
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                if args.policy == 'cfm':
                    noise = torch.randn_like(action); t = torch.rand(action.shape[0], device=device)
                    xt = (1-t[:,None,None])*noise + t[:,None,None]*action
                    prediction, target = model(xt, t, condition), action-noise
                else:
                    prediction, target = model(condition), action
                loss = masked_loss(prediction, target, mask)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss')
            scaler.scale(loss).backward();scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(optimizer);scaler.update()
            total += loss.item()*len(action);count += len(action)
        scheduler.step();model.eval();val_total=val_count=offset=0
        with torch.no_grad():
            for sample in val_loader:
                condition, action, mask = (sample[k].to(device) for k in ('condition', 'action', 'mask'))
                prediction = model.sample_euler(condition, mc['euler_steps'], noise=fixed_noise[offset:offset+len(action)])
                loss = masked_loss(prediction, action, mask)
                val_total += loss.item()*len(action);val_count += len(action);offset += len(action)
        val = val_total/val_count
        improved = val < best;best = min(val, best)
        stats = {key: torch.as_tensor(value) for key,value in train_ds.stats.items()}
        payload = {'schema_version': 'omni-state-v1', 'policy_type': args.policy, 'model_config': mc,
                   'train_config': tc, 'model': model.state_dict(), 'stats': stats, 'epoch': epoch, 'best_val': best,
                   'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                   'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                   'dataset_sha256': dataset_hash, 'splits': train_ds.splits, 'action_dt': action_dt,
                   'torch_version': str(torch.__version__)}
        atomic_save(payload, out/'last.pt')
        if improved:
            atomic_save(payload, out/'best.pt')
        row = {'epoch': epoch+1, 'train_loss': total/count, 'val_sample_mse': val, 'seconds': time.perf_counter()-begun}
        history.append(row);print(json.dumps(row), flush=True)
        with (out/'metrics.jsonl').open('a') as log:
            log.write(json.dumps(row)+'\n')
    train_ds.close();val_ds.close()
    return history


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True);p.add_argument('--policy', choices=['bc','cfm'], default='bc')
    p.add_argument('--config', default=None)
    p.add_argument('--output', default='weights/bc');p.add_argument('--device');p.add_argument('--epochs',type=int)
    p.add_argument('--max-epochs-per-run',type=int,help='Stop early while preserving total-epoch scheduler for resume')
    p.add_argument('--first-action-weight',type=float,default=1.)
    p.add_argument('--batch-size',type=int);p.add_argument('--threads',type=int,default=2);p.add_argument('--resume')
    args=p.parse_args()
    if args.config is None:
        filename='train_bc_kaggle.yaml' if args.policy=='bc' else 'train_cfm_kaggle.yaml'
        args.config=str(Path(__file__).resolve().parents[1]/'configs'/filename)
    if args.first_action_weight<=0 or args.threads<1 or (args.epochs is not None and args.epochs<1) or (args.batch_size is not None and args.batch_size<1):
        p.error('Weights, threads, epochs and batch size must be positive')
    train(args)

if __name__=='__main__':
    main()
