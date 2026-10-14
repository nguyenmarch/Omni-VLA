"""Train direct RGB-D/language recurrent action policy on CPU or Kaggle GPU."""
import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from models.direct_vla import DirectVLA, VOCABULARY
from utils.vla_dataset import VLADataset
from utils.checkpoint import atomic_save, file_hash


def train(args):
    torch.manual_seed(args.seed);torch.set_num_threads(args.threads)
    device=torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    out=Path(args.output)
    resume=getattr(args,'resume',None)
    if out.exists() and not resume:raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=bool(resume))
    train=VLADataset(args.dataset,'train',args.history)
    val=VLADataset(args.dataset,'val',args.history,train.stats)
    config={'width':128,'chunk_size':train.chunk,'vocabulary':VOCABULARY}
    model=DirectVLA(**config).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    loaders=[DataLoader(ds,batch_size=args.batch_size,shuffle=i==0,num_workers=0) for i,ds in enumerate((train,val))]
    dataset_hash=file_hash(args.dataset)
    best=float('inf');log=[];start=0
    if resume:
        previous=torch.load(resume,map_location=device,weights_only=True)
        if (previous.get('schema')!='omni-direct-vla-policy-v1' or 'rng' not in previous
                or previous['dataset_sha256']!=dataset_hash or previous['model_config']!=config
                or previous['history']!=args.history or previous['args']['batch_size']!=args.batch_size
                or previous['args'].get('max_batches')!=args.max_batches):
            raise ValueError('Resume dataset/model/history/batch/smoke mismatch')
        model.load_state_dict(previous['model']);optimizer.load_state_dict(previous['optimizer'])
        torch.set_rng_state(previous['rng'].cpu())
        if device.type=='cuda' and previous.get('cuda_rng'):
            torch.cuda.set_rng_state_all([state.cpu() for state in previous['cuda_rng']])
        start=previous['epoch'];best=previous['best_validation_action_mse']
    for epoch in range(start,args.epochs):
        metrics=[]
        for i,loader in enumerate(loaders):
            model.train(i==0);total=count=0
            for batch_index,batch in enumerate(loader):
                if args.max_batches and batch_index>=args.max_batches:break
                batch={key:value.to(device) for key,value in batch.items()}
                with torch.set_grad_enabled(i==0):
                    pred=model(batch['images'],batch['robot'],batch['tokens'],batch['history_mask'])
                    mask=batch['action_mask']
                    action=((pred['action']-batch['action']).square()*mask).sum()/(mask.sum()*14).clamp_min(1)
                    loss=action+.1*torch.nn.functional.cross_entropy(pred['phase_logits'],batch['phase'])+.1*torch.nn.functional.smooth_l1_loss(pred['landmarks'],batch['landmarks'])
                    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                    if i==0:
                        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
                total+=float(action.detach())*len(batch['robot']);count+=len(batch['robot'])
            if not count:raise ValueError('Empty data split')
            metrics.append(total/count)
        row={'epoch':epoch+1,'train_action_mse':metrics[0],'val_action_mse':metrics[1]};log.append(row);print(json.dumps(row),flush=True)
        improved=metrics[1]<best;best=min(best,metrics[1])
        payload={'schema':'omni-direct-vla-policy-v1','model_config':config,'model':model.state_dict(),
                 'robot_stats':{k:torch.as_tensor(v) for k,v in train.stats.items()},'history':args.history,
                 'image_size':train.size,'action_dt':train.action_dt,'dataset_sha256':dataset_hash,
                 'splits':train.splits,'epoch':epoch+1,'optimizer':optimizer.state_dict(),'args':vars(args),
                 'validation_action_mse':metrics[1],'best_validation_action_mse':best,
                 'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else [],
                 'torch_version':str(torch.__version__),'smoke_only':bool(args.max_batches)}
        atomic_save(payload,out/'last.pt')
        if improved:atomic_save(payload,out/'best.pt')
        with (out/'metrics.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',required=True)
    p.add_argument('--output',default='weights/direct_vla');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--batch-size',type=int,default=32);p.add_argument('--history',type=int,default=8)
    p.add_argument('--device');p.add_argument('--threads',type=int,default=2);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-batches',type=int,help='Smoke only: cap batches per split/epoch')
    p.add_argument('--resume',help='Resume last.pt with identical dataset/model/history/batch settings')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.history,args.threads)<1 or args.max_batches is not None and args.max_batches<1:p.error('Positive parameters required')
    train(args)
