"""Export BC or CFM vector field to ONNX and require deterministic CPU parity."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnxruntime as ort
from utils.checkpoint import load_policy


def export(checkpoint_path, output_path):
    model,ckpt=load_policy(checkpoint_path);mc=ckpt['model_config']
    torch.set_num_threads(2);torch.manual_seed(42)
    condition=torch.randn(1,mc['cond_dim'])
    if ckpt['policy_type']=='bc':
        inputs=(condition,);names=['condition']
    else:
        inputs=(torch.randn(1,mc['chunk_size'],14),torch.tensor([.4]),condition);names=['x_t','t','condition']
    output=Path(output_path);output.parent.mkdir(parents=True,exist_ok=True)
    torch.onnx.export(model,inputs,str(output),input_names=names,output_names=['actions' if ckpt['policy_type']=='bc' else 'velocity'],opset_version=17,dynamo=False)
    opts=ort.SessionOptions();opts.intra_op_num_threads=2
    session=ort.InferenceSession(str(output),sess_options=opts,providers=['CPUExecutionProvider'])
    with torch.no_grad():
        reference=model(*inputs).numpy()
    actual=session.run(None,{name:tensor.numpy() for name,tensor in zip(names,inputs)})[0]
    error=float(np.max(np.abs(reference-actual)))
    if error>1e-4:
        raise RuntimeError(f'ONNX parity failed: {error}')
    metadata={key:ckpt[key] for key in ('schema_version','policy_type','model_config','action_dt','dataset_sha256')}
    metadata['stats']={key:value.tolist() for key,value in ckpt['stats'].items()}
    metadata['parity_max_abs_error']=error
    output.with_suffix('.json').write_text(json.dumps(metadata,indent=2))
    print('Exported',output,'parity max abs error',error)
    return error

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    args=p.parse_args();export(args.checkpoint,args.output)
