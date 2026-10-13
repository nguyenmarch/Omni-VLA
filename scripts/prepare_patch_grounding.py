"""Version supervision using the identical patch decoder used at inference."""
import argparse
import json
from pathlib import Path
import shutil
import h5py
import numpy as np
import torch
from utils.robust_depth import RobustDepthDecoder
from utils.camera import project
from utils.checkpoint import file_hash


def prepare(args):
    torch.set_num_threads(2)
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    config={'patch_size':7,'quantile':args.quantile,'min_support':6,'min_fraction':.2,'max_iqr':.10,
            'depth_range':[.05,5.],'workspace':[[-.1,-.75,.4],[.95,.75,1.5]]}
    decoder=RobustDepthDecoder(**config);temporary=output.with_suffix('.tmp')
    if temporary.exists():raise FileExistsError(temporary)
    source_hash=file_hash(args.source);shutil.copyfile(args.source,temporary)
    report={};total=eligible=0
    try:
        with h5py.File(temporary,'a') as h:
            if h.attrs['schema_version']!='omni-grounding-rgbd-v2':raise ValueError('Require raw v2 RGB-D dataset')
            h.attrs.update(schema_version='omni-grounding-patch-v3',source_sha256=source_hash,
                decoder_config=json.dumps(config),offset_limit_m=.08)
            for name,g in h['episodes'].items():
                records={key:[] for key in ('patch_uv','patch_surface','patch_offset','patch_label_valid','patch_support','patch_iqr')}
                size=g['rgb'].shape[1]
                for index in range(len(g['rgb'])):
                    k=torch.from_numpy(g['K'][index])[None].expand(2,-1,-1)
                    t=torch.from_numpy(g['T'][index])[None].expand(2,-1,-1)
                    depth=torch.from_numpy(g['depth'][index])[None].expand(2,-1,-1)
                    projections=[project(p,g['K'][index],g['T'][index]) for p in g['targets'][index]]
                    uv=np.stack([p[0] for p in projections]);z=torch.tensor([p[1] for p in projections],dtype=torch.float32)
                    in_view=np.isfinite(uv).all(1)&(uv>=0).all(1)&(uv<size).all(1)&(z.numpy()>0)
                    heat=torch.full((2,size,size),-torch.inf)
                    for target in range(2):
                        if in_view[target]:
                            u,v=np.rint(uv[target]).astype(int).clip(0,size-1);heat[target,v,u]=1
                    raw=decoder(heat,depth,k,t);offset=z-raw['surface_depth']
                    corrected=decoder(heat,depth,k,t,offset)
                    valid=corrected['valid']&torch.isfinite(offset)&(offset.abs()<=.08)&torch.from_numpy(in_view)
                    for key,value in [('patch_uv',uv.astype(np.float32)),('patch_surface',raw['surface_depth'].numpy()),
                        ('patch_offset',offset.numpy()),('patch_label_valid',valid.numpy()),
                        ('patch_support',raw['support'].numpy()),('patch_iqr',raw['iqr'].numpy())]:records[key].append(value)
                for key,values in records.items():g.create_dataset(key,data=np.asarray(values),compression='gzip')
                count=int(np.sum(records['patch_label_valid']));targets=2*len(g['rgb'])
                report[name]={'targets':targets,'eligible':count,'coverage':count/targets}
                total+=targets;eligible+=count
            h.attrs['supervision_report']=json.dumps(report)
        temporary.replace(output)
    except BaseException:
        temporary.unlink(missing_ok=True);raise
    result={'source_sha256':source_hash,'dataset_sha256':file_hash(output),'decoder_config':config,
            'targets':total,'eligible':eligible,'coverage':eligible/total,'episodes':report}
    output.with_suffix('.report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default='data/grounding_rgbd_v2_256.h5')
    p.add_argument('--output',default='data/grounding_patch_v3_256.h5')
    p.add_argument('--quantile',type=float,default=.5)
    a=p.parse_args()
    if not np.isfinite(a.quantile) or not 0<=a.quantile<=1:p.error('Invalid quantile')
    prepare(a)
