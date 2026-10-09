import argparse
import json
import h5py
import numpy as np

def validate(path):
    with h5py.File(path,'r') as h5:
        assert h5.attrs['schema_version']=='omni-state-v1'
        splits=json.loads(h5.attrs['splits'])
        names=[name for ids in splits.values() for name in ids]
        assert len(names)==len(set(names)), 'split leakage'
        assert set(names)==set(h5['episodes']), 'unassigned episode'
        counts={}
        for name in names:
            g=h5[f'episodes/{name}'];n=len(g['action'])
            assert g['action'].shape==(n,14)
            assert g['condition'].shape==(n,67)
            assert np.isfinite(g['condition'][:]).all()
            assert np.isfinite(g['action'][:]).all()
            assert np.max(np.abs(g['action'][:]))<=1
            assert np.allclose(np.diff(g['obs/time'][:,0]),h5.attrs['action_dt'],atol=1e-5)
            assert len(g['phase'])==n
            counts[name]=n
        report={'schema':h5.attrs['schema_version'],'episodes':len(names),'steps':sum(counts.values()),'splits':{k:len(v) for k,v in splits.items()}}
        print(json.dumps(report,indent=2))
        return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('dataset');validate(p.parse_args().dataset)
