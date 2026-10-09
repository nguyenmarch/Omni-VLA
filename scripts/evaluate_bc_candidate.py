"""Import a Kaggle BC ZIP and evaluate fixed comparison and fresh CPU seeds."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile
from utils.checkpoint import load_policy, file_hash
from scripts.eval_closed_loop import evaluate


def run(archive, dataset, destination, output_dir):
    destination=Path(destination);output_dir=Path(output_dir)
    if destination.exists():raise FileExistsError(f'Use a new checkpoint destination: {destination}')
    with zipfile.ZipFile(archive) as z:
        entries={}
        for info in z.infolist():
            if info.is_dir():continue
            p=Path(info.filename)
            if len(p.parts)!=2 or p.parts[0]!='bc' or p.name in entries:
                raise ValueError(f'Unexpected ZIP entry: {p}')
            entries[p.name]=z.read(info)
        if not {'best.pt','policy.onnx','policy.json','metrics.jsonl'} <= entries.keys():
            raise ValueError('ZIP lacks required BC outputs')
    destination.mkdir(parents=True)
    for name,contents in entries.items():(destination/name).write_bytes(contents)
    _,checkpoint=load_policy(destination/'best.pt')
    if checkpoint['policy_type']!='bc' or checkpoint['dataset_sha256']!=file_hash(dataset):
        raise ValueError('Candidate must be BC trained on the expected dataset')
    reports={}
    for label,seed in [('same',5000),('fresh',7000)]:
        report_path=output_dir/f'bc_kaggle_v2_{label}.json'
        if report_path.exists():raise FileExistsError(report_path)
        args=SimpleNamespace(checkpoint=str(destination/'best.pt'),expert=False,threads=2,
                             seed=seed,config=None,gui=False,episodes=50,inject_slip=False,
                             recovery=False,euler_steps=6,execute_horizon=1,output=str(report_path))
        r=evaluate(args)
        reports[label]={k:r[k] for k in ('successes','attempted','success_rate','wilson_95','sampler_ms_p50','sampler_ms_p95')}
    baseline=json.loads((output_dir/'bc_kaggle_eval.json').read_text())
    result={'checkpoint_sha256':file_hash(destination/'best.pt'),
            'dataset_sha256':file_hash(dataset),'reports':reports,
            'baseline_success_rate':baseline['success_rate'],
            'same_seed_improvement_percentage_points':100*(reports['same']['success_rate']-baseline['success_rate']),
            'drawer_80_percent_target_met_on_both_sets':all(r['success_rate']>=.8 for r in reports.values()),
            'scope':'state/oracle context, same 5000-5049 and fresh 7000-7049; no scripted action fallback'}
    (output_dir/'bc_kaggle_v2_comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--archive',required=True)
    p.add_argument('--dataset',default='data/drawer_dagger_v2.h5')
    p.add_argument('--destination',default='weights/bc_kaggle_v2')
    p.add_argument('--output-dir',default='artifacts')
    a=p.parse_args();run(a.archive,a.dataset,a.destination,a.output_dir)
