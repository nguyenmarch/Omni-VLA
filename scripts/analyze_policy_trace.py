"""Summarize final-window approach errors and policy/expert disagreement."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np


def analyze(trace, evaluation, output):
    episodes = defaultdict(list)
    for line in Path(trace).read_text().splitlines():
        row = json.loads(line);episodes[row['seed']].append(row)
    report = json.loads(Path(evaluation).read_text())
    failures=[];phase_rows=defaultdict(list)
    for episode in report['episodes']:
        rows=episodes[episode['seed']]
        for row in rows: phase_rows[row['phase']].append(row)
        if episode['failure_reason'] != 'timeout_approach': continue
        rows=[r for r in rows if r['phase']=='approach']
        tail=rows[-50:]
        actions=np.asarray([r['action'] for r in tail])
        errors=np.asarray([r['position_error_m'] for r in tail])
        failures.append({'seed':episode['seed'],
                         'final_position_error_m':rows[-1]['position_error_m'],
                         'final_rotation_error_rad':rows[-1]['rotation_error_rad'],
                         'late_position_mean_m':errors.mean(0).tolist(),
                         'late_position_std_m':errors.std(0).tolist(),
                         'late_gripper_mean':actions[:,[6,13]].mean(0).tolist(),
                         'late_gripper_closing_fraction':(actions[:,[6,13]]<0).mean(0).tolist(),
                         'late_action_mae':float(np.mean([r['action_mae'] for r in tail]))})
    phase_summary={}
    for phase, rows in phase_rows.items():
        action=np.asarray([r['action'] for r in rows]);expert=np.asarray([r['expert_action'] for r in rows])
        phase_summary[phase]={'steps':len(rows),'action_mae_per_dim':np.abs(action-expert).mean(0).tolist()}
    result={'trace':str(trace),'evaluation':str(evaluation),
            'approach_timeout_count':len(failures),'approach_failures':failures,'phase_action_errors':phase_summary,
            'interpretation':'Final-window errors show behavior; they do not prove the cause. Position transition threshold is 0.012 m per arm.'}
    if failures:
        result['failed_approach_median_final_position_m']=np.median([r['final_position_error_m'] for r in failures],axis=0).tolist()
        result['failed_approach_median_final_rotation_rad']=np.median([r['final_rotation_error_rad'] for r in failures],axis=0).tolist()
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace',required=True);p.add_argument('--evaluation',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();r=analyze(a.trace,a.evaluation,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('approach_failures','phase_action_errors')},indent=2))
