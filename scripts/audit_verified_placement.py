"""Post-decision H4 physics/trace audit. Oracle values are scoring-only."""
import argparse
import json
from pathlib import Path

import numpy as np

from scripts.evaluate_visual_suite import validate_inputs, validate_report, validate_trace
from utils.checkpoint import file_hash
from utils.reporting import write_json


def audit_variant(report, rows):
    options=report['effective_options']['controller_options']
    calibrated=options['place_contact_mode']=='calibrated'
    violations=[]
    transitions=[]
    command_bounds=0
    for row in rows:
        if not row.get('executed') or not calibrated:continue
        phase=row['phase']
        if phase is None or not 5<=phase<=8:continue
        action=np.asarray(row['action']).reshape(2,7)
        before=np.asarray(row['gripper_fingers_m'])
        after=np.asarray(row['gripper_fingers_after_step_m'])
        ee_before=np.asarray(row['ee_position_m'])
        ee_after=np.asarray(row['ee_position_after_step_m'])
        current=row['release'];updated=row['release_after_update']
        def require(condition,why):
            if not condition:violations.append({'seed':row['seed'],'step':row['step'],'reason':why})
        require(action[1,6]==-1.,'right_grip_command_lost')
        if phase==5:require(action[0,6]==-1.,'opened_before_authorization')
        if phase in (7,8):
            require(current['open_verified'] and np.all(before[0]>options['release_open_threshold_m']),
                    'retreated_before_actual_open')
        if phase==8 and np.linalg.norm(action[1,:3])>1e-8:
            require(current['substate']=='retreat_verified' and current['open_verified']
                    and ee_before[0,2]>=options['release_retreat_height_m']-options['release_retreat_tolerance_m']
                    and np.all(before[0]>options['release_open_threshold_m']),'closed_without_clearance')
        servo=phase in (6,7,8) or current['substate'] in ('collision_hold','collision_correction')
        if servo:
            arms=action[:1] if phase==8 else action
            require(np.max(np.linalg.norm(arms[:,:3]*.015/report['action_dt_s'],axis=1))
                    <=options['release_max_speed_m_s']+1e-6,'linear_command_bound')
            require(np.max(np.linalg.norm(arms[:,3:6]*.08/report['action_dt_s'],axis=1))
                    <=options['release_max_angular_speed_rad_s']+1e-6,'angular_command_bound')
            command_bounds+=1
        transition=row.get('transition')
        if not transition:continue
        transition={**transition,'seed':row['seed'],'guard_snapshot':updated['guards']}
        transitions.append(transition)
        if transition['from']=='place':
            require(updated['authorization'] in ('verified_valid_pose','calibrated_support'), 'missing_release_authorization')
            require(updated['guards'].get('valid_pose') and updated['guards'].get('low_speed'), 'release_guard_false')
            require(np.all(after[0]<.032),'already_open_at_authorization')
        elif transition['from']=='release':
            require(np.all(after[0]>options['release_open_threshold_m']) and updated['open_verified'], 'unverified_aperture')
            require(all(updated['guards'].get(k) for k in ('new_pixels_after_open','unloaded','separated')), 'unverified_release_load_or_vision')
        elif transition['from']=='retreat':
            require(ee_after[0,2]>=options['release_retreat_height_m']-options['release_retreat_tolerance_m'], 'low_retreat_transition')
            require(updated['guards'].get('retreat_clear') and updated['guards'].get('right_grasp_held'),'retreat_guard_false')
            require(np.all(after[0]>options['release_open_threshold_m']),'closed_gripper_at_clearance')
    audits=[e['control_audit'] for e in report['episodes']]
    return {'calibrated':calibrated,'episodes':[{'seed':e['seed'],'success':e['success'],'steps':e['steps'],
            'failure_reason':e['failure_reason']} for e in report['episodes']],
            'violations':violations,'bounded_servo_ticks':command_bounds,'transitions':transitions,
            'force_releases':sum(t['reason']=='calibrated_support' for t in transitions),
            'pose_releases':sum(t['reason']=='verified_valid_pose' for t in transitions),
            'false_force_releases':sum(a['false_contact_triggers'] for a in audits),
            'release_commands_outside_envelope':sum(a['release_commands_outside_env_envelope'] for a in audits),
            'actual_openings_outside_envelope':sum(a['actual_openings_outside_env_envelope'] for a in audits)}


def audit_suite(root):
    root=Path(root)
    plan,digest=validate_inputs(root)
    reports={};traces={};results={}
    for variant in plan['variants']:
        results[variant]={}
        for profile in plan['sets']:
            path=root/variant/(profile+'.json');trace=path.with_suffix('.jsonl')
            report=json.loads(path.read_text())
            validate_report(plan,profile,report,digest,variant);validate_trace(trace,report)
            rows=[json.loads(line) for line in trace.open()]
            reports[variant,profile]=report;traces[variant,profile]=rows
            results[variant][profile]=audit_variant(report,rows)
    prefix={}
    # F and B must execute the same trajectory before H4; TF and T before H4.
    for candidate,reference in (('F','B'),('TF','T')):
        if candidate not in plan['variants'] or reference not in plan['variants']:continue
        for profile,seeds in plan['sets'].items():
            for seed in seeds:
                old=[r for r in traces[reference,profile] if r['seed']==seed]
                new=[r for r in traces[candidate,profile] if r['seed']==seed]
                count=0
                for a,b in zip(old,new):
                    if a['phase'] is not None and a['phase']>=5:break
                    if b['phase'] is not None and b['phase']>=5:break
                    for key in ('step','phase','action','ee_position_after_step_m'):
                        if a.get(key)!=b.get(key):raise AssertionError(f'Pre-H4 drift {candidate}/{seed}/{key}/{a["step"]}')
                    count+=1
                prefix[f'{candidate}/{profile}/{seed}']=count
    active=[r for profiles in results.values() for r in profiles.values() if r['calibrated']]
    passed=bool(active and all(not r['violations'] and not r['false_force_releases']
        and not r['release_commands_outside_envelope'] and not r['actual_openings_outside_envelope'] for r in active)
        and any(any(t['reason']=='retreat_clear_verified' for t in r['transitions']) for r in active))
    result={'h4_physics_guard_gate_passed':passed,'source_sha256':plan['source_sha256'],
            'plan_sha256':digest,'variants':results,'identical_pre_h4_prefix_ticks':prefix,
            'report_sha256':{f'{v}/{p}':file_hash(root/v/(p+'.json')) for v,p in reports},
            'scope':'Development guard audit on the original scene. Does not establish IID/OOD population success or wall-clock realtime.'}
    write_json(root/'h4_audit.json',result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite',required=True)
    args=p.parse_args();result=audit_suite(args.suite)
    print(json.dumps({'h4_physics_guard_gate_passed':result['h4_physics_guard_gate_passed'],
                      'identical_pre_h4_prefix_ticks':result['identical_pre_h4_prefix_ticks']}))


if __name__=='__main__':main()
