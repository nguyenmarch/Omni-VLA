"""Matched-seed visual bridge pilot with image/text and task-instruction controls."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from scripts.eval_visual_policy import evaluate
from simulation.vision import FULL_INSTRUCTION,OPEN_INSTRUCTION


def benchmark(args):
    out=Path(args.output)
    if out.exists() and any(out.glob('closed_loop_*.json')):
        raise FileExistsError('Use a new directory to preserve prior benchmark results')
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for label,instruction,ablation in [('normal',FULL_INSTRUCTION,'none'),
            ('drop_rgb',FULL_INSTRUCTION,'drop-image'),('swap_queries',FULL_INSTRUCTION,'swap-text'),
            ('open_only',OPEN_INSTRUCTION,'none')]:
        cfg=SimpleNamespace(grounding=args.grounding,policy=args.policy,instruction=instruction,
            ablation=ablation,episodes=args.episodes,seed=args.seed,perception_every=5,gui=False,
            output=str(out/f'closed_loop_{label}.json'))
        report=evaluate(cfg)
        results.append({k:v for k,v in report.items() if k!='perception_diagnostics'})
    (out/'closed_loop_comparison.json').write_text(json.dumps(results,indent=2)+'\n')
    rows=['# Vision-language CPU pilot','',
          'Matched seeds; frozen BC v2; learned RGB localization + measured simulator depth/calibration; no oracle object/phase policy inputs.',
          'Drop-RGB retains depth/calibration. Query swapping differs from changing the finite task instruction. Open-only has a different success criterion.',
          '', '| Setting | Success | Capture + perception p50/p95 ms | BC p50/p95 ms |',
          '|---|---|---|---|']
    def timing(value):
        return 'n/a' if value is None else f'{value:.2f}'
    for label,r in zip(('normal','drop RGB','swap grounding queries','open-only instruction'),results):
        rows.append(f"| {label} | {r['successes']}/{r['attempted']} | {timing(r['perception_capture_and_infer_ms_p50'])}/{timing(r['perception_capture_and_infer_ms_p95'])} | {timing(r['policy_ms_p50'])}/{timing(r['policy_ms_p95'])} |")
    (out/'comparison.md').write_text('\n'.join(rows)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--grounding',required=True);p.add_argument('--policy',default='weights/bc_kaggle_v2/best.pt')
    p.add_argument('--episodes',type=int,default=5);p.add_argument('--seed',type=int,default=9000)
    p.add_argument('--output',default='artifacts/vision_language_pilot')
    a=p.parse_args()
    if a.episodes<1:p.error('episodes must be positive')
    benchmark(a)
