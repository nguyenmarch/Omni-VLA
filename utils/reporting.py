"""Exclusive report writes and episode-level binomial intervals."""
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def wilson(successes, attempted):
    if not attempted:
        return None
    z = 1.96
    p = successes / attempted
    denominator = 1 + z*z / attempted
    center = (p + z*z / (2*attempted)) / denominator
    half = z * (p*(1-p)/attempted + z*z/(4*attempted**2))**.5 / denominator
    return [max(0., center-half), min(1., center+half)]
