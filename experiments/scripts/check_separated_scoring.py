"""Independent reconstruction of the primary result from saved plan and inputs.

Does not import the analysis implementation. Writes an independent check only.
"""
from pathlib import Path
import csv
import hashlib
import json

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/results/separated_scoring'
RAW = ROOT / 'experiments/results/subsets/raw/huggingface'
plan = json.loads((OUT / 'panel_plan.json').read_text(encoding='utf-8'))
names, tasks = plan['models'], plan['tasks']
def matrix(mode):
    rows = {r['Model']: r for r in map(json.loads, (RAW / f'bigcodebench_perf_{mode}.jsonl').read_text(encoding='utf-8').splitlines())}
    return np.array([[int(rows[n][t]) for t in tasks] for n in names], dtype=np.int64)

source, target = matrix('complete'), matrix('instruct')
archive = np.load(OUT / 'bootstrap.npz', allow_pickle=False)
weighted = np.vstack([np.ones(len(tasks)), archive['task_weights'][:20]]).astype(float)
observed = []
for group in plan['panels']['8']:
    panel = group['panel']
    for item in group['members']:
        m, u, r = item['m'], item['u'], item['r']
        # Construct new panels explicitly, not by subtracting/adding vote counts.
        self_panel = [i for i in panel if i != m] + [r]
        reference_panel = [i for i in panel if i != u] + [r]
        assert len(set(self_panel)) == len(set(reference_panel)) == 8
        selected = source[self_panel].mean(axis=0) < .5
        reference = source[reference_panel].mean(axis=0) < .5
        scores = np.vstack([source[m], target[m], target[item['outside']].mean(0)])
        left = (weighted[:, selected] @ scores[:, selected].T)/weighted[:, selected].sum(axis=1, keepdims=True)
        right = (weighted[:, reference] @ scores[:, reference].T)/weighted[:, reference].sum(axis=1, keepdims=True)
        delta = left-right
        observed.append(np.column_stack([delta, delta[:, 1]-delta[:, 2]]))
observed = np.array(observed)
mean = observed.mean(axis=0)
index = next(i for i, (d,p) in enumerate(zip(archive['directions'], archive['sizes'])) if d == 'complete_to_instruct' and p == 8)
expected = archive['values'][index, :21]
error = float(abs(mean-expected).max())
assert error < 1e-12
with (OUT / 'events.tsv').open(encoding='utf-8') as f:
    rows = [r for r in csv.DictReader(f, delimiter='\t') if r['direction']=='complete_to_instruct' and r['p']=='8']
point = np.array([[float(r[k]) for k in ['source','target','outside','excess']] for r in rows])
event_error = float(abs(observed[:,0,:]-point).max())
assert event_error < 1e-12
checks = {'primary_events_checked': len(rows), 'bootstrap_replicates_checked': 20,
          'max_mean_difference': error, 'max_event_difference': event_error,
          'primary_excess': float(mean[0,3]), 'method': 'Explicitly rebuild each panel; independent code; no analysis-module imports'}
(OUT / 'independent_check.json').write_text(json.dumps(checks, indent=2), encoding='utf-8')
print(json.dumps(checks))
