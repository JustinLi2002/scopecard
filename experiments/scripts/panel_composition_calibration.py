"""Null calibration of the task-bootstrap intervals of the panel-composition study (PANEL_COMPOSITION_DESIGN.md, Section 11):
40 scoring splits drawn independently of the votes (each model passes each task with its Complete pass rate), 200 bootstrap
replicates each. Reads no Instruct outcome.

Usage (from experiments/): python -E -P -X utf8 scripts/panel_composition_calibration.py
"""
import pathlib
import sys

import numpy as np

script = pathlib.Path(__file__).resolve().parent / 'panel_composition.py'
src = script.read_text(encoding='utf-8')
marker = '# ---------------------------------------------------------------------------------------------------------------\n# Checks'
head = src.split(marker)[0]
sys.argv = ['calibration', '--check']          # the study script then loads the Complete table only
g = {'__file__': str(script), '__name__': 'calibration'}
exec(compile(head, 'panel_composition_head', 'exec'), g)
Yc = g['mats']['complete']
rates = Yc.mean(axis=1, keepdims=True)
N_DRAWS, N_BOOT = 40, 200
w = g['weights_for'](N_BOOT)
T8, T8_sd, T8_excl, fam_excl, line_ratios, line_excl = [], [], 0, 0, [], 0
for i in range(N_DRAWS):
    gen = g['rng_for'](f'panel_composition/check/calibration/{i}')
    Z = (gen.random(Yc.shape) < rates).astype(np.float64)
    res, _ = g['run'](Yc, Z, g['B_FLOOR'], g['N_CONTROLS'], w, 'calibration')
    keys = list(res)
    f8 = [k for k in keys if k[0] == 8]
    pooled = np.mean([res[k]['line'] for k in f8], axis=0)
    lo, hi = np.percentile(pooled[1:], [2.5, 97.5])
    T8.append(pooled[0])
    T8_sd.append(pooled[1:].std(ddof=1))
    T8_excl += int(lo > 0 or hi < 0)
    boot = np.array([res[k]['line'][1:] for k in keys])
    est = np.array([res[k]['line'][0] for k in keys])
    sd = boot.std(axis=1, ddof=1)
    q = float(np.percentile(np.max(np.abs(boot - est[:, None]) / sd[:, None], axis=0), 95))
    fam_excl += int(np.any(np.abs(est) > q * sd))
    line_ratios.extend(est / sd)
    for k in keys:
        a, b = np.percentile(res[k]['line'][1:], [2.5, 97.5])
        line_excl += int(a > 0 or b < 0)
    print(f'draw {i}: T(8) {100 * pooled[0]:+.3f} points, bootstrap SD {100 * T8_sd[-1]:.3f}, ratio {pooled[0] / T8_sd[-1]:+.2f}, '
          f'interval {"excludes" if (lo > 0 or hi < 0) else "covers"} 0; max |t| over 8 contrasts {np.max(np.abs(est / sd)):.2f} '
          f'(q {q:.2f})', flush=True)
T8, T8_sd, line_ratios = np.array(T8), np.array(T8_sd), np.array(line_ratios)
print()
print(f'{N_DRAWS} null draws, {N_BOOT} bootstrap replicates each')
print(f'T(8): mean {100 * T8.mean():+.3f} points, SD over draws {100 * T8.std(ddof=1):.3f}, mean bootstrap SD {100 * T8_sd.mean():.3f}, '
      f'ratio of the two {T8.std(ddof=1) / T8_sd.mean():.2f}; percentile interval excludes 0 in {T8_excl} of {N_DRAWS} (nominal 5%)')
print(f'T(8)/bootstrap SD over draws: SD {np.std(T8 / T8_sd, ddof=1):.2f} (nominal 1)')
print(f'per-contrast estimate/bootstrap SD: SD {line_ratios.std(ddof=1):.2f} (nominal 1), mean {line_ratios.mean():+.2f}; percentile '
      f'intervals exclude 0 in {line_excl} of {len(line_ratios)}; simultaneous intervals: any of 8 excludes 0 in {fam_excl} of {N_DRAWS} '
      f'(nominal 5%)')
