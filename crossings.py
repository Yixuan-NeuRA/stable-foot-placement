# -*- coding: utf-8 -*-
"""Prediction error of the SFP and the Wang & Srinivasan regression on obstacle-crossing steps.

Per cell and axis, y = p - s (landing about the stance foot), x1 = r - s, x2 = v (sacrum position and velocity at
toe-off). SFP: y = c + x1 + x2/w0 with c the running mean; W&S: y = c + b1 x1 + b2 x2 by least squares. Both
refitted on every preceding step (crossings included) and scored from step 50.
Step groups: crossing (kinematic label), the next one to three same-foot steps, other steps of obstacle blocks,
steps of obstacle-free blocks. Root-mean-square error per participant, then averaged; signed error is the actual
landing minus the forecast.
Reads   results/steps.csv      Writes  results/crossings.txt, .json
"""
import numpy as np, pandas as pd
from scipy import stats
from common import utf8_stdout, res, save_json, Report, CELL, WARM as W

utf8_stdout()
S = pd.read_csv(res('steps.csv')).sort_values(CELL + ['t_hs']).reset_index(drop=True)
w = Report()

CELLS = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        y = (g[ax + '_p'] - g[ax + '_stanceTO']).values
        x1 = (g[ax + '_pos'] - g[ax + '_stanceTO']).values
        x2 = g[ax + '_v'].values
        CELLS.append(dict(sub=sub, ax=ax, y=y, Z=np.c_[np.ones(len(y)), x1, x2],
                          b0=np.r_[0.0, 1.0, 1.0 / g.w0.iloc[0]], idx=g.index.values))


def forecasts(c, method):
    """Refitted on every step so far: 'SFP' (weights fixed at the pendulum's) or 'WS' (least squares)."""
    y, Z, b0 = c['y'], c['Z'], c['b0']
    n = len(y)
    f = np.full(n, np.nan)
    if method == 'SFP':
        m = y - Z[:, 1:] @ b0[1:]
        cs = np.r_[0, np.cumsum(m)]
        for i in range(1, n):
            f[i] = Z[i, 1:] @ b0[1:] + cs[i] / i
        return f
    outer = Z[:, :, None] * Z[:, None, :]
    cO = np.r_[np.zeros((1, 3, 3)), np.cumsum(outer, 0)]
    cB = np.r_[np.zeros((1, 3)), np.cumsum(Z * y[:, None], 0)]
    for i in range(1, n):
        f[i] = Z[i] @ (np.linalg.pinv(cO[i]) @ cB[i])
    return f


E = {}
for c in CELLS:
    for name in ('SFP', 'WS'):
        e = 1000 * (c['y'] - forecasts(c, name))
        e[:W] = np.nan
        E.setdefault((name, c['ax']), pd.Series(np.nan, index=S.index))
        E[(name, c['ax'])].loc[c['idx']] = e

S['after'] = False
for _, g in S.groupby(CELL):
    cr = g['cross'].values.astype(bool)
    aft = np.zeros(len(g), bool)
    for k in np.where(cr)[0]:
        aft[k + 1:k + 4] = True
    S.loc[g.index, 'after'] = aft & ~cr
rate = S.groupby(['sub', 'blk'])['cross'].mean()
obst_blocks = set(rate[(rate.index.get_level_values('blk') != 'V1') | (rate > 0.03)].index)
S['obst'] = [(a, b) in obst_blocks for a, b in zip(S['sub'], S['blk'])]
w(f"steps in obstacle blocks {S.obst.sum()} (labelled crossings {int(S[S.obst].cross.sum())}), "
  f"in obstacle-free blocks {int((~S.obst).sum())} (labelled {int(S[~S.obst].cross.sum())})")
cross = S.cross.astype(bool)
GROUPS = {
    'crossing (kinematic)': S.obst & cross,
    'after 1-3 steps': S.obst & S.after & ~cross,
    'other, obstacle blocks': S.obst & ~cross & ~S.after,
    'obstacle-free blocks': ~S.obst,
}


def per_sub(mask, name, ax, stat):
    e = E[(name, ax)][mask]
    d = pd.DataFrame({'sub': S.loc[mask, 'sub'], 'e': e}).dropna()
    if stat == 'rmse':
        return d.groupby('sub').e.apply(lambda v: np.sqrt(np.mean(v ** 2)))
    return d.groupby('sub').e.mean()


JX = {}
for ax in ('ML', 'AP'):
    for gname, mask in GROUPS.items():
        for m in ('SFP', 'WS'):
            JX[f'{ax}|{gname}|{m}|rmse'] = float(per_sub(mask, m, ax, 'rmse').mean())
            JX[f'{ax}|{gname}|{m}|bias'] = float(per_sub(mask, m, ax, 'bias').mean())
    for m in ('SFP', 'WS'):
        a_ = per_sub(GROUPS['crossing (kinematic)'], m, ax, 'rmse')
        b_ = per_sub(GROUPS['other, obstacle blocks'], m, ax, 'rmse')
        k_ = a_.index.intersection(b_.index)
        JX[f'{ax}|ratio|{m}'] = float((a_[k_] / b_[k_]).mean())
save_json(JX, 'crossings.json')

for ax in ('ML', 'AP'):
    w('')
    w(f"{ax}: held-out error by step group (participant means; refitted on all preceding steps, scored from step 50)")
    w(f"  {'group':26s} {'steps':>6s} {'n':>3s} | {'SFP':>6s} {'W&S':>6s} | {'SFP - W&S':>20s} | signed SFP/W&S")
    for gname, mask in GROUPS.items():
        r = {m: per_sub(mask, m, ax, 'rmse') for m in ('SFP', 'WS')}
        b = {m: per_sub(mask, m, ax, 'bias') for m in ('SFP', 'WS')}
        common = r['SFP'].index.intersection(r['WS'].index)
        x, y_ = r['SFP'][common], r['WS'][common]
        p = stats.wilcoxon(x, y_).pvalue if len(common) > 5 and np.any(x != y_) else np.nan
        nsteps = int(E[('SFP', ax)][mask].notna().sum())
        w(f"  {gname:26s} {nsteps:6d} {len(common):3d} | {x.mean():6.1f} {y_.mean():6.1f} | "
          f"{(x - y_).mean():+5.1f} W&S better {int((y_ < x).sum()):2d}/{len(common)} p={p:.3f} | "
          f"{b['SFP'][common].mean():+6.1f}/{b['WS'][common].mean():+6.1f}")
w('')
w('Error on crossing steps relative to the other steps of obstacle blocks (participant means of the RMSE ratio)')
for ax in ('ML', 'AP'):
    w(f"  {ax}: SFP x{JX[f'{ax}|ratio|SFP']:.2f}, W&S x{JX[f'{ax}|ratio|WS']:.2f}")
w.save('crossings.txt')
