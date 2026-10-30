# -*- coding: utf-8 -*-
"""Supplementary numbers that the other scripts do not compute.

1  the XCoM propagated to the landing through the pendulum's closed form, xi(T) = s + (xi_TO - s) exp(w0 T),
   s the stance foot at toe-off and T the swing time, against the XCoM at toe-off (Supplementary Material S2)
2  the window length of the running mean (Supplementary Table S3)
3  the velocity term block by block (Supplementary Table S1)
Reads   results/steps.csv, compare.json      Writes  results/supplement.json
"""
import numpy as np, pandas as pd
from scipy import stats
from common import utf8_stdout, res, save_json, load_json, CELL, WARM as W

utf8_stdout()
S = pd.read_csv(res('steps.csv')).sort_values(CELL + ['t_hs']).reset_index(drop=True)
J = {}

# 1. propagated XCoM
rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    T = (g.t_hs - g.t_to).values
    grow = np.exp(g.w0.values * T)
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; s = g[ax + '_stanceTO'].values
        prop = s + (xi - s) * grow
        rows.append(dict(sub=sub, ax=ax, inst=np.corrcoef(xi, p)[0, 1], prop=np.corrcoef(prop, p)[0, 1],
                         grow=grow.mean()))
P = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    d = P[P.ax == ax]
    J[f'prop_{ax}'] = dict(inst=d.inst.mean(), prop=d.prop.mean(), dr=(d.prop - d.inst).mean(),
                           worse=int((d.prop < d.inst).sum()), n=len(d), p=stats.ttest_rel(d.prop, d.inst).pvalue)
J['prop_growth'] = float(P.grow.mean())
print('propagated XCoM:', {k: v for k, v in J.items() if k.startswith('prop')})

# 2. window length of the running mean
NS = (5, 10, 20, 40, 80, 160, None)
acc = []
for (sub, blk, foot), g in S.groupby(CELL):
    if len(g) <= W + 20:
        continue
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = p - xi
        rec = dict(sub=sub, ax=ax)
        for N in NS:
            f = np.array([m[(0 if N is None else max(0, i - N)):i].mean() for i in range(W, len(m))])
            rec['all' if N is None else str(N)] = 1000 * np.sqrt(np.mean((m[W:] - f) ** 2))
        acc.append(rec)
M = pd.DataFrame(acc).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    v = [M[M.ax == ax][k].mean() for k in ['5', '10', '20', '40', '80', '160', 'all']]
    J[f'memory_{ax}'] = v
    J[f'memory_span_{ax}'] = 100 * (max(v) / min(v) - 1)
print('window length:', {k: v for k, v in J.items() if k.startswith('memory')})
C = load_json('compare.json')
assert all(abs(J[f'memory_{ax}'][-1] - C['B_MA_expand'][ax]) < 1e-9 for ax in ('ML', 'AP'))   # 'all' is the SFP

# 3. the velocity term by block
rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        rows.append(dict(sub=sub, blk=blk, ax=ax, dr=np.corrcoef(g[ax + '_xiTO'], p)[0, 1]
                         - np.corrcoef(g[ax + '_pos'], p)[0, 1]))
B = pd.DataFrame(rows).groupby(['sub', 'blk', 'ax']).dr.mean().reset_index()
for blk in ('V1', 'V2', 'V3'):
    for ax in ('ML', 'AP'):
        d = B[(B.blk == blk) & (B.ax == ax)].dr
        J[f'block_{blk}_{ax}'] = dict(dr=d.mean(), up=int((d > 0).sum()), n=len(d), p=stats.ttest_1samp(d, 0).pvalue)
print('by block:', {k: v for k, v in J.items() if k.startswith('block')})
save_json(J, 'supplement.json')
