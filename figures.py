# -*- coding: utf-8 -*-
"""Figures 2 and 3 and Supplementary Figure S1.

Figure 2   correlation of the landing with sacrum position and with the XCoM at toe-off, per participant
Figure 3   (a) the offset in one cell; (b) both targets calibrated once on the first n steps;
           (c, d) both refitted from a cell's first step, by the number of same-foot steps before
Figure S1  the velocity term on the same-foot steps after a labelled crossing
Reads   results/steps.csv, compare.csv, compare.json      Writes  results/fig2.png, fig3.png, figS1.png, figures.json
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import utf8_stdout, res, save_json, load_json, CELL, WARM as W, OBSTACLE_V1

utf8_stdout()
S = pd.read_csv(res('steps.csv')).sort_values(CELL + ['t_hs']).reset_index(drop=True)
R = pd.read_csv(res('compare.csv'))
CJ = load_json('compare.json')
SUBS = sorted(S['sub'].unique())
FJ = {}
plt.rcParams.update({'font.size': 8.5, 'axes.titlesize': 9, 'axes.labelsize': 8.5,
                     'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7.5})

# ------------------------------------------------------------------ Figure 2
rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        rows.append(dict(sub=sub, ax=ax, pos=np.corrcoef(g[ax + '_pos'].values, p)[0, 1],
                         xcom=np.corrcoef(g[ax + '_xiTO'].values, p)[0, 1]))
C = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
for a, ax, tag in ((axs[0], 'ML', '(a)'), (axs[1], 'AP', '(b)')):
    d = C[C.ax == ax].set_index('sub').loc[SUBS]
    for s in SUBS:
        a.plot([0, 1], [d.loc[s, 'pos'], d.loc[s, 'xcom']], '-', color='0.72', lw=0.9,
               marker='o', ms=2.8, mfc='0.72', mec='0.72')
    a.plot([0, 1], [d.pos.mean(), d.xcom.mean()], '-', color='C3', lw=2.2, marker='o', ms=5.5)
    n_up = int((d.xcom.values > d.pos.values).sum())
    FJ[f'fig2_{ax}'] = dict(pos=d.pos.mean(), xcom=d.xcom.mean(), up=n_up, n=len(d))
    a.set_xticks([0, 1]); a.set_xticklabels(['position only', 'XCoM  $r+\\dot r/\\omega_0$'])
    a.set_xlim(-0.35, 1.35)
    a.set_ylabel('correlation with the landing' if ax == 'ML' else '')
    a.set_title(f'{tag} {ax} at toe-off: $\\Delta r$ = {d.xcom.mean() - d.pos.mean():+.3f}, '
                f'{n_up}/{len(d)} rise', loc='left')
fig.tight_layout(w_pad=1.2)
fig.savefig(res('fig2.png'), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure S1
OB = S[(S.blk != 'V1') | S['sub'].isin(OBSTACLE_V1)]
prof = {a: {k: [] for k in range(0, 7)} for a in ('ML', 'AP')}
per = []
for (sub, blk, foot), g in OB.groupby(CELL):
    g = g.reset_index(drop=True)
    lab = np.zeros(len(g), int)                 # same-foot steps since the nearest preceding crossing
    for c in np.flatnonzero(g.cross.values):
        for k in range(1, 7):
            if c + k < len(g):
                lab[c + k] = min(lab[c + k], k) if lab[c + k] else k
    lab[g.cross.values] = -1
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        d = lambda m: (np.corrcoef(g[ax + '_xiTO'].values[m], p[m])[0, 1]
                       - np.corrcoef(g[ax + '_pos'].values[m], p[m])[0, 1]) if m.sum() > 8 else np.nan
        for k in range(0, 7):
            prof[ax][k].append(dict(sub=sub, v=d(lab == k)))
        if ax == 'ML':
            per.append(dict(sub=sub, base=d(lab == 0), post=d((lab >= 1) & (lab <= 3))))

fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
a = axs[0]
mus = {}
for ax, col in (('ML', 'C0'), ('AP', 'C1')):
    mu = [pd.DataFrame(prof[ax][k]).groupby('sub').v.mean().mean() for k in range(0, 7)]
    mus[ax] = mu
    a.plot(range(0, 7), mu, '-o', color=col, ms=3.5, label=ax)
    a.axhline(mu[0], color=col, ls=':', lw=1)
a.axvspan(0.6, 3.4, color='0.9', zorder=0)
a.set_xlabel('same-foot steps after a labelled crossing (0 = steady)')
a.set_ylabel('XCoM advantage $\\Delta r$')
a.set_title('(a) the advantage after a crossing', loc='left')
a.legend(frameon=False)
FJ['fig2_profile'] = {ax: [float(v) for v in mus[ax]] for ax in mus}
a = axs[1]
d = pd.DataFrame(per).groupby('sub').mean(numeric_only=True).dropna()
for s in d.index:
    a.plot([0, 1], [d.loc[s, 'base'], d.loc[s, 'post']], '-o', color='0.72', lw=0.9, ms=2.8)
a.plot([0, 1], [d.base.mean(), d.post.mean()], '-o', color='C3', lw=2.2, ms=5.5)
a.set_xticks([0, 1]); a.set_xticklabels(['steady', '1–3 same-foot steps\nafter a crossing'])
a.set_xlim(-0.35, 1.35)
a.set_ylabel('mediolateral $\\Delta r$')
n_up = int((d.post.values > d.base.values).sum())
FJ['fig2_post_ML'] = dict(base=d.base.mean(), post=d.post.mean(), up=n_up, n=len(d))
a.set_title(f'(b) per participant, ML: {n_up}/{len(d)} rise', loc='left')
fig.tight_layout(w_pad=1.2)
fig.savefig(res('figS1.png'), dpi=300); plt.close(fig)

# ------------------------------------------------------------------ Figure 3
SCH = ['frozen', 'expand'] + [f'window{N}' for N in (10, 20, 30, 40, 50)]
GRID = [f'{m}_{s}' for m in ('MA', 'LS', 'LR') for s in SCH]
P = R.groupby(['sub', 'ax'])[['A_' + k for k in GRID] + ['B_' + k for k in GRID]].mean().reset_index()
for k in GRID:                                   # the figure uses the numbers of compare.json
    for L in ('A', 'B'):
        for ax in ('ML', 'AP'):
            assert abs(P[P.ax == ax][f'{L}_{k}'].mean() - CJ[f'{L}_{k}'][ax]) < 1e-9
fig, axs = plt.subplots(2, 2, figsize=(7.0, 6.6))
# (a) the offset being estimated, one cell
EX = ('P08', 'V2', 'right')
g = S[(S['sub'] == EX[0]) & (S.blk == EX[1]) & (S.foot == EX[2])].reset_index(drop=True)
m = 1000 * (g.ML_p - g.ML_xiTO).values
x = np.arange(len(m))
run = np.full(len(m), np.nan); c = np.cumsum(m)
for i in range(W, len(m)):
    run[i] = c[i - 1] / i
a = axs[0, 0]
a.plot(x, m, color='0.76', lw=0.7, label='margin at toe-off')
a.plot(x, np.full(len(m), m[:W].mean()), color='tab:red', lw=1.8, label='frozen on 50 steps')
a.plot(x, run, color='tab:blue', lw=1.8, label='updated at every step')
a.plot(x[W:], np.full(len(m) - W, m[W:].mean()), color='k', ls='--', lw=1.1, label='best constant (in sample)')
a.axvline(W, color='k', ls=':', lw=0.9)
lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.45 * (hi - lo))
a.set_xlabel('step within the cell'); a.set_ylabel('mediolateral margin (mm)')
a.set_title('(a) the offset being estimated', loc='left')
a.legend(frameon=True, facecolor='white', edgecolor='none', framealpha=1.0, loc='upper left', ncol=2,
         columnspacing=0.8, handlelength=1.4, borderaxespad=0.2, fontsize=6.0)
FJ['fig3a_cell'] = dict(sub=EX[0], blk=EX[1], foot=EX[2], frozen=float(m[:W].mean()),
                        final_run=float(run[-1]), best=float(m[W:].mean()))
# (b) calibrated once on the first n steps and frozen, scored from step 100
a = axs[0, 1]
CAL = (10, 20, 30, 50, 100)
FJ['fig3b'] = {}
for ax, ls, mk in (('ML', '-', 'o'), ('AP', '--', 's')):
    sf = [CJ[f'cal{n}_{ax}']['SFP'] for n in CAL]
    ws = [CJ[f'cal{n}_{ax}']['WS'] for n in CAL]
    a.plot(CAL, ws, ls=ls, marker=mk, color='0.25', ms=4, lw=1.6, label=f'weights fitted (W&S), {ax}')
    a.plot(CAL, sf, ls=ls, marker=mk, color='C0', ms=4, lw=1.6, label=f'weights from the pendulum (SFP), {ax}')
    for n, s_, w_ in zip(CAL, sf, ws):
        if CJ[f'cal{n}_{ax}']['p_w'] < 0.05:
            a.annotate('*', xy=(n, max(s_, w_)), xytext=(0, 1), textcoords='offset points', ha='center',
                       fontsize=11, color='C0' if s_ < w_ else '0.25')
    FJ['fig3b'][ax] = dict(SFP=sf, WS=ws, p=[CJ[f'cal{n}_{ax}']['p_w'] for n in CAL])
lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.06 * (hi - lo))
a.set_xscale('log'); a.set_xticks(CAL); a.set_xticklabels([str(n) for n in CAL])
a.minorticks_off()
a.set_xlabel('steps in the calibration (then frozen)')
a.set_ylabel('held-out error from step 100 (mm)')
a.set_title('(b) how much calibration each needs', loc='left')
a.legend(frameon=False, fontsize=6.5, loc='upper right')
# (c, d) both refitted from a cell's first step, from 5 steps before
for a, ax, tag in ((axs[1, 0], 'ML', '(c)'), (axs[1, 1], 'AP', '(d)')):
    B = [b for b in CJ['learning'][ax] if b['lo'] >= 5]
    xs = np.arange(len(B))
    a.plot(xs, [b['WS'] for b in B], '-o', color='0.25', ms=3.5, lw=1.5, label='weights fitted (W&S)')
    a.plot(xs, [b['SFP'] for b in B], '-o', color='C0', ms=3.5, lw=1.5, label='weights from the pendulum (SFP)')
    for x_, b in zip(xs, B):
        if b['p_w'] < 0.05:
            a.annotate('*', xy=(x_, max(b['SFP'], b['WS'])), xytext=(0, 1), textcoords='offset points', ha='center',
                       fontsize=11, color='C0' if b['SFP'] < b['WS'] else '0.25')
    a.set_xticks(xs); a.set_xticklabels([b['bin'].replace('-', '–') for b in B], fontsize=7)
    a.set_xlabel("same-foot steps before (both refitted from the first)")
    a.set_ylabel(f'held-out error, {ax} (mm)')
    a.set_title(f'{tag} learning from the first step, {ax}', loc='left')
    lo, hi = a.get_ylim(); a.set_ylim(lo, hi + 0.08 * (hi - lo))
    FJ[f'fig3_learning_{ax}'] = [dict(bin=b['bin'], SFP=b['SFP'], WS=b['WS'], p=b['p_w']) for b in B]
axs[1, 0].legend(frameon=False, fontsize=6.5, loc='upper right')
fig.tight_layout(h_pad=1.4, w_pad=1.0)
fig.savefig(res('fig3.png'), dpi=300); plt.close(fig)
save_json(FJ, 'figures.json')
print('saved fig2.png, fig3.png, figS1.png, figures.json')
