# -*- coding: utf-8 -*-
"""Descriptives (Table 1), the margin, the velocity term, its robustness and negative control, and crossing steps.

Usage   python analyses.py [--steps SUFFIX] [--drop ...] [--keep-subs ...] [--tag NAME] [--quick]
Reads   results/steps{SUFFIX}.csv (and steps_cut4, _cut10, _causal for the robustness rows), step_blocks, legs
Writes  results/analyses{tag}.txt, .json
"""
import os, sys
import numpy as np, pandas as pd
from scipy import stats
from common import utf8_stdout, arg, res, save_json, Report, CELL, WARM as W, is_obstacle_block

utf8_stdout()
RNG = np.random.default_rng(20261009)
PSUF = arg('--steps')
DROP = [x for x in arg('--drop').split(',') if x]
KEEP = [x for x in arg('--keep-subs').split(',') if x]
TAG = arg('--tag', PSUF)
QUICK = '--quick' in sys.argv


def subset(D):
    for d in DROP:
        D = D[~((D['sub'] == d) | ((D['sub'] + '-' + D['blk']) == d))]
    if KEEP:
        D = D[D['sub'].isin(KEEP)]
    return D


S = subset(pd.read_csv(res(f'steps{PSUF}.csv'))).sort_values(CELL + ['t_hs']).reset_index(drop=True)
BL = subset(pd.read_csv(res(f'step_blocks{PSUF}.csv')))
LEGS = pd.read_csv(res(f'legs{PSUF}.csv'))
LEGS = LEGS[LEGS['sub'].isin(S['sub'].unique())]
N = S['sub'].nunique()
NB = S.groupby(['sub', 'blk']).ngroups
J = {'n_participants': int(N), 'n_blocks': int(NB), 'n_steps': int(len(S))}
w = Report()

# ------------------------------------------------------------------ Table 1
w(f"Table 1: descriptives, per participant, then over the {N} participants")
d = S.copy()
d['step_len'] = 1000 * (d.AP_p - d.AP_other)
d['step_wid'] = 1000 * (d.ML_p - d.ML_other).abs()
d['m_hs_ML'] = 1000 * (d.ML_p - d.ML_xiHS)
d['m_hs_AP'] = 1000 * (d.AP_p - d.AP_xiHS)
d['m_to_ML'] = 1000 * (d.ML_p - d.ML_xiTO)
d['m_to_AP'] = 1000 * (d.AP_p - d.AP_xiTO)
step_time = []
for (sub, blk), g in d.groupby(['sub', 'blk']):
    t = np.sort(g.t_hs.values)
    step_time.append(dict(sub=sub, blk=blk, st=float(np.median(np.diff(t)))))
ST = pd.DataFrame(step_time).groupby('sub').st.mean()
P1 = d.groupby('sub').agg(speed=('speed', 'mean'), swing=('Tc', 'mean'), h=('h', 'mean'),
                          w0=('w0', 'mean'), sl=('step_len', 'mean'), sw=('step_wid', 'mean'))
P1['st'] = ST
P1['cad'] = 60.0 / P1.st
for sub, g in d.groupby('sub'):
    for k, col in (('mML', 'm_hs_ML'), ('mAP', 'm_hs_AP'), ('tML', 'm_to_ML'), ('tAP', 'm_to_AP')):
        v = g.groupby('foot')[col].mean()
        P1.loc[sub, k] = float(np.mean(np.abs(v))) if k.endswith('ML') else float(v.mean())
ROWS = [('Gait speed (m/s)', 'speed', 2), ('Cadence (steps/min)', 'cad', 0),
        ('Step time (s)', 'st', 3), ('Swing time (s)', 'swing', 3),
        ('Step length (mm)', 'sl', 0), ('Step width (mm)', 'sw', 0),
        ('Leg length l (m)', 'h', 2), ('w0 (1/s)', 'w0', 2),
        ('Margin at heel strike, ML (magnitude, mm)', 'mML', 0),
        ('Margin at heel strike, AP (mm)', 'mAP', 0),
        ('Margin at toe-off, ML (magnitude, mm)', 'tML', 0),
        ('Margin at toe-off, AP (mm)', 'tAP', 0)]
J['table1'] = {}
for lab, k, nd in ROWS:
    v = P1[k]
    J['table1'][k] = dict(mean=v.mean(), sd=v.std(ddof=1), lo=v.min(), hi=v.max(), nd=nd)
    w(f"  {lab:44s}{v.mean():8.{nd}f} +- {v.std(ddof=1):.{nd}f}   range {v.min():.{nd}f}-{v.max():.{nd}f}")
lr = (LEGS.leg_left - LEGS.leg_right).abs()
J['leg_LR'] = dict(median=1000 * lr.median(), max=1000 * lr.max(), max_sub=LEGS.loc[lr.idxmax(), 'sub'])
J['leg_vs_vertical'] = dict(ratio=float((LEGS.leg / LEGS.h_vert).mean()),
                            diff_mm=float(1000 * (LEGS.leg - LEGS.h_vert).mean()))
w(f"  leg length: left-right difference median {1000 * lr.median():.0f} mm, largest "
  f"{1000 * lr.max():.0f} mm ({J['leg_LR']['max_sub']}); {J['leg_vs_vertical']['diff_mm']:.0f} mm "
  f"longer than the vertical sacrum height on average")

w("")
belt = BL.groupby('blk').belt.mean()
J['belt'] = {b: float(v) for b, v in belt.items()}
J['badfrac'] = dict(mean=100 * BL.badfrac.mean(), max=100 * BL.badfrac.max(),
                    zero=int((BL.badfrac == 0).sum()), n=len(BL))
w(f"  belt speed by block (m/s): " + ", ".join(f"{b} {v:.2f}" for b, v in belt.items()))
w(f"  frames rejected (qc_bad, jump or missing): mean {J['badfrac']['mean']:.1f}%, none in "
  f"{J['badfrac']['zero']} of {J['badfrac']['n']} blocks, largest {J['badfrac']['max']:.1f}%")

# ------------------------------------------------------------------ landing on the XCoM at heel strike
w("")
w("Landing against the XCoM at heel strike, p = c xi + b per cell")
st = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        x, y = g[ax + '_xiHS'].values, g[ax + '_p'].values
        c, b = np.polyfit(x, y, 1)
        r = np.corrcoef(x, y)[0, 1]
        st.append(dict(sub=sub, ax=ax, c=c, R2=r ** 2, sdxi=1000 * x.std(ddof=1)))
STc = pd.DataFrame(st).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    v = STc[STc.ax == ax]
    t, p = stats.ttest_1samp(v.c.values, 1.0)
    cost = abs(1 - v.c.mean()) * v.sdxi.mean()
    J[f'struct_{ax}'] = dict(c=v.c.mean(), c_sd=v.c.std(ddof=1), R2=v.R2.mean(), t=t, p=p,
                             below=int((v.c.values < 1).sum()), n=len(v), cost=cost)
    w(f"  {ax}: c = {v.c.mean():.3f} +- {v.c.std(ddof=1):.3f}, R2 = {v.R2.mean():.3f}; "
      f"against one t({len(v)-1}) = {t:.2f}, p = {p:.4f}, below one in {(v.c.values < 1).sum()}/{len(v)}")
    w(f"      a unit slope misplaces the target by {cost:.1f} mm per SD of the within-cell XCoM")

# ------------------------------------------------------------------ the margin
w("")
w("The margin, by side and by read instant")
for ax in ('ML', 'AP'):
    hs = S.groupby(['sub', 'foot']).apply(
        lambda g: 1000 * (g[ax + '_p'] - g[ax + '_xiHS']).mean(), include_groups=False).unstack()
    to = S.groupby(['sub', 'foot']).apply(
        lambda g: 1000 * (g[ax + '_p'] - g[ax + '_xiTO']).mean(), include_groups=False).unstack()
    t, p = stats.ttest_rel(hs['right'].values, hs['left'].values)
    exc_r = (to['right'] - hs['right']).mean(); exc_l = (to['left'] - hs['left']).mean()
    J[f'margin_{ax}'] = dict(hs_right=hs['right'].mean(), hs_left=hs['left'].mean(), t=t, p=p,
                             to_right=to['right'].mean(), to_left=to['left'].mean(),
                             exc_right=exc_r, exc_left=exc_l, n=len(hs))
    w(f"  {ax} at heel strike: right {hs['right'].mean():+.0f} mm, left {hs['left'].mean():+.0f} mm"
      f" (right-left {hs['right'].mean() - hs['left'].mean():+.0f}, t({len(hs)-1}) = {t:.2f}, p = {p:.4f})")
    w(f"  {ax} at toe-off:     right {to['right'].mean():+.0f} mm, left {to['left'].mean():+.0f} mm;"
      f" swing excursion right {exc_r:+.0f}, left {exc_l:+.0f}")
    if ax == 'ML':
        ssum = hs['right'] + hs['left']
        opp = sum(1 for _, g in S.groupby(['sub', 'blk'])
                  if np.prod([(gg[ax + '_p'] - gg[ax + '_xiHS']).mean() for _, gg in g.groupby('foot')]) < 0)
        J['margin_ML'].update(side_sum=ssum.mean(), opposite=opp, nblocks=NB,
                              notstraddle=int((hs['right'] <= 0).sum() + (hs['left'] >= 0).sum()))
        w(f"      side means sum to {ssum.mean():+.0f} mm ({ssum.min():+.0f} to {ssum.max():+.0f}); "
          f"opposite-signed on the two feet in {opp}/{NB} blocks")
w("")
for ax in ('ML', 'AP'):
    a1 = [np.corrcoef(v[:-1], v[1:])[0, 1] for _, g in S.groupby(CELL)
          for v in [(g[ax + '_p'] - g[ax + '_xiHS']).values] if len(v) > 20]
    a2 = [np.corrcoef(v[:-1], v[1:])[0, 1] for _, g in S.groupby(CELL)
          for v in [(g[ax + '_p'] - g[ax + '_xiTO']).values] if len(v) > 20]
    J[f'acf_{ax}'] = dict(hs=float(np.mean(a1)), to=float(np.mean(a2)))
    w(f"  {ax} lag-1 autocorrelation: margin at heel strike {np.mean(a1):+.3f}, at toe-off {np.mean(a2):+.3f}")

# ------------------------------------------------------------------ prediction from toe-off
w("")
w("Prediction from toe-off: correlation of the landing with sacrum position and with the XCoM")


def corr_table(D):
    rows = []
    for (sub, blk, foot), g in D.groupby(CELL):
        for ax in ('ML', 'AP'):
            p = g[ax + '_p'].values
            rows.append(dict(sub=sub, ax=ax,
                             pos=np.corrcoef(g[ax + '_pos'].values, p)[0, 1],
                             xcom=np.corrcoef(g[ax + '_xiTO'].values, p)[0, 1]))
    return pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()


C = corr_table(S)
for ax in ('ML', 'AP'):
    v = C[C.ax == ax]
    dr = v.xcom.values - v.pos.values
    n = len(v)
    t, p = stats.ttest_rel(v.xcom.values, v.pos.values)
    ci = stats.t.interval(0.95, n - 1, dr.mean(), dr.std(ddof=1) / np.sqrt(n))
    J[f'dr_{ax}'] = dict(pos=v.pos.mean(), xcom=v.xcom.mean(), dr=dr.mean(), lo=ci[0], hi=ci[1],
                         t=t, p=p, better=int((dr > 0).sum()), n=n)
    w(f"  {ax}: r {v.pos.mean():.3f} -> {v.xcom.mean():.3f}  dr = {dr.mean():+.3f} "
      f"(95% CI {ci[0]:+.3f} to {ci[1]:+.3f}), t({n-1}) = {t:.2f}, p = {p:.5f}, higher in {(dr > 0).sum()}/{n}")

w("")
rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        X = np.c_[np.ones(len(g)), g[ax + '_pos'].values, g[ax + '_v'].values]
        bta = np.linalg.lstsq(X, p, rcond=None)[0]
        rows.append(dict(sub=sub, ax=ax, r=np.corrcoef(X @ bta, p)[0, 1]))
FIT = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    v = C[C.ax == ax]; f = FIT[FIT.ax == ax]
    gain_fixed = v.xcom.mean() - v.pos.mean()
    gain_free = f.r.mean() - v.pos.mean()
    t, p = stats.ttest_rel(f.r.values, v.xcom.values)
    J[f'free_{ax}'] = dict(r=f.r.mean(), share=100 * gain_fixed / gain_free, p=p)
    w(f"  {ax}: freely fitted within cell r = {f.r.mean():.3f}; the pendulum's weight gives "
      f"{100 * gain_fixed / gain_free:.0f}% of the gain of a free fit (remaining gap p = {p:.4f})")

w("")
rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        X = np.c_[np.ones(len(g)), g[ax + '_rMS'].values, g[ax + '_vMS'].values]
        bta = np.linalg.lstsq(X, p, rcond=None)[0]
        rows.append(dict(sub=sub, ax=ax, c=bta[2] / bta[1] if bta[1] != 0 else np.nan))
MS = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
J['inv_w0'] = float(np.mean(1 / S.w0))
for ax in ('ML', 'AP'):
    v = MS[MS.ax == ax]
    J[f'midswing_{ax}'] = dict(c=v.c.mean(), ratio=v.c.mean() / np.mean(1 / S.w0))
    w(f"  {ax} read at mid-swing: fitted velocity coefficient {v.c.mean():+.3f} s = "
      f"{v.c.mean() / np.mean(1 / S.w0):.2f} of 1/w0 ({np.mean(1 / S.w0):.3f} s)")

if not QUICK:
    w("")
    w("  Robustness of dr to the processing")
    for tag, fn in (('6 Hz zero-lag (main analysis)', f'steps{PSUF}.csv'),
                    ('4 Hz zero-lag', 'steps_cut4.csv'),
                    ('10 Hz zero-lag', 'steps_cut10.csv'),
                    ('6 Hz causal', 'steps_causal.csv')):
        if not os.path.exists(res(fn)):
            continue
        Cc = corr_table(subset(pd.read_csv(res(fn))).sort_values(CELL + ['t_hs']))
        vals = []
        for ax in ('ML', 'AP'):
            v = Cc[Cc.ax == ax]
            dr = v.xcom.values - v.pos.values
            t, p = stats.ttest_rel(v.xcom.values, v.pos.values)
            J[f'rob_{fn}_{ax}'] = dict(dr=dr.mean(), better=int((dr > 0).sum()), n=len(v), p=p)
            vals.append(f"{ax} {dr.mean():+.3f} ({(dr > 0).sum()}/{len(v)}, p = {p:.4f})")
        w(f"    {tag:38s}" + "   ".join(vals))
    w("")
    for blk in ('V1', 'V2', 'V3'):
        Cc = corr_table(S[S.blk == blk])
        vals = []
        for ax in ('ML', 'AP'):
            v = Cc[Cc.ax == ax]
            dr = v.xcom.values - v.pos.values
            t, p = stats.ttest_rel(v.xcom.values, v.pos.values)
            J[f'blk_{blk}_{ax}'] = dict(dr=dr.mean(), better=int((dr > 0).sum()), n=len(v), p=p)
            vals.append(f"{ax} {dr.mean():+.3f} ({(dr > 0).sum()}/{len(v)}, p = {p:.4f})")
        w(f"    block {blk} alone{'':22s}" + "   ".join(vals))

    w("")
    NPERM = 2000
    for ax in ('ML', 'AP'):
        real = (C[C.ax == ax].xcom.values - C[C.ax == ax].pos.values).mean()
        null = np.empty(NPERM)
        cells = [(g[ax + '_pos'].values, g[ax + '_v'].values, g[ax + '_p'].values, g.w0.values,
                  g['sub'].iat[0]) for _, g in S.groupby(CELL)]
        subs = np.array([c[4] for c in cells])
        usub = np.unique(subs)
        base = np.array([np.corrcoef(r0, p)[0, 1] for r0, v0, p, w0, s in cells])
        for k in range(NPERM):
            dd = np.array([np.corrcoef(r0 + RNG.permutation(v0) / w0, p)[0, 1]
                           for r0, v0, p, w0, s in cells]) - base
            null[k] = np.mean([dd[subs == s].mean() for s in usub])
        pp = (1 + (null >= real).sum()) / (NPERM + 1)
        J[f'negctrl_{ax}'] = dict(real=real, null_mean=null.mean(), p95=np.percentile(null, 95), p=pp)
        w(f"  {ax} negative control (velocity shuffled within cell, {NPERM} permutations): "
          f"observed {real:+.3f}, null mean {null.mean():+.3f}, 95th percentile {np.percentile(null, 95):+.3f}, "
          f"p = {pp:.4f}")

# ------------------------------------------------------------------ crossing steps
w("")
w("Obstacle-crossing steps and the steps after them")
OB = S[is_obstacle_block(S)]
V1 = S[~is_obstacle_block(S)]
J['cross'] = dict(n=int(OB.cross.sum()), of=len(OB), pct=100 * OB.cross.mean(),
                  clear_reg=1000 * OB[~OB.cross].clear.median(), clear_cross=1000 * OB[OB.cross].clear.median(),
                  v1_n=int(V1.cross.sum()), v1_of=len(V1), v1_pct=100 * V1.cross.mean())
w(f"  crossing steps labelled: {J['cross']['n']} of {len(OB)} ({J['cross']['pct']:.1f}%) in the obstacle "
  f"blocks; clearance {J['cross']['clear_reg']:.0f} mm on other steps, {J['cross']['clear_cross']:.0f} mm on crossings")
w(f"  labelled in the obstacle-free blocks: {J['cross']['v1_n']} of {len(V1)} ({J['cross']['v1_pct']:.2f}%)")

rows = []
for (sub, blk, foot), g in OB.groupby(CELL):
    for ax in ('ML', 'AP'):
        cr = g.cross.values
        if cr.sum() < 5 or (~cr).sum() < 5:
            continue
        p = g[ax + '_p'].values
        dfn = lambda k: (np.corrcoef(g[ax + '_xiTO'].values[k], p[k])[0, 1]
                         - np.corrcoef(g[ax + '_pos'].values[k], p[k])[0, 1])
        rows.append(dict(sub=sub, ax=ax, dr_reg=dfn(~cr), dr_cross=dfn(cr)))
CR = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    v = CR[CR.ax == ax]
    t, p = stats.ttest_rel(v.dr_cross.values, v.dr_reg.values)
    J[f'cross_step_{ax}'] = dict(reg=v.dr_reg.mean(), cross=v.dr_cross.mean(), t=t, p=p, n=len(v))
    w(f"  {ax} dr on the crossing step: other steps {v.dr_reg.mean():+.3f}, crossing "
      f"{v.dr_cross.mean():+.3f} (t({len(v)-1}) = {t:.2f}, p = {p:.4f})")

rows = []
for (sub, blk, foot), g in OB.groupby(CELL):
    g = g.reset_index(drop=True)
    cr = np.flatnonzero(g.cross.values)
    lab = np.zeros(len(g), int)                 # same-foot steps since the nearest preceding crossing
    for c in cr:
        for k in range(1, 7):
            if c + k < len(g):
                lab[c + k] = min(lab[c + k], k) if lab[c + k] else k
    lab[g.cross.values] = -1
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values
        dfn = lambda k: (np.corrcoef(g[ax + '_xiTO'].values[k], p[k])[0, 1]
                         - np.corrcoef(g[ax + '_pos'].values[k], p[k])[0, 1]) if k.sum() > 8 else np.nan
        rows.append(dict(sub=sub, ax=ax, base=dfn(lab == 0), post=dfn((lab >= 1) & (lab <= 3))))
PC = pd.DataFrame(rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    v = PC[PC.ax == ax].dropna()
    t, p = stats.ttest_rel(v.post.values, v.base.values)
    J[f'post_{ax}'] = dict(base=v.base.mean(), post=v.post.mean(), t=t, p=p,
                           higher=int((v.post.values > v.base.values).sum()), n=len(v))
    w(f"  {ax} the 1-3 same-foot steps after a crossing: {v.base.mean():+.3f} -> {v.post.mean():+.3f} "
      f"(t({len(v)-1}) = {t:.2f}, p = {p:.4f}, higher in {(v.post.values > v.base.values).sum()}/{len(v)})")

w.save(f'analyses{TAG}.txt')
save_json(J, f'analyses{TAG}.json')
