# -*- coding: utf-8 -*-
"""The foot placement error of Lang et al. computed in these data, against the SPE.

Lang et al. regressed step width (ML) and step length (AP) on the CoM position relative to the stance foot and the
CoM velocity, in sample, at mid-swing and at terminal swing; the foot placement error (FPE) is the SD of the
residual. Here, per participant x block x side cell, then averaged per participant:
  y  = landing - stance foot at heel strike
  mid-swing       x1 = sacrum - stance foot at mid-swing,   x2 = sacrum velocity at mid-swing
  terminal swing  x1 = sacrum - stance foot at heel strike, x2 = sacrum velocity at heel strike
  toe-off         x1 = sacrum - stance foot at toe-off,     x2 = sacrum velocity at toe-off
The SPE (running-mean offset, every step forecast from the steps before it, scored from step 50) is summarised
as its SD per cell. Spearman correlations across participants.
Reads   results/steps.csv      Writes  results/lang_fpe.txt, .json
"""
import numpy as np, pandas as pd
from scipy import stats
from common import utf8_stdout, res, save_json, Report, CELL, WARM as W

utf8_stdout()
S = pd.read_csv(res('steps.csv')).sort_values(CELL + ['t_hs']).reset_index(drop=True)
w = Report()


def lang(y, x1, x2):
    """In-sample regression on demeaned data: SD of the residual and R2."""
    Y = y - y.mean(); X = np.c_[x1 - x1.mean(), x2 - x2.mean()]
    b = np.linalg.lstsq(X, Y, rcond=None)[0]
    pred = X @ b
    return np.std(Y - pred, ddof=3), np.var(pred) / np.var(Y)


def sfp_spe(g, ax):
    """SPE of every step from the running mean of the earlier steps' margins; NaN before step W."""
    y = (g[ax + '_p'] - g[ax + '_stanceTO']).values
    x = (g[ax + '_pos'] - g[ax + '_stanceTO']).values + g[ax + '_v'].values / g.w0.values
    m = y - x
    cs = np.r_[0, np.cumsum(m)]
    e = np.full(len(y), np.nan)
    for i in range(1, len(y)):
        e[i] = m[i] - cs[i] / i
    e[:W] = np.nan
    return e


def ws_oos(g, ax):
    """Wang & Srinivasan regression at toe-off, refitted on all earlier steps."""
    y = (g[ax + '_p'] - g[ax + '_stanceTO']).values
    Z = np.c_[np.ones(len(y)), (g[ax + '_pos'] - g[ax + '_stanceTO']).values, g[ax + '_v'].values]
    e = np.full(len(y), np.nan)
    for i in range(W, len(y)):
        b = np.linalg.lstsq(Z[:i], y[:i], rcond=None)[0]
        e[i] = y[i] - Z[i] @ b
    return e


def run(sel_name, keep):
    rows = []
    for (sub, blk, foot), g0 in S.groupby(CELL):
        for ax in ('ML', 'AP'):
            spe = sfp_spe(g0, ax); wse = ws_oos(g0, ax)
            k = keep(g0).values
            g = g0[k]
            if len(g) < 30:
                continue
            y = (g[ax + '_p'] - g[ax + '_other']).values
            r = dict(sub=sub, blk=blk, foot=foot, ax=ax, n=len(g))
            r['FPE_mid'], r['R2_mid'] = lang(y, (g[ax + '_rMS'] - g[ax + '_stanceMS']).values, g[ax + '_vMS'].values)
            r['FPE_term'], r['R2_term'] = lang(y, (g[ax + '_posHS'] - g[ax + '_other']).values, g[ax + '_vHS'].values)
            r['FPE_to'], r['R2_to'] = lang(y, (g[ax + '_pos'] - g[ax + '_stanceTO']).values, g[ax + '_v'].values)
            s_ = spe[k]; s_ = s_[~np.isnan(s_)]
            e_ = wse[k]; e_ = e_[~np.isnan(e_)]
            r['SPE_sd'] = np.std(s_, ddof=1) if len(s_) > 10 else np.nan
            r['WS_oos_sd'] = np.std(e_, ddof=1) if len(e_) > 10 else np.nan
            rows.append(r)
    D = pd.DataFrame(rows)
    for c in ('FPE_mid', 'FPE_term', 'FPE_to', 'SPE_sd', 'WS_oos_sd'):
        D[c] *= 1000
    P = D.groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
    w('')
    w(f"{sel_name}: participant means (n = {P['sub'].nunique()}), mm")
    JS = {}
    for ax in ('ML', 'AP'):
        v = P[P.ax == ax]
        JS[ax] = {k: float(v[k].mean()) for k in ('FPE_mid', 'FPE_term', 'FPE_to', 'SPE_sd', 'WS_oos_sd', 'R2_mid',
                                                   'R2_term', 'R2_to')}
        for c in ('FPE_mid', 'FPE_term', 'FPE_to', 'WS_oos_sd'):
            ok = v[['SPE_sd', c]].dropna()
            rs, ps = stats.spearmanr(ok.SPE_sd, ok[c])
            JS[ax]['rho_' + c], JS[ax]['p_' + c] = float(rs), float(ps)
        JS[ax]['n'] = int(v['sub'].nunique())
        w(f"  {ax}: FPE mid-swing {v.FPE_mid.mean():5.1f} (R2 {100 * v.R2_mid.mean():4.0f}%), terminal swing "
          f"{v.FPE_term.mean():5.1f} (R2 {100 * v.R2_term.mean():4.0f}%), toe-off {v.FPE_to.mean():5.1f} "
          f"(R2 {100 * v.R2_to.mean():4.0f}%) | SPE SD {v.SPE_sd.mean():5.1f}, W&S held-out SD {v.WS_oos_sd.mean():5.1f}")
        for c in ('FPE_mid', 'FPE_term', 'FPE_to', 'WS_oos_sd'):
            ok = v[['SPE_sd', c]].dropna()
            rs, ps = stats.spearmanr(ok.SPE_sd, ok[c]); rp, pp = stats.pearsonr(ok.SPE_sd, ok[c])
            w(f"       SPE SD against {c:9s}: Spearman {rs:+.2f} (p = {ps:.4f}), Pearson {rp:+.2f} (p = {pp:.4f})")
    return JS


OUT = {'all': run('All steps', lambda g: pd.Series(True, index=g.index)),
       'nocross': run('Steps without an obstacle crossing', lambda g: ~g['cross'].astype(bool))}
save_json(OUT, 'lang_fpe.json')
w.save('lang_fpe.txt')
