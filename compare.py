# -*- coding: utf-8 -*-
"""The SFP against the Wang & Srinivasan regression, under one grid of estimators.

Two families of forecasts of the next landing, both causal (step i is forecast from steps before i only):
  A  weights fitted to the walker:   p - s = [rule] + b1 (r - s) + b2 v
     s = stance foot, r = sacrum position, v = sacrum velocity, all at toe-off. With the moving average as the
     rule this is the Wang & Srinivasan regression; least squares adds the previous landing as a regressor,
     the linear trend adds the step number.
  B  weights fixed by the pendulum:  p = xi + [rule on the margin m = p - xi],  xi = r + v / w0.
     The moving average refitted on all steps is the SFP.
Each rule is scored frozen after the 50-step warm-up, refitted on all steps so far, and refitted on the last
10, 20, 30, 40 and 50 steps (3 methods x 7 schemes). Also: calibration length, learning curve, transfer between
blocks, the fitted weights, the ordered sequence (unfreeze, add one, two, three lags; Wilcoxon, stop at the first
non-rejection), the velocity term, the SPE, the warm-up sweep and the per-participant moving-block bootstrap.

Usage   python compare.py [--steps SUFFIX] [--drop P06-V1,...] [--keep-subs P03,...] [--tag NAME] [--quick]
Writes  results/compare{tag}.txt, .csv, .json and results/spe{tag}.csv
"""
import sys
import numpy as np, pandas as pd
from scipy import stats
from common import utf8_stdout, arg, res, save_json, Report, CELL, WARM as W, is_obstacle_block

utf8_stdout()
WINS = (10, 20, 30, 40, 50)
SCHEMES = ['frozen', 'expand'] + [f'window{N}' for N in WINS]
METHODS = ['MA', 'LS', 'LR']
PSUF = arg('--steps')
DROP = [x for x in arg('--drop').split(',') if x]
KEEP = [x for x in arg('--keep-subs').split(',') if x]
TAG = arg('--tag', PSUF)
QUICK = '--quick' in sys.argv

S = pd.read_csv(res(f'steps{PSUF}.csv')).sort_values(CELL + ['t_hs'])
for d in DROP:
    S = S[~((S['sub'] == d) | ((S['sub'] + '-' + S['blk']) == d))]
if KEEP:
    S = S[S['sub'].isin(KEEP)]
S = S.reset_index(drop=True)
for ax in ('ML', 'AP'):
    S[ax + '_MoS'] = S[ax + '_p'] - S[ax + '_xiHS']       # margin of stability at heel strike
    S[ax + '_m'] = S[ax + '_p'] - S[ax + '_xiTO']         # margin at toe-off, the quantity B forecasts
    S[ax + '_y'] = S[ax + '_p'] - S[ax + '_stanceTO']     # landing about the stance foot, the quantity A forecasts
    S[ax + '_x1'] = S[ax + '_pos'] - S[ax + '_stanceTO']  # sacrum about the stance foot at toe-off

w = Report()
J = {}


def _blank(n):
    return np.full(n, np.nan)


def _ols(X, z):
    return np.linalg.lstsq(X, z, rcond=None)[0]


# ------------------------------------------------------------------ B: rules on one series
# Each rule takes a series y and returns f, with f[i] forecast from y[:i]; NaN before the warm-up.
def ma_frozen(y, warm=W):
    """Mean of the warm-up steps, held."""
    f = _blank(len(y)); f[warm:] = y[:warm].mean(); return f


def ma_expand(y, warm=W):
    """Mean of every step so far: the SFP offset, Eq. (3)."""
    f = _blank(len(y)); c = np.cumsum(y)
    for i in range(warm, len(y)):
        f[i] = c[i - 1] / i
    return f


def ma_window(y, N, warm=W):
    f = _blank(len(y))
    for i in range(warm, len(y)):
        f[i] = y[max(0, i - N):i].mean()
    return f


def ls_frozen(y, warm=W):
    """One lag, fitted once on the warm-up."""
    f = _blank(len(y))
    g = _ols(np.c_[np.ones(warm - 1), y[:warm - 1]], y[1:warm])
    for i in range(warm, len(y)):
        f[i] = g[0] + g[1] * y[i - 1]
    return f


def ls_expand(y, warm=W):
    """One lag, least squares over every step so far (normal equations)."""
    n = len(y); A = np.zeros((2, 2)); c = np.zeros(2); f = _blank(n)
    for i in range(1, n):
        x = np.r_[1.0, y[i - 1]]
        if i >= warm:
            f[i] = _ols(A, c) @ x
        A += np.outer(x, x); c += x * y[i]
    return f


def ls_window(y, N, warm=W):
    """One lag, refitted at every step on the last N pairs."""
    f = _blank(len(y))
    for i in range(warm, len(y)):
        j0 = max(1, i - N)
        g = _ols(np.c_[np.ones(i - j0), y[j0 - 1:i - 1]], y[j0:i])
        f[i] = g[0] + g[1] * y[i - 1]
    return f


def ar_expand(y, K, warm=W):
    """K lags, refitted at every step. K = 0 is ma_expand."""
    n = len(y); d = K + 1
    A = np.zeros((d, d)); c = np.zeros(d); f = _blank(n)
    for i in range(K, n):
        idx = [i - k for k in range(1, K + 1)]
        assert all(j < i for j in idx)
        x = np.r_[1.0, [y[j] for j in idx]] if K else np.r_[1.0]
        if i >= warm:
            f[i] = _ols(A, c) @ x
        A += np.outer(x, x); c += x * y[i]
    return f


def lr_frozen(y, warm=W):
    f = _blank(len(y))
    g = _ols(np.c_[np.ones(warm), np.arange(warm, dtype=float)], y[:warm])
    f[warm:] = g[0] + g[1] * np.arange(warm, len(y), dtype=float)
    return f


def lr_expand(y, warm=W):
    n = len(y); f = _blank(n)
    A = np.zeros((2, 2)); c = np.zeros(2)
    for i in range(n):
        if i >= warm:
            f[i] = _ols(A, c) @ np.r_[1.0, float(i)]
        x = np.r_[1.0, float(i)]
        A += np.outer(x, x); c += x * y[i]
    return f


def lr_window(y, N, warm=W):
    f = _blank(len(y))
    for i in range(warm, len(y)):
        j0 = max(0, i - N)
        g = _ols(np.c_[np.ones(i - j0), np.arange(j0, i, dtype=float)], y[j0:i])
        f[i] = g[0] + g[1] * i
    return f


def best_memory(y, warm=W):
    """The best rectangular window for this cell, chosen on the scored steps (in sample, a bound)."""
    best, out_ = np.inf, None
    for N in (5, 10, 20, 40, 80, 160, 10 ** 9):
        f = ma_window(y, N, warm)
        e = np.nanmean((y[warm:] - f[warm:]) ** 2)
        if e < best:
            best, out_ = e, f
    return out_


def best_constant(y, warm=W):
    """Mean of the scored steps (in sample, the bound on any fixed offset)."""
    f = _blank(len(y)); f[warm:] = y[warm:].mean(); return f


RULE_B = {'MA_frozen': ma_frozen, 'MA_expand': ma_expand,
          'LS_frozen': ls_frozen, 'LS_expand': ls_expand,
          'LR_frozen': lr_frozen, 'LR_expand': lr_expand}
for N in WINS:
    RULE_B[f'MA_window{N}'] = (lambda N: lambda y, warm=W: ma_window(y, N, warm))(N)
    RULE_B[f'LS_window{N}'] = (lambda N: lambda y, warm=W: ls_window(y, N, warm))(N)
    RULE_B[f'LR_window{N}'] = (lambda N: lambda y, warm=W: lr_window(y, N, warm))(N)
GRID = [f'{m}_{s}' for m in METHODS for s in SCHEMES]
assert set(GRID) == set(RULE_B)
EXTRA_B = {'best_constant': best_constant, 'best_memory': best_memory,
           'K2_expand': lambda y, warm=W: ar_expand(y, 2, warm),
           'K3_expand': lambda y, warm=W: ar_expand(y, 3, warm)}
BOUNDS = ('best_constant', 'best_memory')
RULES = {**RULE_B, **EXTRA_B}


# ------------------------------------------------------------------ A: the regression
def design_A(method, y, x1, x2):
    """Regressors and usable rows. MA: [1, x1, x2] (Wang & Srinivasan); LS: [1, previous y, x1, x2];
    LR: [1, step number / 100, x1, x2]."""
    n = len(y)
    if method == 'MA':
        X = np.c_[np.ones(n), x1, x2]; ok = np.ones(n, bool)
    elif method == 'LS':
        X = np.c_[np.ones(n), np.r_[np.nan, y[:-1]], x1, x2]; ok = np.r_[False, np.ones(n - 1, bool)]
    else:
        X = np.c_[np.ones(n), np.arange(n) / 100.0, x1, x2]; ok = np.ones(n, bool)
    return X, ok


def reg_forecast(X, ok, y, scheme, warm=W, lo_fixed=None, hi_fixed=None):
    """f[i] = X[i] @ beta, beta by least squares on the usable rows j < i that the scheme selects:
    'frozen' (rows < warm), 'expand' (all rows < i), 'windowN' (rows i-N .. i-1), 'fixed' (rows lo..hi-1)."""
    n, d = X.shape
    Xz = np.where(ok[:, None], X, 0.0); yz = np.where(ok, y, 0.0)
    cO = np.r_[np.zeros((1, d, d)), np.cumsum(Xz[:, :, None] * Xz[:, None, :], 0)]
    cB = np.r_[np.zeros((1, d)), np.cumsum(Xz * yz[:, None], 0)]
    f = _blank(n)
    idx = np.arange(warm, n)
    if not len(idx):
        return f
    if scheme in ('frozen', 'fixed'):
        lo, hi = (0, warm) if scheme == 'frozen' else (lo_fixed, hi_fixed)
        beta = np.linalg.pinv(cO[hi] - cO[lo]) @ (cB[hi] - cB[lo])
        f[idx] = X[idx] @ beta
        return f
    if scheme == 'expand':
        lo = np.zeros_like(idx)
    else:
        N = int(scheme[len('window'):])
        lo = np.maximum(0, idx - N)
    A = cO[idx] - cO[lo]; b = cB[idx] - cB[lo]
    beta = np.einsum('nij,nj->ni', np.linalg.pinv(A), b)
    f[idx] = np.einsum('ni,ni->n', X[idx], beta)
    return f


def layer_A(rule, y, x1, x2, warm=W):
    if rule == 'best_constant':
        X, ok = design_A('MA', y, x1, x2)
        return reg_forecast(X, ok, y, 'fixed', warm, warm, len(y))
    if rule == 'best_memory':
        X, ok = design_A('MA', y, x1, x2)
        best, out_ = np.inf, None
        for N in (5, 10, 20, 40, 80, 160, 10 ** 9):
            f = reg_forecast(X, ok, y, f'window{N}', warm)
            e = np.nanmean((y[warm:] - f[warm:]) ** 2)
            if e < best:
                best, out_ = e, f
        return out_
    method, scheme = rule.split('_')
    X, ok = design_A(method, y, x1, x2)
    return reg_forecast(X, ok, y, scheme, warm)


# ------------------------------------------------------------------ checks: causality and exact refits
def _selftest():
    g = S.groupby(CELL).filter(lambda x: len(x) > 150)
    g = g[g.groupby(CELL).ngroup() == 0]
    m = g.ML_m.values.copy(); y = g.ML_y.values.copy(); x1 = g.ML_x1.values.copy(); x2 = g.ML_v.values.copy()
    i0 = 90
    for nm in GRID + [k for k in EXTRA_B if k not in BOUNDS]:
        a = RULES[nm](m); m2 = m.copy(); m2[i0:] += 0.5
        assert np.nanmax(np.abs(a[:i0 + 1] - RULES[nm](m2)[:i0 + 1])) < 1e-12, f"B {nm} is not causal"
        assert np.all(np.isnan(a[:W])), f"B {nm} forecasts inside the warm-up"
    for nm in GRID:
        a = layer_A(nm, y, x1, x2)
        y2, a1, a2 = y.copy(), x1.copy(), x2.copy(); y2[i0:] += 0.5; a1[i0 + 1:] += 0.3; a2[i0 + 1:] -= 0.2
        b = layer_A(nm, y2, a1, a2)
        assert np.nanmax(np.abs(a[:i0 + 1] - b[:i0 + 1])) < 1e-9, f"A {nm} is not causal"
        assert np.all(np.isnan(a[:W])), f"A {nm} forecasts inside the warm-up"
    fa = layer_A('MA_expand', y, x1, x2)
    for i in (W, W + 37, len(y) - 1):
        bb = _ols(np.c_[np.ones(i), x1[:i], x2[:i]], y[:i])
        assert abs(fa[i] - (bb[0] + bb[1] * x1[i] + bb[2] * x2[i])) < 1e-9
        bl = _ols(np.c_[np.ones(i - 1), y[:i - 1], x1[1:i], x2[1:i]], y[1:i])
        fl = layer_A('LS_expand', y, x1, x2)[i]
        assert abs(fl - (bl[0] + bl[1] * y[i - 1] + bl[2] * x1[i] + bl[3] * x2[i])) < 1e-9
        b10 = _ols(np.c_[np.ones(10), x1[i - 10:i], x2[i - 10:i]], y[i - 10:i])
        assert abs(layer_A('MA_window10', y, x1, x2)[i] - (b10 @ np.r_[1, x1[i], x2[i]])) < 1e-9
    # B with the weights fixed at (1, 1/w0) is A's regression with those weights
    w0 = g.w0.values
    assert np.allclose(y - x1 - x2 / w0, m, atol=1e-12)
    fe = ls_expand(m)
    for i in (W, W + 37, len(m) - 1):
        gg = _ols(np.c_[np.ones(i - 1), m[:i - 1]], m[1:i])
        assert abs(fe[i] - (gg[0] + gg[1] * m[i - 1])) < 1e-9
        assert abs(ma_expand(m)[i] - m[:i].mean()) < 1e-12
    assert np.allclose(ar_expand(m, 0)[W:], ma_expand(m)[W:], atol=1e-12)
    assert np.allclose(ar_expand(m, 1)[W:], ls_expand(m)[W:], atol=1e-9)
    for ax in ('ML', 'AP'):
        assert np.nanmax(np.abs(S[ax + '_p'] - (S[ax + '_xiTO'] + S[ax + '_m']))) < 1e-12
        assert np.nanmax(np.abs(S[ax + '_xiTO'] - (S[ax + '_pos'] + S[ax + '_v'] / S.w0))) < 1e-12
    return True


assert _selftest()

# ------------------------------------------------------------------ score every rule in every cell
rows, RESID = [], {}
for (sub, blk, foot), g in S.groupby(CELL):
    g = g.reset_index(drop=True)
    if len(g) <= W + 20:
        continue
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = g[ax + '_m'].values
        y = g[ax + '_y'].values; x1 = g[ax + '_x1'].values; x2 = g[ax + '_v'].values
        keep = np.arange(len(g)) >= W
        rec = dict(sub=sub, blk=blk, foot=foot, ax=ax, n=int(keep.sum()))
        rmse = lambda e: 1000.0 * np.sqrt(np.mean(e[keep] ** 2))
        for nm in list(RULES):
            eb = p - (xi + RULES[nm](m))
            rec['B_' + nm] = rmse(eb)
            if nm in ('MA_frozen', 'MA_expand', 'LS_expand', 'K2_expand', 'K3_expand'):
                RESID[(sub, blk, foot, ax, 'B_' + nm)] = 1000 * eb[keep]
            if nm in GRID or nm in BOUNDS:
                ea = y - layer_A(nm, y, x1, x2)
                rec['A_' + nm] = rmse(ea)
                if nm == 'MA_expand':
                    RESID[(sub, blk, foot, ax, 'A_' + nm)] = 1000 * ea[keep]
        rows.append(rec)

R = pd.DataFrame(rows)
COLS = [c for c in R.columns if c.startswith(('A_', 'B_'))]
P = R.groupby(['sub', 'ax'])[COLS].mean().reset_index()
NSUB = P['sub'].nunique()
J['n_participants'] = int(NSUB)
J['n_cells'] = int(R.groupby(CELL).ngroups)
J['n_blocks'] = int(R.groupby(['sub', 'blk']).ngroups)
J['n_steps'] = int(len(S))
J['n_scored'] = int(R.n.sum() / 2)


def pair(ax, worse, better, drop=(), frame=None):
    """`worse` minus `better` in mm (positive: `better` is better), paired t, Wilcoxon, count better."""
    F = P if frame is None else frame
    d = F[(F.ax == ax) & (~F['sub'].isin(drop))]
    x, y = d[worse].values, d[better].values
    t, pt = stats.ttest_rel(x, y)
    try:
        pw = stats.wilcoxon(x, y).pvalue
    except ValueError:
        pw = np.nan
    return x.mean() - y.mean(), t, pt, pw, int((y < x).sum()), len(d)


def line(tag, worse, better, key=None, frame=None):
    for ax in ('ML', 'AP'):
        d, t, pt, pw, c, n = pair(ax, worse, better, frame=frame)
        w(f"  {ax}  {tag:44s}{d:+7.2f} mm   t({n-1}) = {t:5.2f}  p = {pt:.4f} | "
          f"Wilcoxon p = {pw:.4f} | better in {c}/{n}")
        if key:
            J[f'{key}_{ax}'] = dict(diff=d, t=t, p_t=pt, p_w=pw, better=c, n=n)


# ------------------------------------------------------------------ the ordered sequence (B)
SEQ_RULES = ['B_MA_frozen', 'B_MA_expand', 'B_LS_expand', 'B_K2_expand', 'B_K3_expand']
SEQ_TAGS = ['update the mean (frozen -> refitted)',
            'add one lag (K = 0 -> 1)',
            'add a second lag (K = 1 -> 2)',
            'add a third lag (K = 2 -> 3)']
SELECTED, SEQLOG = {}, {}
for ax in ('ML', 'AP'):
    sel = None
    SEQLOG[ax] = []
    for k, tag in enumerate(SEQ_TAGS):
        a, b = SEQ_RULES[k], SEQ_RULES[k + 1]
        d, t, pt, pw, c, n = pair(ax, a, b)
        rejected = (pw < 0.05) and d > 0
        SEQLOG[ax].append((tag, d, pt, pw, c, n, rejected if sel is None else None))
        if sel is None and not rejected:
            sel = a
    SELECTED[ax] = (sel or SEQ_RULES[-1])[2:]
SELFN = {ax: RULES[SELECTED[ax]] for ax in ('ML', 'AP')}
J['selected'] = SELECTED
for ax in ('ML', 'AP'):
    P.loc[P.ax == ax, 'SFP'] = P.loc[P.ax == ax, 'B_' + SELECTED[ax]]
    R.loc[R.ax == ax, 'SFP'] = R.loc[R.ax == ax, 'B_' + SELECTED[ax]]
P['SFP_K0'] = P['B_MA_expand']; P['SFP_K1'] = P['B_LS_expand']
R.to_csv(res(f'compare{TAG}.csv'), index=False)

MNAME = {'MA': 'moving average', 'LS': 'least squares ', 'LR': 'linear trend  '}
SNAME = {'frozen': 'frozen after 50 steps', 'expand': 'refitted, all steps'}
SNAME.update({f'window{N}': f'refitted, last {N} steps' for N in WINS})
NICE = {f'{m}_{s}': f'{MNAME[m]:15s}{SNAME[s]}' for m in METHODS for s in SCHEMES}
NICE.update({'best_constant': 'best constant (in sample)',
             'best_memory': 'best memory (in sample)',
             'K2_expand': 'two lags, refitted, all steps', 'K3_expand': 'three lags, refitted, all steps'})
ORDER = GRID + ['best_constant', 'best_memory', 'K2_expand', 'K3_expand']

w("SFP and Wang & Srinivasan regression, one grid of estimators")
w(f"steps{PSUF}.csv | dropped: {', '.join(DROP) or 'none'} | kept: {', '.join(KEEP) or 'all'}")
w(f"participants {NSUB} | blocks {J['n_blocks']} | cells {J['n_cells']} | steps {J['n_steps']} | "
  f"warm-up {W} | scored steps per cell {R.n.mean():.0f} | total scored {J['n_scored']}")
w("checks passed: every rule causal and empty in its warm-up; refits exact")
w("")
w(f"Held-out RMSE of the landing about the target (mm), mean of the {NSUB} participants")
w(f"  {'':46s}{'A: fitted':>18s}{'B: pendulum':>18s}")
w(f"  {'':46s}{'ML':>9s}{'AP':>9s}{'ML':>9s}{'AP':>9s}")
for k in ORDER:
    vals = []
    for L_ in ('A', 'B'):
        if f'{L_}_{k}' in P:
            vm, va = P[P.ax == 'ML'][f'{L_}_{k}'].mean(), P[P.ax == 'AP'][f'{L_}_{k}'].mean()
            J[f'{L_}_{k}'] = dict(ML=vm, AP=va); vals += [vm, va]
        else:
            vals += [np.nan, np.nan]
    w(f"  {NICE[k]:46s}" + "".join(f"{v:9.1f}" for v in vals))
J['SFP'] = dict(ML=P[P.ax == 'ML'].SFP.mean(), AP=P[P.ax == 'AP'].SFP.mean())
w(f"  {'SFP (XCoM + the selected rule)':46s}{'':18s}{J['SFP']['ML']:9.1f}{J['SFP']['AP']:9.1f}")

w("")
w(f"SFP against each of the {len(GRID)} A rules (+: SFP better)")
for k in GRID:
    line(NICE[k], 'A_' + k, 'SFP', key='SFPvsA_' + k)
for ax in ('ML', 'AP'):
    means = {k: J[f'A_{k}'][ax] for k in GRID}
    best = min(means, key=means.get)
    J[f'bestA_{ax}'] = best
    J[f'SFP_beats_{ax}'] = sorted(k for k in GRID if J[f'SFPvsA_{k}_{ax}']['diff'] > 0
                                  and J[f'SFPvsA_{k}_{ax}']['p_w'] < 0.05)
    J[f'SFP_loses_{ax}'] = sorted(k for k in GRID if J[f'SFPvsA_{k}_{ax}']['diff'] < 0
                                  and J[f'SFPvsA_{k}_{ax}']['p_w'] < 0.05)
    w(f"  {ax}: best A rule {best} ({means[best]:.1f} mm); SFP better than {len(J[f'SFP_beats_{ax}'])}, "
      f"worse than {len(J[f'SFP_loses_{ax}'])} {J[f'SFP_loses_{ax}']}")

w("")
w("Fitted against pendulum weights, the same rule and scheme (+: fitted weights better)")
for k in GRID:
    line(NICE[k], 'B_' + k, 'A_' + k, key='slopes_' + k)

# ------------------------------------------------------------------ calibration length
w("")
w("Calibration length: both calibrated once on the first n steps and frozen, scored from step 100 "
  "(+: regression better)")
CAL = (10, 20, 30, 50, 100)
cal_rows = []
for (sub, blk, foot), g in S.groupby(CELL):
    g = g.reset_index(drop=True)
    if len(g) <= 120:
        continue
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = g[ax + '_m'].values
        y = g[ax + '_y'].values; x1 = g[ax + '_x1'].values; x2 = g[ax + '_v'].values
        X, ok = design_A('MA', y, x1, x2)
        rec = dict(sub=sub, ax=ax)
        for n in CAL:
            fs = xi + m[:n].mean()
            fa = reg_forecast(X, ok, y, 'fixed', 0, 0, n)
            rec[f'SFP_{n}'] = 1000 * np.sqrt(np.mean((p - fs)[100:] ** 2))
            rec[f'WS_{n}'] = 1000 * np.sqrt(np.mean((y - fa)[100:] ** 2))
        cal_rows.append(rec)
CP = pd.DataFrame(cal_rows).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for n in CAL:
    line(f'calibrated on {n} steps', f'SFP_{n}', f'WS_{n}', key=f'cal{n}', frame=CP)
    for ax in ('ML', 'AP'):
        J[f'cal{n}_{ax}'].update(SFP=CP[CP.ax == ax][f'SFP_{n}'].mean(), WS=CP[CP.ax == ax][f'WS_{n}'].mean())

# ------------------------------------------------------------------ learning curve
w("")
w("Learning curve: both refitted from a cell's first step; RMSE for a step with i same-foot steps before it")
BINS = [(3, 5), (5, 10), (10, 20), (20, 30), (30, 50), (50, 100), (100, 10 ** 6)]
lc = []
for (sub, blk, foot), g in S.groupby(CELL):
    g = g.reset_index(drop=True)
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = g[ax + '_m'].values
        y = g[ax + '_y'].values; x1 = g[ax + '_x1'].values; x2 = g[ax + '_v'].values
        X, ok = design_A('MA', y, x1, x2)
        fa = reg_forecast(X, ok, y, 'expand', warm=3)
        cm = np.cumsum(m)
        for i in range(3, len(g)):
            lc.append((sub, ax, i, p[i] - xi[i] - cm[i - 1] / i, y[i] - fa[i]))
LC = pd.DataFrame(lc, columns=['sub', 'ax', 'i', 'SFP', 'WS'])
J['learning'] = {}
for ax in ('ML', 'AP'):
    J['learning'][ax] = []
    for lo, hi in BINS:
        d = LC[(LC.ax == ax) & (LC.i >= lo) & (LC.i < hi)]
        q = d.groupby('sub')[['SFP', 'WS']].agg(lambda e: 1000 * np.sqrt(np.mean(e ** 2)))
        pw = stats.wilcoxon(q.SFP, q.WS).pvalue
        lab = f"{lo}-{hi - 1}" if hi < 10 ** 6 else f"{lo}+"
        J['learning'][ax].append(dict(bin=lab, lo=lo, hi=hi, SFP=q.SFP.mean(), WS=q.WS.mean(),
                                      sfp_better=int((q.SFP < q.WS).sum()), n=len(q), p_w=pw))
        w(f"  {ax}  {lab:>8s} steps before: SFP {q.SFP.mean():6.1f}  WS {q.WS.mean():6.1f}  "
          f"SFP better in {int((q.SFP < q.WS).sum()):2d}/{len(q)}  Wilcoxon p = {pw:.4f}")

# ------------------------------------------------------------------ transfer between blocks
w("")
w("Transfer: calibrated on one whole block, applied to another block of the same participant and side")
tr = []
for (sub, foot), gs in S.groupby(['sub', 'foot']):
    blocks = {b: g.reset_index(drop=True) for b, g in gs.groupby('blk')}
    for a in blocks:
        for b in blocks:
            if a == b:
                continue
            for ax in ('ML', 'AP'):
                ga, gb = blocks[a], blocks[b]
                Xa, _ = design_A('MA', ga[ax + '_y'].values, ga[ax + '_x1'].values, ga[ax + '_v'].values)
                Xb, _ = design_A('MA', gb[ax + '_y'].values, gb[ax + '_x1'].values, gb[ax + '_v'].values)
                beta = _ols(Xa, ga[ax + '_y'].values)
                tr.append(dict(sub=sub, ax=ax,
                               SFP=1000 * np.sqrt(np.mean((gb[ax + '_m'] - ga[ax + '_m'].mean()) ** 2)),
                               WS=1000 * np.sqrt(np.mean((gb[ax + '_y'].values - Xb @ beta) ** 2))))
if tr:
    TP = pd.DataFrame(tr).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
    line('transferred to another block', 'SFP', 'WS', key='transfer', frame=TP)

# ------------------------------------------------------------------ the fitted weights
w("")
w("Fitted weights (whole-cell Wang & Srinivasan fits) against the pendulum's 1 and 1/w0")
co = []
for (sub, blk, foot), g in S.groupby(CELL):
    for ax in ('ML', 'AP'):
        X, _ = design_A('MA', g[ax + '_y'].values, g[ax + '_x1'].values, g[ax + '_v'].values)
        b = _ols(X, g[ax + '_y'].values)
        co.append(dict(sub=sub, blk=blk, ax=ax, b1=b[1], b2w0=b[2] * g.w0.median()))
CO = pd.DataFrame(co)
for ax in ('ML', 'AP'):
    d = CO[CO.ax == ax]
    spread = d.groupby('sub').b2w0.agg(lambda v: v.max() - v.min())
    spread = spread[d.groupby('sub').blk.nunique() > 1]
    J[f'slopes_fit_{ax}'] = dict(b1_med=d.b1.median(), b1_q1=d.b1.quantile(.25), b1_q3=d.b1.quantile(.75),
                                 b2_med=d.b2w0.median(), b2_q1=d.b2w0.quantile(.25), b2_q3=d.b2w0.quantile(.75),
                                 b2_between_blocks=spread.median(), n_cells=len(d))
    w(f"  {ax}  position b1 median {d.b1.median():.2f} (IQR {d.b1.quantile(.25):.2f}-{d.b1.quantile(.75):.2f}); "
      f"velocity b2*w0 median {d.b2w0.median():.2f} (IQR {d.b2w0.quantile(.25):.2f}-{d.b2w0.quantile(.75):.2f}); "
      f"a participant's blocks differ by a median {spread.median():.2f} x 1/w0")

# ------------------------------------------------------------------ the ordered sequence
w("")
w("Ordered sequence (B): each step tested against the rule it extends, stopping at the first non-rejection "
  "(Wilcoxon, alpha 0.05)")
for ax in ('ML', 'AP'):
    w(f"  {ax}:")
    for k, (tag, d, pt, pw, c, n, rej) in enumerate(SEQLOG[ax]):
        verdict = "(after the stop)" if rej is None else "rejected, continue" if rej else "not rejected, stop"
        w(f"    {tag:40s}{d:+7.2f} mm  t p = {pt:.3f}  Wilcoxon p = {pw:.3f}  {c}/{n}  {verdict}")
        J[f'seq{k + 1}_{ax}'] = dict(diff=d, p_t=pt, p_w=pw, better=c, n=n, rejected=rej)
    w(f"    selected: {NICE[SELECTED[ax]].strip()}")

w("")
w("The first lag, K = 0 -> K = 1, per participant (+: the lag helps)")
for ax in ('ML', 'AP'):
    d = P[P.ax == ax].set_index('sub')
    diff = (d.SFP_K0 - d.SFP_K1).sort_values()
    w(f"  {ax}  " + "  ".join(f"{s} {v:+.2f}" for s, v in diff.items()))
    dd, t, pt, pw, c, n = pair(ax, 'SFP_K0', 'SFP_K1')
    top = diff.index[-1]
    d2, t2, pt2, pw2, c2, n2 = pair(ax, 'SFP_K0', 'SFP_K1', drop=(top,))
    w(f"       mean {diff.mean():+.2f} | median {diff.median():+.2f} | K = 1 better in {c}/{n} | "
      f"t p = {pt:.3f}, Wilcoxon p = {pw:.3f}")
    J[f'K0K1_{ax}'] = dict(mean=diff.mean(), median=diff.median(), better=c, n=n, p_t=pt, p_w=pw,
                           top=top, top_gain=diff.iloc[-1], without_top=d2, without_better=c2)
w("")
w("Lags against K = 0")
for k in ('LS_expand', 'K2_expand', 'K3_expand'):
    line(NICE[k], 'B_MA_expand', 'B_' + k, key='K0vs_' + k)

w("")
w("Refitted rules (B, all steps)")
for ax in ('ML', 'AP'):
    d = P[P.ax == ax]
    v = {'MA': d.B_MA_expand.mean(), 'LS': d.B_LS_expand.mean(), 'LR': d.B_LR_expand.mean()}
    J[f'refit_spread_{ax}'] = max(v.values()) - min(v.values())
    w(f"  {ax}: MA {v['MA']:.1f}  LS {v['LS']:.1f}  trend {v['LR']:.1f}  spread "
      f"{J[f'refit_spread_{ax}']:.1f} mm | best constant {d.B_best_constant.mean():.1f}")
w("")
w("Frozen against refitted, within each method (B)")
for m_ in METHODS:
    line(m_ + ': frozen -> refitted', f'B_{m_}_frozen', f'B_{m_}_expand', key='unfreeze_' + m_)
w("")
w("All steps so far against the last N only, within each method (+: the window better)")
for L_ in ('B', 'A'):
    for m_ in METHODS:
        for N in WINS:
            line(f'{L_} {m_}: all steps -> last {N}', f'{L_}_{m_}_expand', f'{L_}_{m_}_window{N}',
                 key=f'window{L_}_{m_}_{N}')
for ax in ('ML', 'AP'):
    for L_ in ('A', 'B'):
        for m_ in METHODS:
            J[f'bestscheme_{L_}_{m_}_{ax}'] = min(SCHEMES, key=lambda s: J[f'{L_}_{m_}_{s}'][ax])

# ------------------------------------------------------------------ the velocity term
w("")
w("Velocity term under the selected rule: sacrum position alone against the XCoM")
rows2 = []
for (sub, blk, foot), g in S.groupby(CELL):
    g = g.reset_index(drop=True)
    if len(g) <= W + 20:
        continue
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; keep = np.arange(len(g)) >= W
        f = SELFN[ax]
        rec = dict(sub=sub, ax=ax)
        for nm, pred in (('pos', g[ax + '_pos'].values), ('xcom', g[ax + '_xiTO'].values)):
            rec[nm] = 1000.0 * np.sqrt(np.mean((p - (pred + f(p - pred)))[keep] ** 2))
        c_hat = np.polyfit(g[ax + '_v'].values[:W], (p - g[ax + '_pos'].values)[:W], 1)[0]
        pred = g[ax + '_pos'].values + c_hat * g[ax + '_v'].values
        rec['fitted_c'] = 1000.0 * np.sqrt(np.mean((p - (pred + f(p - pred)))[keep] ** 2))
        rec['c_hat'] = c_hat
        rec['inv_w0'] = float(np.mean(1.0 / g.w0.values))
        rows2.append(rec)
P2 = pd.DataFrame(rows2).groupby(['sub', 'ax']).mean(numeric_only=True).reset_index()
for ax in ('ML', 'AP'):
    d = P2[P2.ax == ax]
    t, pt = stats.ttest_rel(d.pos.values, d.xcom.values)
    pw = stats.wilcoxon(d.pos.values, d.xcom.values).pvalue
    t2, pt2 = stats.ttest_rel(d.fitted_c.values, d.xcom.values)
    J[f'vel_{ax}'] = dict(pos=d.pos.mean(), xcom=d.xcom.mean(), diff=d.pos.mean() - d.xcom.mean(),
                          p_t=pt, p_w=pw, better=int((d.xcom.values < d.pos.values).sum()), n=len(d),
                          fitted=d.fitted_c.mean(), fitted_diff=d.fitted_c.mean() - d.xcom.mean(),
                          fitted_p=pt2, c_hat=d.c_hat.mean(), inv_w0=d.inv_w0.mean())
    w(f"  {ax}  sacrum position only {d.pos.mean():5.1f} -> XCoM {d.xcom.mean():5.1f} mm  "
      f"({d.pos.mean()-d.xcom.mean():+.2f} mm, t({len(d)-1}) = {t:.2f}, p = {pt:.4f}, Wilcoxon p = {pw:.4f}, "
      f"better in {(d.xcom.values < d.pos.values).sum()}/{len(d)})")

# ------------------------------------------------------------------ the SPE
w("")
w(f"SPE under the selected rule (ML {SELECTED['ML']}, AP {SELECTED['AP']})")
sp = []
for (sub, blk, foot), g in S.groupby(CELL):
    g = g.reset_index(drop=True)
    if len(g) <= W + 20:
        continue
    for ax in ('ML', 'AP'):
        p = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = g[ax + '_m'].values
        f = SELFN[ax]
        e = (p - (xi + f(m)))[W:] * 1000.0
        cr = g['cross'].values[W:]
        fixed = (p - f(p))[W:] * 1000.0          # a target forecast from past landings alone
        sp.append(dict(sub=sub, blk=blk, foot=foot, ax=ax, sd=np.std(e, ddof=1), bias=np.mean(e),
                       sd_reg=np.std(e[~cr], ddof=1) if (~cr).sum() > 5 else np.nan,
                       sd_cross=np.std(e[cr], ddof=1) if cr.sum() > 5 else np.nan,
                       fx_reg=np.std(fixed[~cr], ddof=1) if (~cr).sum() > 5 else np.nan,
                       fx_cross=np.std(fixed[cr], ddof=1) if cr.sum() > 5 else np.nan))
SP = pd.DataFrame(sp)
SP.to_csv(res(f'spe{TAG}.csv'), index=False)
for ax in ('ML', 'AP'):
    d = SP[SP.ax == ax].groupby('sub')[['sd', 'bias']].mean()
    J[f'spe_{ax}'] = dict(mean=d.sd.mean(), sd=d.sd.std(ddof=1), lo=d.sd.min(), hi=d.sd.max())
    w(f"  {ax}  SPE SD {d.sd.mean():.1f} +- {d.sd.std(ddof=1):.1f} mm "
      f"(range {d.sd.min():.1f}-{d.sd.max():.1f}) | held-out bias {d.bias.abs().mean():.1f} mm")
w("  obstacle blocks, other steps against crossing steps:")
for ax in ('ML', 'AP'):
    d = SP[(SP.ax == ax) & is_obstacle_block(SP)].groupby('sub')[
        ['sd_reg', 'sd_cross', 'fx_reg', 'fx_cross']].mean().dropna()
    t, pt = stats.ttest_rel(d.sd_cross.values, d.sd_reg.values)
    pw = stats.wilcoxon(d.sd_cross.values, d.sd_reg.values).pvalue
    tr_, ptr = stats.ttest_rel((d.sd_cross / d.sd_reg).values, (d.fx_cross / d.fx_reg).values)
    J[f'cross_spe_{ax}'] = dict(reg=d.sd_reg.mean(), cross=d.sd_cross.mean(), p_t=pt, p_w=pw,
                                higher=int((d.sd_cross.values > d.sd_reg.values).sum()), n=len(d),
                                fx_reg=d.fx_reg.mean(), fx_cross=d.fx_cross.mean(), p_ratio=ptr)
    w(f"  {ax}  SPE SD {d.sd_reg.mean():.0f} -> {d.sd_cross.mean():.0f} mm (p = {pt:.4f}, higher in "
      f"{(d.sd_cross.values > d.sd_reg.values).sum()}/{len(d)}); target from past landings only "
      f"{d.fx_reg.mean():.0f} -> {d.fx_cross.mean():.0f} mm (ratios differ p = {ptr:.4f})")

if not QUICK:
    # -------------------------------------------------------------- warm-up sweep
    w("")
    w("Warm-up length (the scoring boundary moves with it)")
    w(f"  {'warm-up':>9s}{'K=0 frozen':>12s}{'K=0 refit':>12s}{'K=1 refit':>12s}{'best const':>12s}")
    for WW in (30, 50, 80, 110, 140):
        acc = {k: [] for k in ('MA_frozen', 'MA_expand', 'LS_expand', 'best_constant')}
        for (sub, blk, foot), g in S.groupby(CELL):
            g = g.reset_index(drop=True)
            if len(g) <= WW + 20:
                continue
            for ax in ('ML', 'AP'):
                p_ = g[ax + '_p'].values; xi = g[ax + '_xiTO'].values; m = g[ax + '_m'].values
                keep = np.arange(len(g)) >= WW
                for k, f in (('MA_frozen', ma_frozen), ('MA_expand', ma_expand),
                             ('LS_expand', ls_expand), ('best_constant', best_constant)):
                    e = 1000.0 * np.sqrt(np.mean((p_ - (xi + f(m, warm=WW)))[keep] ** 2))
                    acc[k].append(dict(sub=sub, ax=ax, v=e))
        T = {k: pd.DataFrame(v).groupby(['sub', 'ax']).v.mean().reset_index() for k, v in acc.items()}
        for ax in ('ML', 'AP'):
            vals = [T[k][T[k].ax == ax].v.mean() for k in ('MA_frozen', 'MA_expand', 'LS_expand', 'best_constant')]
            J[f'warm{WW}_{ax}'] = dict(zip(('MA_frozen', 'MA_expand', 'LS_expand', 'best_constant'), vals))
            w(f"  {ax} {WW:6d}" + "".join(f"{v:12.1f}" for v in vals))

    # -------------------------------------------------------------- per-participant bootstrap
    w("")
    w("Per participant: moving-block bootstrap (20-step blocks within a cell, 2,000 resamples, both targets "
      "scored on the same resampled steps); counted when the 95% CI excludes zero")
    rng = np.random.default_rng(20261009)
    NBOOT, BL = 2000, 20

    def boot(sub, ax, a, b):
        cells = [(RESID[k], RESID[k[:4] + (b,)]) for k in RESID
                 if k[0] == sub and k[3] == ax and k[4] == a]
        base = np.mean([np.sqrt(np.mean(x ** 2)) - np.sqrt(np.mean(y ** 2)) for x, y in cells])
        idxs = []
        for x, _ in cells:
            n = len(x); nb = int(np.ceil(n / BL))
            st = rng.integers(0, n - BL + 1, (NBOOT, nb))
            idxs.append((st[:, :, None] + np.arange(BL)[None, None, :]).reshape(NBOOT, -1)[:, :n])
        bs = np.mean([np.sqrt(np.mean(x[ix] ** 2, axis=1)) - np.sqrt(np.mean(y[ix] ** 2, axis=1))
                      for (x, y), ix in zip(cells, idxs)], axis=0)
        return base, np.percentile(bs, 2.5), np.percentile(bs, 97.5)

    # boot_WSvsSFP: + = the regression better than the SFP (both refitted on all steps)
    for key, a, b in (('boot_WSvsSFP', 'B_MA_expand', 'A_MA_expand'),
                      ('boot_unfreeze', 'B_MA_frozen', 'B_MA_expand'),
                      ('boot_K0K1', 'B_MA_expand', 'B_LS_expand')):
        for ax in ('ML', 'AP'):
            res_ = [(s,) + boot(s, ax, a, b) for s in sorted(P['sub'].unique())]
            better = sum(1 for _, d, lo, hi in res_ if lo > 0)
            worse = sum(1 for _, d, lo, hi in res_ if hi < 0)
            J[f'{key}_{ax}'] = dict(better=better, worse=worse, n=len(res_))
            w(f"  {key:14s} {ax}: {a} -> {b}: better in {better}, worse in {worse} of {len(res_)}   "
              + "  ".join(f"{s} {d:+.1f}{'*' if lo > 0 else ('-' if hi < 0 else '')}" for s, d, lo, hi in res_))

w("")
w("Per participant, held-out RMSE (mm): Wang & Srinivasan (refitted, all steps) / SFP")
for ax in ('ML', 'AP'):
    d = P[P.ax == ax]
    w(f"  {ax}   " + "  ".join(f"{r['sub']} {r['A_MA_expand']:.1f}/{r['SFP']:.1f}" for _, r in d.iterrows()))

w.save(f'compare{TAG}.txt')
save_json(J, f'compare{TAG}.json')
