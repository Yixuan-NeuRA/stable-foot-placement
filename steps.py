# -*- coding: utf-8 -*-
"""Detect steps and extract the per-step variables.

For every block: anatomical axes, gait events (Zeni et al. coordinate method), the sacrum state at toe-off,
mid-swing and heel strike, the landing position, and the step gates. A block is left out if the two feet do not
lie on either side of the sacrum tracker (two blocks in this dataset). The pendulum length is each participant's
leg length, the median sacrum-to-stance-foot 3-D distance over single support, so w0 = sqrt(g / leg length) is one
number per participant.

Options
  --cut N            low-pass cut-off in Hz (default 6)
  --causal           one-pass (causal) filtering and differencing of the sacrum
  --legdef vertical  per-step vertical sacrum height above the stance foot as the pendulum length
Writes  results/steps{suffix}.csv, step_blocks{suffix}.csv, legs{suffix}.csv, step_gates{suffix}.json
"""
import sys
import numpy as np, pandas as pd
from scipy.signal import butter, filtfilt, lfilter, find_peaks
from common import utf8_stdout, load_tracking, res, save_json

utf8_stdout()
G, FS, DT = 9.81, 100.0, 0.01
CUT = float(sys.argv[sys.argv.index('--cut') + 1]) if '--cut' in sys.argv else 6.0
CAUSAL = '--causal' in sys.argv
LEGDEF = 'vertical' if '--legdef' in sys.argv and sys.argv[sys.argv.index('--legdef') + 1] == 'vertical' \
    else 'leg3d'
SUF = (('_causal' if CAUSAL else '') + ('' if CUT == 6.0 else f'_cut{CUT:g}')
       + ('_hvert' if LEGDEF == 'vertical' else ''))
GATES = {}


def count(sub, gate):
    GATES[(sub, gate)] = GATES.get((sub, gate), 0) + 1


def block_axes(W, L, R, ok):
    """Forward (AP) and rightward (ML) unit vectors in the horizontal plane, and the belt speed.
    None if the two feet do not lie on either side of the sacrum."""
    rel = np.vstack([L[ok][:, [0, 2]] - W[ok][:, [0, 2]], R[ok][:, [0, 2]] - W[ok][:, [0, 2]]])
    rel = rel - rel.mean(0)
    ev, V = np.linalg.eigh(np.cov(rel.T))
    ap = V[:, int(np.argmax(ev))]
    ml = np.array([-ap[1], ap[0]])
    prj = lambda M, u: M[:, 0] * u[0] + M[:, 2] * u[1]
    planted_low = L[:, 1] < R[:, 1]
    vL = np.gradient(prj(L, ap), DT)
    vR = np.gradient(prj(R, ap), DT)
    belt = np.where(planted_low, vL, vR)
    if np.nanmedian(belt[ok]) > 0:              # the stance foot moves backwards with the belt
        ap, ml = -ap, -ml
    latR = np.nanmedian(prj(R, ml)[ok] - prj(W, ml)[ok])
    latL = np.nanmedian(prj(L, ml)[ok] - prj(W, ml)[ok])
    if latR < latL:
        ml = -ml
        latR, latL = -latR, -latL
    if not latR > 0 > latL:
        return None
    belt_speed = float(np.nanmedian(np.abs(np.where(planted_low, vL, vR)[ok])))
    return ap, ml, belt_speed, 1000 * latR, 1000 * latL


def process(g):
    sub = g['participant'].iat[0]; sess = g['session'].iat[0]; blk = g['block'].iat[0]
    W = g[['waist_x', 'waist_y', 'waist_z']].values
    L = g[['left_foot_x', 'left_foot_y', 'left_foot_z']].values
    R = g[['right_foot_x', 'right_foot_y', 'right_foot_z']].values
    t = g['time_s'].values
    bad = ((g['qc_bad'].values == 1) | (g['jump'].values == 1)
           | np.isnan(W).any(1) | np.isnan(L).any(1) | np.isnan(R).any(1))
    ok = ~bad
    axes = block_axes(W, L, R, ok)
    if axes is None:
        return None, None
    assert ok.sum() > 0.5 * len(ok) or sub in ('P03', 'P05'), sub
    ap, ml, belt, latR, latL = axes
    prj = lambda M, u: M[:, 0] * u[0] + M[:, 2] * u[1]

    def fill(x):
        x = x.copy()
        m = np.isnan(x)
        if m.any():
            x[m] = np.interp(np.flatnonzero(m), np.flatnonzero(~m), x[~m])
        return x

    b6, a6 = butter(4, CUT / (FS / 2), 'low')
    lp = lambda x: filtfilt(b6, a6, fill(x))
    lpc = (lambda x: lfilter(b6, a6, fill(x))) if CAUSAL else lp
    ddt = (lambda x: np.r_[0.0, np.diff(x)] / DT) if CAUSAL else (lambda x: np.gradient(x, DT))

    sAP, sML = lpc(prj(W, ap)), lpc(prj(W, ml))
    vAP, vML = ddt(sAP), ddt(sML)
    F = {'left': dict(AP=lp(prj(L, ap)), ML=lp(prj(L, ml)), Y=fill(L[:, 1]),
                      P3=np.c_[fill(L[:, 0]), fill(L[:, 1]), fill(L[:, 2])]),
         'right': dict(AP=lp(prj(R, ap)), ML=lp(prj(R, ml)), Y=fill(R[:, 1]),
                       P3=np.c_[fill(R[:, 0]), fill(R[:, 1]), fill(R[:, 2])])}
    W3 = np.c_[fill(W[:, 0]), fill(W[:, 1]), fill(W[:, 2])]
    floor = {k: np.percentile(F[k]['Y'][ok], 5) for k in F}
    AX = {'AP': (sAP, vAP), 'ML': (sML, vML)}

    rows = []
    for k in ('left', 'right'):
        other = 'right' if k == 'left' else 'left'
        fAP, fML, fY = F[k]['AP'], F[k]['ML'], F[k]['Y']
        rel = lp(fAP - sAP)                     # foot relative to the sacrum: heel strike = max, toe-off = min
        d = int(0.30 * FS)
        hs, _ = find_peaks(rel, distance=d, prominence=0.05)
        to, _ = find_peaks(-rel, distance=d, prominence=0.05)
        for t0 in to:
            nx = hs[hs > t0]
            if not len(nx):
                continue
            t1 = int(nx[0])
            count(sub, 'candidates')
            Tc = (t1 - t0) * DT
            if not (0.20 <= Tc <= 1.00):
                count(sub, 'swing_time'); continue
            if t0 < 3 or t1 >= len(t) - int(0.6 * FS):
                count(sub, 'window_edge'); continue
            if bad[t0:t1 + 1].any():
                count(sub, 'rejected_frame'); continue
            # the pendulum during this swing is the other (stance) leg
            h_vert = float(np.mean(W[t0:t1 + 1, 1] - F[other]['Y'][t0:t1 + 1]))
            leg_step = float(np.mean(np.linalg.norm(W3[t0:t1 + 1] - F[other]['P3'][t0:t1 + 1], axis=1)))
            gate = h_vert if LEGDEF == 'vertical' else leg_step
            if not (0.50 <= gate <= 1.30):
                count(sub, 'pendulum_length'); continue
            if (fAP[t1] - fAP[t0]) < 0.05:
                count(sub, 'foot_advance'); continue
            v_step = float(-np.median(np.gradient(F[other]['AP'][t0:t1 + 1], DT)))
            if not (0.15 <= v_step <= 2.5):
                count(sub, 'gait_speed'); continue
            rec = dict(sub=sub, sess=sess, blk=blk, foot=k, stance=other, Tc=Tc,
                       h_vert=h_vert, leg_step=leg_step,
                       clear=float(np.max(fY[t0:t1 + 1]) - floor[k]), speed=v_step,
                       belt=belt, t_to=t[t0], t_hs=t[t1])
            tm = (t0 + t1) // 2
            for axn, (sP, vP) in AX.items():
                fk = fAP if axn == 'AP' else fML
                fo = F[other]['AP'] if axn == 'AP' else F[other]['ML']
                rec[axn + '_p'] = fk[t1]                # landing
                rec[axn + '_other'] = fo[t1]            # stance foot at heel strike
                rec[axn + '_stanceTO'] = fo[t0]         # stance foot at toe-off
                rec[axn + '_stanceMS'] = fo[tm]         # stance foot at mid-swing
                rec[axn + '_pos'] = sP[t0]
                rec[axn + '_v'] = vP[t0]
                rec[axn + '_posHS'] = sP[t1]
                rec[axn + '_vHS'] = vP[t1]
                rec[axn + '_rMS'] = sP[tm]
                rec[axn + '_vMS'] = vP[tm]
            rows.append(rec)
    meta = dict(sub=sub, sess=sess, blk=blk, badfrac=float(bad.mean()), belt=belt,
                latR=latR, latL=latL, nsteps=len(rows),
                speed=float(np.median([r['speed'] for r in rows])) if rows else np.nan)
    return rows, meta


D = load_tracking()
allrows, metas, LEFT_OUT = [], [], []
for (sub, sess, blk), g in D.groupby(['participant', 'session', 'block'], sort=True):
    r, m = process(g.sort_values('frame').reset_index(drop=True))
    if m is None:
        LEFT_OUT.append(f'{sub}-{blk}')
        print(f'{sub:4s} {blk:3s} left out: the feet do not lie on either side of the sacrum')
        continue
    allrows += r; metas.append(m)
    print(f"{m['sub']:4s} {m['blk']:3s} rejected frames {100*m['badfrac']:4.1f}%  belt {m['belt']:.2f} m/s  "
          f"speed {m['speed']:.2f}  steps {m['nsteps']:4d}  "
          f"feet right {m['latR']:+4.0f} mm / left {m['latL']:+4.0f} mm of the sacrum")

S = pd.DataFrame(allrows).sort_values(['sub', 'blk', 'foot', 't_hs']).reset_index(drop=True)
M = pd.DataFrame(metas)

# the pendulum: one leg length per participant, or the per-step vertical height
LEGS = []
for sub, g in S.groupby('sub'):
    LEGS.append(dict(sub=sub, leg=float(np.median(g.leg_step)),
                     leg_left=float(np.median(g[g.stance == 'left'].leg_step)),
                     leg_right=float(np.median(g[g.stance == 'right'].leg_step)),
                     leg_block_range=float(np.ptp(g.groupby('blk').leg_step.median())),
                     h_vert=float(np.mean(g.h_vert)), n=len(g)))
LEGS = pd.DataFrame(LEGS)
S['h'] = S['h_vert'] if LEGDEF == 'vertical' else S['sub'].map(LEGS.set_index('sub').leg)
S['w0'] = np.sqrt(G / S['h'])
for ax in ('ML', 'AP'):
    S[ax + '_xiTO'] = S[ax + '_pos'] + S[ax + '_v'] / S['w0']      # XCoM at toe-off: the SFP
    S[ax + '_xiHS'] = S[ax + '_posHS'] + S[ax + '_vHS'] / S['w0']  # XCoM at heel strike: margin of stability

# obstacle crossing: peak clearance more than four scaled MADs above the cell's median
for _, gi in S.groupby(['sub', 'blk', 'foot']):
    c = gi['clear'].values
    mad = np.median(np.abs(c - np.median(c))) or 1e-6
    S.loc[gi.index, 'clear_z'] = (c - np.median(c)) / (1.4826 * mad)
S['cross'] = S['clear_z'] > 4.0

S.to_csv(res(f'steps{SUF}.csv'), index=False)
M.to_csv(res(f'step_blocks{SUF}.csv'), index=False)
LEGS.to_csv(res(f'legs{SUF}.csv'), index=False)
GG = pd.Series(GATES).unstack().fillna(0).astype(int)
tot = GG.sum()
save_json(dict(blocks_left_out=LEFT_OUT, candidates=int(tot['candidates']), kept=len(S),
               **{k: int(tot[k]) for k in GG.columns if k != 'candidates'},
               pendulum_length_by_participant={s: int(v) for s, v in GG.get('pendulum_length', {}).items() if v}),
          f'step_gates{SUF}.json')

print()
print(f"pendulum length: {'vertical sacrum height per step' if LEGDEF == 'vertical' else 'leg length per participant'}")
print(f"steps {len(S)} | participants {S['sub'].nunique()} | blocks {len(M)} | cells "
      f"{S.groupby(['sub', 'blk', 'foot']).ngroups}")
print(f"swing time {S.Tc.mean():.3f} s | pendulum length {S.h.mean():.3f} m | 1/w0 {np.mean(1 / S.w0):.3f} s")
print("step gates: " + ', '.join(f'{k} {int(v)}' for k, v in tot.items()))
print(f"crossing steps {S.cross.sum()} ({100 * S.cross.mean():.1f}%)")
