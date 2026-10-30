# -*- coding: utf-8 -*-
"""The main results in every analysis set (Supplementary Table S6).

Sets: main (18 participants, 51 blocks); without the three drifting blocks; the first ten participants analysed;
the eight added later; the vertical sacrum height as the pendulum length; without P01 and P09; without the two
recordings from later sessions; causal filtering (comparison only).
Reads   results/compare{tag}.json, analyses{tag}.json      Writes  results/sensitivity.txt, .json
"""
import json, os
from common import utf8_stdout, res, save_json, Report

utf8_stdout()
TAGS = ['', '_nodrift', '_first10', '_added', '_hvert', '_without2', '_notfirst', '_causal']
SJ = {}
w = Report()


def pf(p):
    return '<.001' if p < 0.001 else f'{p:.3f}'


def cell(d):
    return (f"{d['ML']['diff']:+4.1f}/{d['AP']['diff']:+4.1f} {d['ML']['better']:2d}/{d['AP']['better']:2d} "
            f"p{pf(d['ML']['p_w'])}/{pf(d['AP']['p_w'])}")


w("Main results in every analysis set (ML/AP; +: regression better; counts: regression better of n)")
for t in TAGS:
    if not os.path.exists(res(f'compare{t}.json')):
        w(f"{t or 'main':10s} (not run)"); continue
    C = json.load(open(res(f'compare{t}.json')))
    A = json.load(open(res(f'analyses{t}.json'))) if os.path.exists(res(f'analyses{t}.json')) else None
    g = lambda key: {ax: C[f'{key}_{ax}'] for ax in ('ML', 'AP')}
    row = dict(n=C['n_participants'], SFP=C['SFP'], all=g('slopes_MA_expand'), frozen=g('slopes_MA_frozen'),
               win10=g('slopes_MA_window10'), cal10=g('cal10'), cal20=g('cal20'), cal100=g('cal100'),
               unfreeze=g('seq1'), lag=g('seq2'), selected=C['selected'])
    SJ[t or 'main'] = row
    stop = '/'.join('K0' if C['selected'][ax] == 'MA_expand' else C['selected'][ax] for ax in ('ML', 'AP'))
    w(f"{t or 'main':10s} n={row['n']:2d}  SFP {C['SFP']['ML']:4.1f}/{C['SFP']['AP']:4.1f}")
    w(f"            all steps {cell(row['all'])} | frozen {cell(row['frozen'])} | last 10 {cell(row['win10'])}")
    w(f"            cal 10 {cell(row['cal10'])} | cal 20 {cell(row['cal20'])} | cal 100 {cell(row['cal100'])}")
    w(f"            update {row['unfreeze']['ML']['diff']:+.1f}/{row['unfreeze']['AP']['diff']:+.1f} "
      f"p{pf(row['unfreeze']['ML']['p_w'])}/{pf(row['unfreeze']['AP']['p_w'])} | K0->K1 "
      f"{row['lag']['ML']['diff']:+.1f}/{row['lag']['AP']['diff']:+.1f} "
      f"p{pf(row['lag']['ML']['p_w'])}/{pf(row['lag']['AP']['p_w'])} | stop {stop}")
    if A:
        dr = {ax: A[f'dr_{ax}'] for ax in ('ML', 'AP')}
        po = A['post_ML']
        row['dr'] = dr; row['post'] = po
        w(f"            dr {dr['ML']['dr']:+.3f}/{dr['AP']['dr']:+.3f} p{pf(dr['ML']['p'])}/{pf(dr['AP']['p'])} | "
          f"after crossings ML {po['base']:+.3f}->{po['post']:+.3f} p{pf(po['p'])}")
w("")
w("p: Wilcoxon signed-rank (dr and after crossings: paired t), participant means.")
w.save('sensitivity.txt')
save_json(SJ, 'sensitivity.json')
