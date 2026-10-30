# -*- coding: utf-8 -*-
"""Paths, participant groups and small helpers shared by the analysis scripts."""
import io, json, os, sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.environ.get('SFP_DATA', os.path.join(ROOT, 'data'))
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

CELL = ['sub', 'blk', 'foot']
WARM = 50                                   # warm-up: no step before this one is scored

# By protocol, blocks V2 and V3 had obstacles. The scene log shows obstacles in the first block of P07 and P16.
OBSTACLE_V1 = ('P07', 'P16')

# Participant and block sets for the sensitivity analyses
DRIFT_BLOCKS = 'P06-V1,P06-V2,P12-V3'       # the sacrum tracker drifted against the feet
FIRST_TEN = 'P03,P04,P05,P06,P07,P08,P10,P11,P12,P13'
ADDED_LATER = 'P01,P02,P09,P14,P15,P16,P17,P18'
WITHOUT_TWO = 'P01,P09'                     # P01: not in the session records; P09: belt perturbations suspected
NOT_FIRST = 'P07,P14'                       # recordings from a later training session


def is_obstacle_block(D):
    return (D['blk'] != 'V1') | D['sub'].isin(OBSTACLE_V1)


def utf8_stdout():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def arg(name, default=''):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def res(name):
    return os.path.join(RESULTS, name)


TRACKING = 'sfp_tracking_100Hz.csv.gz'


def load_tracking():
    path = os.path.join(DATA, TRACKING)
    if not os.path.exists(path):
        raise SystemExit(f'{TRACKING} is not in {DATA}. Put the Zenodo file there (see README) or set SFP_DATA.')
    return pd.read_csv(path)


class Report:
    """Collects printed lines and writes them to a text file."""

    def __init__(self):
        self.lines = []

    def __call__(self, s=''):
        self.lines.append(s)
        print(s, flush=True)

    def save(self, name):
        open(res(name), 'w', encoding='utf-8').write('\n'.join(self.lines) + '\n')


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def save_json(obj, name):
    json.dump(jsonable(obj), open(res(name), 'w'), indent=1)


def load_json(name):
    return json.load(open(res(name)))
