# -*- coding: utf-8 -*-
"""Run the whole analysis in order (about 5 minutes on a laptop).

python run_all.py          everything
python run_all.py --main   the main analysis only (no sensitivity sets)
"""
import os, subprocess, sys
import common

HERE = os.path.dirname(os.path.abspath(__file__))
MAIN_ONLY = '--main' in sys.argv


def run(*args):
    print('>', 'python', *args, flush=True)
    with open(os.path.join(common.RESULTS, 'log_' + os.path.splitext(args[0])[0] + '.txt'), 'a', encoding='utf-8') as log:
        subprocess.run([sys.executable, *args], cwd=HERE, check=True, stdout=log, stderr=subprocess.STDOUT,
                       env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})


for f in os.listdir(common.RESULTS):
    if f.startswith('log_'):
        os.remove(os.path.join(common.RESULTS, f))

run('steps.py')
if not MAIN_ONLY:
    for opt in (['--cut', '4'], ['--cut', '10'], ['--causal'], ['--legdef', 'vertical']):
        run('steps.py', *opt)
run('compare.py')
run('analyses.py')
if not MAIN_ONLY:
    SETS = [('_nodrift', '--drop', common.DRIFT_BLOCKS), ('_first10', '--keep-subs', common.FIRST_TEN),
            ('_added', '--keep-subs', common.ADDED_LATER), ('_without2', '--drop', common.WITHOUT_TWO),
            ('_notfirst', '--drop', common.NOT_FIRST)]
    for tag, opt, val in SETS:
        run('compare.py', '--quick', opt, val, '--tag', tag)
        run('analyses.py', '--quick', opt, val, '--tag', tag)
    run('compare.py', '--quick', '--steps', '_hvert', '--tag', '_hvert')
    run('analyses.py', '--quick', '--steps', '_hvert', '--tag', '_hvert')
    run('compare.py', '--quick', '--steps', '_causal', '--tag', '_causal')
    run('sensitivity.py')
run('supplement.py')
run('lang_fpe.py')
run('crossings.py')
run('figures.py')
run('figure1.py')
print('done; outputs in', common.RESULTS)
