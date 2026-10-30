# Stable foot placement in Parkinson's disease

Code for the paper *Pendulum mechanics predict foot placement from toe-off without fitted weights in Parkinson's
disease* by Yixuan He, Matthew A. Brodie, Juno Kim, Sjoerd M. Bruijn, Jaap H. van Dieën, Stephen R. Lord and
Yoshiro Okubo (in preparation).

The stable foot placement (SFP) is the extrapolated centre of mass (XCoM) at toe-off plus the walker's mean margin of
stability over the steps before. We compare it with the foot placement regression of Wang and Srinivasan (2014), in
which the weights on centre-of-mass position and velocity are fitted to each walker. The data come from 18 people
with Parkinson's disease walking on a treadmill in virtual reality.

## Data

The recordings are open on Zenodo, https://doi.org/10.5281/zenodo.23555055, together with a copy of this code
(version 1.0). The file `sfp_tracking_100Hz.csv.gz` has the sacrum and both feet of 18 participants, 4 min per block
at 100 Hz, with frame-by-frame quality flags. Two blocks are left out because the feet do not lie on either side of
the sacrum tracker, which leaves 18 participants and 51 blocks. Put the file in `data/`, or set the environment
variable `SFP_DATA` to its folder.

## Running

Python 3.11 and the packages in `requirements.txt`:

```
pip install -r requirements.txt
python run_all.py
```

This takes about five minutes and writes everything to `results/`. `python run_all.py --main` leaves out the
sensitivity analyses.

| script | what it does | in the paper |
|---|---|---|
| `steps.py` | gait events, per-step variables and step gates; other filter cut-offs, causal filtering and the vertical pendulum length as options | all |
| `compare.py` | SFP against the Wang & Srinivasan regression under each estimator and updating scheme, calibration length, learning curve, transfer between blocks, fitted weights, the test sequence, SPE, warm-up length, bootstrap | Table 2, Results, Tables S2, S4, S5 |
| `analyses.py` | Table 1, the margin, the velocity term with its robustness checks and shuffle control, crossing steps | Table 1, Results, Table S1 |
| `sensitivity.py` | main results in each analysis set | Table S6 |
| `supplement.py` | propagated XCoM, window length of the running mean, velocity term by block | S2, Tables S1, S3 |
| `lang_fpe.py` | foot placement error of Lang et al. in these data, against the SPE | Results |
| `crossings.py` | SFP and regression errors on obstacle-crossing steps | Results |
| `figures.py`, `figure1.py` | Figures 1–3 and S1 | |

Three checks are not in this repository because they need the VR software's event log and the full recordings,
which are not shared: the left/right assignment of the foot trackers (Methods), the crossing label against the
logged obstacle crossings, and the belt speed after obstacle collisions (Supplementary S3).

## Citing

If you use the data or the code, please cite the Zenodo record, https://doi.org/10.5281/zenodo.23555055, and the
paper once it is out.

## Licence

Code: MIT (see `LICENSE`). Data: CC BY 4.0, on Zenodo.
