# -*- coding: utf-8 -*-
"""Figure 1: (a) the offset applied to the next step, schematic; (b) the target across the step cycle.
Writes results/fig1.png"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Ellipse, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 11, "mathtext.fontset": "cm", "figure.dpi": 200})
NAVY = "#0E2A3B"; TEAL = "#1C7293"; AMBER = "#D98A1F"; RED = "#B23A48"
GREY = "#8794a0"; GROUND = "#3a3a3a"


def arrow(ax, p, q, color=NAVY, lw=1.8, ms=11, style='-|>'):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=ms, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=7))


def dim(ax, p, q, color=GREY, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle='<|-|>', mutation_scale=7, lw=lw, color=color,
                                 shrinkA=0, shrinkB=0, zorder=4))


def footF(ax, xc, y=0.0, w=0.26, h=0.075, fc="white", ec=NAVY, ls='-', lw=1.8, z=3):
    ax.add_patch(FancyBboxPatch((xc - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.035",
                                fc=fc, ec=ec, lw=lw, ls=ls, zorder=z))


def txt(ax, x, y, s, color=NAVY, size=10.5, ha="center", va="center", weight="normal", z=8):
    ax.text(x, y, s, color=color, fontsize=size, ha=ha, va=va, weight=weight, zorder=z)


fig, (axA, axB) = plt.subplots(1, 2, figsize=(9.9, 4.1),
                               gridspec_kw={'width_ratios': [1.0, 1.25]})

# ------------------------------------------------------------ (a) the offset (illustrative values, not data)
# the seed gives a series without a trend whose first five steps sit visibly off its mean
rng = np.random.default_rng(236)
n = 13                                                   # observed steps at x = 0 .. n-1
mos = 60 + rng.normal(0, 7, n)                           # a constant margin plus step-to-step noise
const = mos[:5].mean()          # frozen on a short calibration
fc = mos.mean()                 # mean of every step so far
axA.axhline(const, color=RED, lw=1.6, zorder=3)
axA.plot(np.arange(n), mos, '-', lw=0.9, color=GREY, alpha=0.6, zorder=3)
axA.plot(np.arange(n), mos, 'o', ms=5.5, color=GREY, zorder=4)
axA.plot([n], [fc], 'o', ms=9, mfc='white', mec=TEAL, mew=2.4, zorder=6)
lo = min(mos.min(), fc)
arrow(axA, (n - 4.6, lo - 3.4), (n - 0.35, fc - 1.8), TEAL, 1.5, 10)
txt(axA, n - 5.0, lo - 3.6, "re-estimated over\nevery step so far", TEAL, 9.2, ha='right')
txt(axA, n + 0.5, fc, r"$M^*_{n+1}$", TEAL, 11, ha='left')
txt(axA, 0.0, const + 1.6, "frozen on a short calibration", RED, 9, ha='left')
axA.set_xlim(-1.4, n + 3.4)
axA.set_ylim(lo - 8, max(mos.max(), const) + 4)
axA.set_xticks([0, n - 1, n])
axA.set_xticklabels(['1', r'$n$', r'$n{+}1$'])
axA.set_xlabel("step")
axA.set_ylabel("margin of stability  " + r"$\mathrm{MoS}_n$" + "  (schematic)")
axA.set_title("(a)  The offset applied to step $n{+}1$ (schematic)",
              fontsize=11, color=NAVY, pad=6, loc='left')
for sp in ('top', 'right'):
    axA.spines[sp].set_visible(False)

# ------------------------------------------------------------ (b) across the step cycle
axB.set_xlim(-0.35, 1.95); axB.set_ylim(-0.62, 1.50); axB.axis("off"); axB.set_aspect("equal")
axB.plot([-0.35, 1.95], [0, 0], color=GROUND, lw=2, zorder=2)
fo, cxt, h = 0.0, 0.30, 1.05
footF(axB, fo, w=0.24)
axB.plot([fo, cxt], [0, h], color=NAVY, lw=5.0, solid_capstyle='round', zorder=4)
axB.add_patch(Ellipse((fo, 0.0), 0.05, 0.05, fc='white', ec=NAVY, lw=2.0, zorder=5))
axB.add_patch(Ellipse((cxt, h), 0.17, 0.17, fc=NAVY, ec='white', lw=1.4, zorder=6))
txt(axB, cxt - 0.02, h + 0.14, "CoM at toe-off", NAVY, 9)
arrow(axB, (cxt, h), (cxt + 0.30, h), TEAL, 1.9)
txt(axB, cxt + 0.34, h + 0.03, r"$\dot r$", TEAL, 11, ha='left')
txt(axB, 1.15, h + 0.30, r"XCoM $\xi = r + \dot r/\omega_0$  (Eq 1)", TEAL, 9.6)
xa = np.linspace(-0.05, 1.38, 60); ya = 0.42 * np.sin(np.pi * (xa + 0.05) / 1.43); ya[ya < 0] = 0
axB.plot(xa, ya, color=GREY, lw=1.3, ls=(0, (5, 3)), zorder=3)
txt(axB, 0.66, 0.24, "swing foot", GREY, 9)
xi = 1.05
axB.plot([xi], [0], marker='*', ms=15, color=TEAL, zorder=7)
axB.plot([xi, xi], [0, 0.95], color=TEAL, lw=1.0, ls=(0, (4, 3)), zorder=3)
axB.plot([cxt, xi], [h, 0.30], color=TEAL, lw=2.0, ls=(0, (6, 3)), zorder=3)
xsfp = 1.58
axB.plot([xsfp], [0], marker='x', ms=11, mew=2.4, color=AMBER, zorder=8)
txt(axB, xsfp, 0.17, "SFP", AMBER, 9.6)
axB.plot([cxt, 1.47], [h, 0], color=TEAL, lw=2.4, ls=(0, (4, 2)), zorder=3)
arrow(axB, (xi, 0.60), (xsfp, 0.60), AMBER, 1.7, 9)
txt(axB, (xi + xsfp) / 2, 0.70, r"$b^*_{n+1}$  (Eq 3)", AMBER, 10)
xact = 1.47
footF(axB, xact, w=0.24, fc="white", ec=NAVY, lw=2.0)
txt(axB, xact, -0.21, "actual landing", NAVY, 9)
dim(axB, (xact, 0.32), (xsfp, 0.32), color=RED)
txt(axB, (xact + xsfp) / 2, 0.41, "SPE", RED, 8.8, weight='bold')
y0 = -0.46
axB.add_patch(FancyArrowPatch((cxt, y0), (xact, y0), arrowstyle='-', lw=6, color="#dfe7ec",
                              shrinkA=0, shrinkB=0, zorder=2))
for xx, lab in [(cxt, "toe-off"), (xact, "heel strike")]:
    axB.plot([xx], [y0], marker='|', ms=13, color=NAVY, zorder=4)
    txt(axB, xx, y0 - 0.12, lab, NAVY, 8.8)
txt(axB, (cxt + xact) / 2, y0 + 0.11, r"swing  $T_c$", NAVY, 9)
axB.set_title("(b)  Across the step cycle (fore–aft)", fontsize=11, color=NAVY, pad=6, loc='left')

fig.tight_layout()
out = os.path.join(HERE, "results", "fig1.png")
fig.savefig(out, bbox_inches="tight")
print("wrote results/fig1.png")
