"""Draws the README figures into assets/. Run from the project root: python assets/make_readme_figures.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd

from interpolation import _compute_flow, linear_interpolate, matsi_interpolate
from metrics import compute_psnr

OUT = Path("assets")
OUT.mkdir(exist_ok=True)

# Colours (colour-blind safe blue / orange / aqua)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.titlesize": 13,
    "axes.titleweight": "semibold",
    "axes.grid": False,
})


def style_axes(ax, grid_axis="y"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(AXIS)
    ax.spines["bottom"].set_color(AXIS)
    ax.grid(axis=grid_axis, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("saved", OUT / name)


def load_rgb(p):
    return cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)


# 1. Hero: linear vs MATSI on Hydrangea/frame08
seq = Path("other-color-allframes/other-data/Hydrangea")
I0, GT, I2 = (load_rgb(seq / f"frame{n:02d}.png") for n in (7, 8, 9))
flow = _compute_flow(I0, I2)
lin = linear_interpolate(I0, I2)
mat = matsi_interpolate(I0, I2, precomputed_flow=flow)
err = lambda p: np.mean(np.abs(GT.astype(float) - p.astype(float)), axis=2)
mag = np.linalg.norm(flow, axis=2)

fig, axes = plt.subplots(2, 3, figsize=(15, 8.2))
panels = [
    (GT, "Ground truth (t = 0.5)"),
    (lin, f"Linear / Cubic / B-spline  ·  {compute_psnr(GT, lin):.1f} dB"),
    (mat, f"MATSI (motion-compensated)  ·  {compute_psnr(GT, mat):.1f} dB"),
]
for ax, (img, title) in zip(axes[0], panels):
    ax.imshow(img)
    ax.set_title(title, fontsize=12)
    ax.axis("off")

flow_cmap = LinearSegmentedColormap.from_list("blue_seq", BLUE_RAMP)
im = axes[1, 0].imshow(mag, cmap=flow_cmap, vmin=0, vmax=np.percentile(mag, 99))
axes[1, 0].set_title("Optical-flow magnitude (px)", fontsize=12)
cb = fig.colorbar(im, ax=axes[1, 0], fraction=0.035, pad=0.02)
cb.outline.set_visible(False)
for ax, e, title in ((axes[1, 1], err(lin), "Error: linear (ghosting)"),
                     (axes[1, 2], err(mat), "Error: MATSI")):
    im = ax.imshow(e, cmap="inferno", vmin=0, vmax=60)
    ax.set_title(title, fontsize=12)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.outline.set_visible(False)
for ax in axes[1]:
    ax.axis("off")
fig.suptitle("Middlebury Hydrangea: predicting frame 08 from frames 07 and 09",
             fontsize=14, color=INK, y=0.99)
fig.tight_layout()
save(fig, "hero_comparison.png")

# 2. PSNR by method, both benchmarks
vimeo = pd.read_csv("outputs/results.csv")
midd = pd.read_csv("outputs_middlebury/results.csv")
order = ["linear", "fdsi", "assi", "vtpsi", "gpfi", "matsi"]
labels = {"linear": "Linear = Cubic = B-spline", "matsi": "MATSI", "gpfi": "GPFI",
          "vtpsi": "VTPSI", "assi": "ASSI", "fdsi": "FDSI"}
vm = vimeo.groupby("method").PSNR.mean()
mm = midd.groupby("method").PSNR.mean()

fig, ax = plt.subplots(figsize=(10, 5.2))
y = np.arange(len(order))
h = 0.36
for vals, off, color, name in ((vm, h / 2 + 0.02, BLUE, "Vimeo-90K (3,782 triplets)"),
                               (mm, -h / 2 - 0.02, ORANGE, "Middlebury (60 triplets)")):
    v = [vals[m] for m in order]
    ax.barh(y + off, v, height=h, color=color, label=name)
    for yi, vi in zip(y + off, v):
        ax.text(vi + 0.3, yi, f"{vi:.2f}", va="center", fontsize=10, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels([labels[m] for m in order], fontsize=11)
ax.set_xlim(0, 38)
ax.set_xlabel("Mean PSNR (dB), higher is better")
ax.set_title("Motion compensation adds about 7 dB over classical splines", loc="left")
style_axes(ax, grid_axis="x")
ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.13), ncol=2, frameon=False,
          fontsize=10, labelcolor=INK2)
save(fig, "psnr_by_method.png")

# 3. Motion robustness (Vimeo)
bins = ["Slow", "Moderate", "Fast"]
strat = vimeo.pivot_table(index="method", columns="motion_bin", values="PSNR")[bins]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(3)
for m, color, name in (("matsi", BLUE, "MATSI / GPFI / VTPSI / ASSI"),
                       ("fdsi", AQUA, "FDSI"),
                       ("linear", ORANGE, "Linear / Cubic / B-spline")):
    v = strat.loc[m].values
    ax.plot(x, v, color=color, linewidth=2.2, marker="o", markersize=8,
            markeredgecolor=SURFACE, markeredgewidth=2, label=name)
    ax.text(2.08, v[-1], f"{name}\n{v[0]:.1f} → {v[-1]:.1f} dB", va="center",
            fontsize=10, color=INK2)
ax.set_xticks(x)
ax.set_xticklabels(["Slow\n(≤ 3.8 px)", "Moderate\n(3.8-6.4 px)", "Fast\n(> 6.4 px)"])
ax.set_xlim(-0.15, 2.9)
ax.set_ylim(20, 35)
ax.set_ylabel("Mean PSNR (dB)")
ax.set_title("Classical splines lose 5.3 dB as motion grows; warping methods stay within 1 dB",
             loc="left")
style_axes(ax)
ax.legend(loc="lower left", frameon=False, fontsize=10, labelcolor=INK2)
save(fig, "motion_robustness.png")

# 4. Per-sequence dumbbell (Middlebury)
seqp = midd.pivot_table(index="sequence", columns="method", values="PSNR")
seqp = seqp.assign(gain=seqp.matsi - seqp.linear).sort_values("gain")
fig, ax = plt.subplots(figsize=(10, 5.8))
y = np.arange(len(seqp))
ax.hlines(y, seqp.linear, seqp[["matsi", "fdsi"]].max(axis=1), color=AXIS, linewidth=2)
for col, color, name in (("linear", ORANGE, "Linear / Cubic / B-spline"),
                         ("fdsi", AQUA, "FDSI"),
                         ("matsi", BLUE, "MATSI")):
    ax.scatter(seqp[col], y, s=80, color=color, edgecolor=SURFACE, linewidth=2,
               zorder=3, label=name)
for yi, (_, r) in zip(y, seqp.iterrows()):
    ax.text(max(r.matsi, r.fdsi) + 0.5, yi, f"+{r.gain:.1f} dB", va="center",
            fontsize=10, color=INK2)
ax.set_yticks(y)
ax.set_yticklabels(seqp.index)
ax.set_xlim(16, 44)
ax.set_xlabel("Mean PSNR (dB)")
ax.set_title("Middlebury, per sequence: MATSI gain over linear", loc="left")
style_axes(ax, grid_axis="x")
ax.legend(loc="lower right", frameon=False, fontsize=10, labelcolor=INK2)
save(fig, "per_sequence_gain.png")
