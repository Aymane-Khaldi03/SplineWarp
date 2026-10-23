"""
Plotting functions for the Vimeo-90K and Middlebury runs.

Every function takes an optional ``figures_dir`` so the two benchmarks write
to separate folders.

  1.  plot_frame_comparison       : GT + all methods + diff maps
  2.  plot_average_metrics_bar    : Mean ± std bar chart (PSNR, SSIM)
  3.  plot_box_plots              : Per-method PSNR / SSIM box plots
  4.  plot_uncertainty_map        : GPFI posterior variance heatmap
  5.  plot_psnr_histogram         : Overlaid PSNR histograms
  6.  plot_summary_table          : Metrics rendered as a table figure
  7.  plot_uncertainty_overlay    : GPFI: frame | heatmap | blended overlay
  8.  plot_motion_stratified      : PSNR / SSIM vs. motion bin (line chart)
  9.  plot_ssim_psnr_scatter      : PSNR vs. SSIM scatter, one dot per triplet
  10. plot_psnr_cdf               : Cumulative distribution of PSNR per method
  11. plot_improvement_over_linear: Delta PSNR / SSIM relative to linear baseline
  12. plot_per_sequence_bar       : Mean PSNR per sequence per method (Middlebury)

All figures are saved as PNG at 150 DPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns

from interpolation import METHODS

_DEFAULT_FIGURES_DIR = Path("figures")
DEFAULT_DPI = 150

METHOD_COLORS = {
    "linear":  "#1f77b4",
    "cubic":   "#ff7f0e",
    "bspline": "#2ca02c",
    "matsi":   "#d62728",
    "gpfi":    "#9467bd",
    "vtpsi":   "#8c564b",
    "assi":    "#7f7f7f",
    "fdsi":    "#bcbd22",
}

METHOD_LABELS = {
    "linear":  "Linear",
    "cubic":   "Cubic Spline",
    "bspline": "B-Spline",
    "matsi":   "MATSI",
    "gpfi":    "GPFI",
    "vtpsi":   "VTPSI",
    "assi":    "ASSI",
    "fdsi":    "FDSI",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


# Internal helpers

def _resolve_dirs(figures_dir: Optional[Path]):
    """Return (figures_dir, comp_dir, gpfi_dir), creating them on demand."""
    fd = Path(figures_dir) if figures_dir is not None else _DEFAULT_FIGURES_DIR
    cd = fd / "comparisons"
    gd = fd / "gpfi"
    fd.mkdir(parents=True, exist_ok=True)
    cd.mkdir(parents=True, exist_ok=True)
    gd.mkdir(parents=True, exist_ok=True)
    return fd, cd, gd


def _save(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=DEFAULT_DPI, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved %s", path)


def _method_order(df: pd.DataFrame) -> List[str]:
    return [m for m in METHODS if m in df["method"].unique()]


# 1. Frame comparison grid

def plot_frame_comparison(
    gt_frame: np.ndarray,
    interp_frames: Dict[str, np.ndarray],
    triplet_id: str,
    uncertainty: Optional[np.ndarray] = None,
    figures_dir: Optional[Path] = None,
) -> Path:
    """2-row grid: top = GT + reconstructions, bottom = abs-diff maps."""
    fd, cd, _ = _resolve_dirs(figures_dir)
    methods = [m for m in METHODS if m in interp_frames]
    n = len(methods)
    ncols = n + 1

    fig, axes = plt.subplots(2, ncols, figsize=(4 * ncols, 8))

    axes[0, 0].imshow(gt_frame)
    axes[0, 0].set_title("Ground Truth", fontsize=9)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")

    for col, method in enumerate(methods, start=1):
        pred = interp_frames[method]
        axes[0, col].imshow(pred)
        axes[0, col].set_title(METHOD_LABELS.get(method, method.upper()), fontsize=9)
        axes[0, col].axis("off")

        diff = np.mean(np.abs(gt_frame.astype(float) - pred.astype(float)), axis=2)
        im = axes[1, col].imshow(diff, cmap="hot", vmin=0, vmax=50)
        axes[1, col].set_title(f"|Diff|", fontsize=8)
        axes[1, col].axis("off")
        fig.colorbar(im, ax=axes[1, col], fraction=0.046, pad=0.04)

    fig.suptitle(f"Frame comparison: {triplet_id}", fontsize=13, y=1.01)
    fig.tight_layout()
    safe_id = triplet_id.replace("/", "_").replace("\\", "_")
    out = cd / f"comparison_{safe_id}.png"
    _save(fig, out)
    return out


# 2. Average metrics bar chart

def plot_average_metrics_bar(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Grouped bar chart of mean ± std for PSNR and SSIM."""
    fd, _, _ = _resolve_dirs(figures_dir)
    summary = df.groupby("method")[["PSNR", "SSIM"]].agg(["mean", "std"])
    methods = _method_order(df)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    x = np.arange(len(methods))

    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        means  = [summary.loc[m, (metric, "mean")] for m in methods]
        stds   = [summary.loc[m, (metric, "std")]  for m in methods]
        colors = [METHOD_COLORS.get(m, "#333")     for m in methods]
        bars = ax.bar(x, means, yerr=stds, color=colors, capsize=4,
                      alpha=0.88, edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS.get(m, m.upper()) for m in methods],
                           rotation=25, ha="right", fontsize=9)
        ax.set_title(f"Mean {metric} ± 1σ", fontsize=11)
        ax.set_ylabel(metric)
        ax.grid(axis="y", alpha=0.3)
        # Value labels on top of bars
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(means),
                    f"{mean:.2f}", ha="center", va="bottom", fontsize=7)

    fig.suptitle("Average Reconstruction Quality: All Methods", fontsize=13)
    fig.tight_layout()
    out = fd / "average_metrics_bar.png"
    _save(fig, out)
    return out


# 3. Box plots

def plot_box_plots(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Box plots of PSNR and SSIM distributions per method."""
    fd, _, _ = _resolve_dirs(figures_dir)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    order = _method_order(df)

    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        sns.boxplot(data=df, x="method", y=metric, order=order, ax=ax,
                    hue="method", hue_order=order, palette=METHOD_COLORS,
                    legend=False, flierprops={"marker": ".", "markersize": 3})
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels([METHOD_LABELS.get(m, m.upper()) for m in order],
                           rotation=25, ha="right", fontsize=9)
        ax.set_xlabel("")
        ax.set_title(f"{metric} Distribution", fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-Triplet Metric Distributions", fontsize=13)
    fig.tight_layout()
    out = fd / "box_plots.png"
    _save(fig, out)
    return out


# 4. Uncertainty map (GPFI)

def plot_uncertainty_map(
    uncertainty: np.ndarray,
    triplet_id: str,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Heatmap of GPFI posterior variance (confidence map)."""
    fd, _, gd = _resolve_dirs(figures_dir)
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(uncertainty, cmap="inferno")
    ax.set_title(f"GPFI Uncertainty: {triplet_id}")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Posterior variance")
    fig.tight_layout()
    safe_id = triplet_id.replace("/", "_").replace("\\", "_")
    out = gd / f"uncertainty_{safe_id}.png"
    _save(fig, out)
    return out


# 5. PSNR histogram

def plot_psnr_histogram(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Overlaid PSNR histograms per method."""
    fd, _, _ = _resolve_dirs(figures_dir)
    fig, ax = plt.subplots(figsize=(10, 5))
    for method in METHODS:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        ax.hist(sub["PSNR"], bins=30, alpha=0.5,
                label=METHOD_LABELS.get(method, method.upper()),
                color=METHOD_COLORS.get(method))
    ax.set_xlabel("PSNR (dB)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title("PSNR Distribution by Method", fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = fd / "psnr_histogram.png"
    _save(fig, out)
    return out


# 6. Summary table figure

def plot_summary_table(
    summary_df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Render the summary metrics as a publication-ready table figure."""
    fd, _, _ = _resolve_dirs(figures_dir)
    methods = [m for m in METHODS if m in summary_df.index]

    cell_text, row_colors = [], []
    for m in methods:
        row = []
        for metric in ["MSE", "PSNR", "SSIM"]:
            mean = summary_df.loc[m, (metric, "mean")]
            std  = summary_df.loc[m, (metric, "std")]
            row.append(f"{mean:.4f} ± {std:.4f}")
        cell_text.append(row)
        # Shade the motion-compensated methods
        is_novel = m in {"matsi", "gpfi", "vtpsi", "assi", "fdsi"}
        row_colors.append(["#f0f8ff"] * 3 if is_novel else ["#ffffff"] * 3)

    fig, ax = plt.subplots(figsize=(11, 1.8 + 0.45 * len(methods)))
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        cellColours=row_colors,
        rowLabels=[METHOD_LABELS.get(m, m.upper()) for m in methods],
        colLabels=["MSE ↓", "PSNR (dB) ↑", "SSIM ↑"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)
    ax.set_title("Quantitative Results: Mean ± Std (blue rows = motion-compensated)",
                 fontsize=12, pad=20)
    fig.tight_layout()
    out = fd / "summary_table.png"
    _save(fig, out)
    return out


# 7. Uncertainty overlay (GPFI)

def plot_uncertainty_overlay(
    interp_frame: np.ndarray,
    uncertainty: np.ndarray,
    triplet_id: str,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Three-panel: GPFI interpolated frame | raw heatmap | blended overlay."""
    fd, _, gd = _resolve_dirs(figures_dir)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].imshow(interp_frame)
    axes[0].set_title("GPFI Interpolated Frame", fontsize=11)
    axes[0].axis("off")

    vmax = float(np.percentile(uncertainty, 99))
    im = axes[1].imshow(uncertainty, cmap="inferno", vmin=0, vmax=vmax)
    axes[1].set_title(r"Posterior Variance $\sigma^2(x,y)$", fontsize=11)
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04, label="Variance")

    u_norm  = np.clip(uncertainty / (vmax + 1e-10), 0.0, 1.0)
    unc_rgb = plt.get_cmap("inferno")(u_norm)[..., :3]
    frame_f = interp_frame.astype(np.float32) / 255.0
    alpha   = 0.45
    overlay = np.clip((1.0 - alpha) * frame_f + alpha * unc_rgb, 0.0, 1.0)
    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (α={alpha}), brighter = higher uncertainty",
                      fontsize=11)
    axes[2].axis("off")

    fig.suptitle(f"GPFI Uncertainty Analysis: {triplet_id}", fontsize=13, y=1.01)
    fig.tight_layout()
    safe_id = triplet_id.replace("/", "_").replace("\\", "_")
    out = gd / f"uncertainty_overlay_{safe_id}.png"
    _save(fig, out)
    return out


# 8. Motion-stratified line chart

def plot_motion_stratified(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Line chart (±1σ) of PSNR and SSIM per motion bin."""
    if "motion_bin" not in df.columns:
        LOGGER.warning("motion_bin column missing: skipping stratified plot")
        return None

    fd, _, _ = _resolve_dirs(figures_dir)
    bin_order = ["Slow", "Moderate", "Fast"]
    methods   = _method_order(df)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric in zip(axes, ["PSNR", "SSIM"]):
        for method in methods:
            sub = df[df["method"] == method]
            means, errs = [], []
            for bn in bin_order:
                bsub = sub[sub["motion_bin"] == bn][metric]
                means.append(float(bsub.mean()) if not bsub.empty else np.nan)
                errs.append(float(bsub.std())  if not bsub.empty else np.nan)
            ax.errorbar(
                bin_order, means, yerr=errs,
                marker="o", linewidth=2, capsize=4,
                label=METHOD_LABELS.get(method, method.upper()),
                color=METHOD_COLORS.get(method, "#333"),
            )
        ax.set_title(f"Mean {metric} vs. Motion Level", fontsize=11)
        ax.set_ylabel(metric)
        ax.set_xlabel("Motion Category")
        ax.legend(loc="best", fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("Motion-Stratified Evaluation", fontsize=13)
    fig.tight_layout()
    out = fd / "motion_stratified.png"
    _save(fig, out)
    return out


# 9. PSNR vs SSIM scatter

def plot_ssim_psnr_scatter(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Scatter of PSNR vs SSIM: one point per (triplet, method), colour-coded."""
    fd, _, _ = _resolve_dirs(figures_dir)
    methods = _method_order(df)

    fig, ax = plt.subplots(figsize=(9, 6))
    for method in methods:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["SSIM"], sub["PSNR"],
            s=8, alpha=0.35,
            color=METHOD_COLORS.get(method, "#333"),
            label=METHOD_LABELS.get(method, method.upper()),
        )
    # Draw mean markers on top
    for method in methods:
        sub = df[df["method"] == method]
        ax.scatter(
            sub["SSIM"].mean(), sub["PSNR"].mean(),
            s=120, marker="D",
            color=METHOD_COLORS.get(method, "#333"),
            edgecolors="black", linewidths=0.8, zorder=5,
        )

    ax.set_xlabel("SSIM ↑", fontsize=11)
    ax.set_ylabel("PSNR (dB) ↑", fontsize=11)
    ax.set_title("PSNR vs. SSIM Trade-off (diamonds = method means)", fontsize=12)
    ax.legend(fontsize=8, markerscale=2, loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = fd / "ssim_psnr_scatter.png"
    _save(fig, out)
    return out


# 10. PSNR CDF

def plot_psnr_cdf(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Path:
    """Empirical CDF of PSNR per method: shows how often each method achieves
    a given quality threshold across all triplets."""
    fd, _, _ = _resolve_dirs(figures_dir)
    methods = _method_order(df)

    fig, ax = plt.subplots(figsize=(9, 5))
    for method in methods:
        sub = df[df["method"] == method]["PSNR"].dropna().sort_values()
        if sub.empty:
            continue
        cdf = np.arange(1, len(sub) + 1) / len(sub)
        ax.plot(sub.values, cdf,
                color=METHOD_COLORS.get(method, "#333"),
                linewidth=2,
                label=METHOD_LABELS.get(method, method.upper()))

    ax.set_xlabel("PSNR threshold (dB)", fontsize=11)
    ax.set_ylabel("Fraction of triplets ≤ threshold", fontsize=11)
    ax.set_title("Cumulative Distribution of PSNR", fontsize=12)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    fig.tight_layout()
    out = fd / "psnr_cdf.png"
    _save(fig, out)
    return out


# 11. Improvement over linear baseline

def plot_improvement_over_linear(
    df: pd.DataFrame,
    figures_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Delta PSNR and Delta SSIM relative to the linear baseline.

    Shows clearly which methods beat the simplest possible approach and by how much.
    """
    if "linear" not in df["method"].values:
        LOGGER.warning("linear method not found: skipping improvement plot")
        return None

    fd, _, _ = _resolve_dirs(figures_dir)
    methods = [m for m in _method_order(df) if m != "linear"]

    summary = df.groupby("method")[["PSNR", "SSIM"]].mean()
    lin_psnr = summary.loc["linear", "PSNR"]
    lin_ssim = summary.loc["linear", "SSIM"]

    delta_psnr = [summary.loc[m, "PSNR"] - lin_psnr for m in methods]
    delta_ssim = [summary.loc[m, "SSIM"] - lin_ssim for m in methods]

    x = np.arange(len(methods))
    labels = [METHOD_LABELS.get(m, m.upper()) for m in methods]
    colors = [METHOD_COLORS.get(m, "#333") for m in methods]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, deltas, metric, unit in zip(
        axes,
        [delta_psnr, delta_ssim],
        ["PSNR", "SSIM"],
        [" (dB)", ""],
    ):
        bar_colors = [c if d >= 0 else "#cccccc" for c, d in zip(colors, deltas)]
        bars = ax.bar(x, deltas, color=bar_colors, edgecolor="white",
                      linewidth=0.5, alpha=0.9)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(f"Δ{metric}{unit}", fontsize=11)
        ax.set_title(f"Δ{metric} vs. Linear Baseline", fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for bar, d in zip(bars, deltas):
            va = "bottom" if d >= 0 else "top"
            offset = 0.001 * (max(deltas) - min(deltas)) * (1 if d >= 0 else -1)
            ax.text(bar.get_x() + bar.get_width() / 2, d + offset,
                    f"{d:+.3f}", ha="center", va=va, fontsize=7)

    fig.suptitle("Improvement over Linear Baseline (grey bars = worse than linear)",
                 fontsize=12)
    fig.tight_layout()
    out = fd / "improvement_over_linear.png"
    _save(fig, out)
    return out


# 12. Per-sequence bars (Middlebury)

def plot_per_sequence_bar(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    figures_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Mean PSNR per sequence per method: most useful for Middlebury where
    each sequence has a meaningful name.

    Requires a ``sequence`` column (or whichever name is passed as *seq_col*)
    in *df*.
    """
    if seq_col not in df.columns:
        LOGGER.warning("%s column not found: skipping per-sequence plot", seq_col)
        return None

    fd, _, _ = _resolve_dirs(figures_dir)
    sequences = sorted(df[seq_col].unique())
    methods   = _method_order(df)
    n_seq     = len(sequences)
    n_met     = len(methods)

    pivot = df.groupby([seq_col, "method"])["PSNR"].mean().unstack("method")

    fig, ax = plt.subplots(figsize=(max(10, n_seq * 1.2), 6))
    x = np.arange(n_seq)
    width = 0.8 / n_met

    for i, method in enumerate(methods):
        if method not in pivot.columns:
            continue
        vals = [pivot.loc[s, method] if s in pivot.index else np.nan
                for s in sequences]
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.92,
               color=METHOD_COLORS.get(method, "#333"),
               label=METHOD_LABELS.get(method, method.upper()),
               alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean PSNR (dB)", fontsize=11)
    ax.set_title("Mean PSNR per Sequence: All Methods", fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fd / "per_sequence_psnr.png"
    _save(fig, out)
    return out


def plot_per_sequence_ssim(
    df: pd.DataFrame,
    seq_col: str = "sequence",
    figures_dir: Optional[Path] = None,
) -> Optional[Path]:
    """Mean SSIM per sequence per method: companion to per_sequence_psnr."""
    if seq_col not in df.columns:
        return None

    fd, _, _ = _resolve_dirs(figures_dir)
    sequences = sorted(df[seq_col].unique())
    methods   = _method_order(df)
    n_seq     = len(sequences)
    n_met     = len(methods)

    pivot = df.groupby([seq_col, "method"])["SSIM"].mean().unstack("method")

    fig, ax = plt.subplots(figsize=(max(10, n_seq * 1.2), 6))
    x = np.arange(n_seq)
    width = 0.8 / n_met

    for i, method in enumerate(methods):
        if method not in pivot.columns:
            continue
        vals = [pivot.loc[s, method] if s in pivot.index else np.nan
                for s in sequences]
        offset = (i - n_met / 2 + 0.5) * width
        ax.bar(x + offset, vals, width=width * 0.92,
               color=METHOD_COLORS.get(method, "#333"),
               label=METHOD_LABELS.get(method, method.upper()),
               alpha=0.88)

    ax.set_xticks(x)
    ax.set_xticklabels(sequences, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean SSIM", fontsize=11)
    ax.set_title("Mean SSIM per Sequence: All Methods", fontsize=12)
    ax.legend(fontsize=8, ncol=3)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = fd / "per_sequence_ssim.png"
    _save(fig, out)
    return out
