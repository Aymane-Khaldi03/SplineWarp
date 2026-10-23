"""
Evaluate all interpolation methods on the Middlebury
"other-color-allframes" frames.

Dataset layout expected (relative to --data-root):
    <data-root>/
        Beanbags/    frame07.png … frame14.png
        Dimetrodon/  frame10.png  frame11.png
        DogDance/    frame07.png … frame14.png
        Grove2/      frame07.png … frame14.png
        ...

For each sequence that has at least 3 frames, every consecutive triple
  (frameN, frameN+1, frameN+2) yields one evaluation triplet:
      I0 = frameN,   I1_gt = frameN+1,   I2 = frameN+2

Outputs (separate from Vimeo-90K results):
    outputs_middlebury/
        results.csv         per-triplet metrics  (all methods x all triplets)
        summary.csv         mean ± std per method
    figures_middlebury/
        average_metrics_bar.png
        box_plots.png
        psnr_histogram.png
        summary_table.png
        ssim_psnr_scatter.png
        psnr_cdf.png
        improvement_over_linear.png
        per_sequence_psnr.png
        per_sequence_ssim.png
        comparisons/        visual grids (with --vis)
        gpfi/               GPFI uncertainty maps (with --vis)

Usage:
    python eval_middlebury.py
    python eval_middlebury.py --vis
    python eval_middlebury.py --data-root other-color-allframes/other-data
    python eval_middlebury.py --output-dir outputs_middlebury --figures-dir figures_middlebury
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from interpolation import (
    METHODS, METHODS_META, _compute_flow, _CUDA,
    gpfi_interpolate, interpolate,
)
from metrics import compute_mse, compute_psnr, compute_ssim, summarise_metrics
from visualisation import (
    plot_average_metrics_bar,
    plot_box_plots,
    plot_frame_comparison,
    plot_improvement_over_linear,
    plot_per_sequence_bar,
    plot_per_sequence_ssim,
    plot_psnr_cdf,
    plot_psnr_histogram,
    plot_ssim_psnr_scatter,
    plot_summary_table,
    plot_uncertainty_map,
    plot_uncertainty_overlay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

DEFAULT_DATA_ROOT   = "other-color-allframes/other-data"
DEFAULT_OUTPUT_DIR  = "outputs_middlebury"
DEFAULT_FIGURES_DIR = "figures_middlebury"


# Dataset helpers

def _load_rgb(path: Path) -> np.ndarray:
    """Load PNG as uint8 RGB (H, W, 3)."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _parse_frame_number(fname: str) -> int:
    """Extract integer from filenames like 'frame07.png'."""
    stem = Path(fname).stem          # e.g. "frame07"
    digits = "".join(c for c in stem if c.isdigit())
    return int(digits)


def build_triplets(data_root: Path) -> List[Tuple[str, Path, Path, Path]]:
    """Scan data_root for sequences and build evaluation triplets.

    Returns a list of (triplet_id, I0_path, I1_gt_path, I2_path) tuples.
    triplet_id format: "<sequence>/<frameN+1_stem>"
    """
    triplets = []
    for seq_dir in sorted(data_root.iterdir()):
        if not seq_dir.is_dir():
            continue
        frames = sorted(
            [f for f in seq_dir.iterdir() if f.suffix.lower() == ".png"],
            key=lambda f: _parse_frame_number(f.name),
        )
        if len(frames) < 3:
            LOGGER.info("Skipping %s: only %d frame(s), need at least 3",
                        seq_dir.name, len(frames))
            continue
        for i in range(len(frames) - 2):
            f0, f1_gt, f2 = frames[i], frames[i + 1], frames[i + 2]
            tid = f"{seq_dir.name}/{f1_gt.stem}"
            triplets.append((tid, f0, f1_gt, f2))
    return triplets


# Single-triplet evaluation

def eval_triplet(
    tid: str,
    I0_path: Path,
    I1_gt_path: Path,
    I2_path: Path,
    do_vis: bool = False,
) -> Tuple[List[dict], Optional[Dict[str, np.ndarray]],
           Optional[np.ndarray], Optional[np.ndarray]]:
    """Evaluate every method on one triplet."""
    I0    = _load_rgb(I0_path)
    I1_gt = _load_rgb(I1_gt_path)
    I2    = _load_rgb(I2_path)

    flow     = _compute_flow(I0, I2)
    flow_mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

    # Extract sequence name from tid ("Beanbags/frame08" -> "Beanbags")
    sequence = tid.split("/")[0]

    preds: Dict[str, np.ndarray] = {}
    rows: List[dict] = []
    uncertainty = None

    for method in METHODS:
        pred = interpolate(method, I0, I2, precomputed_flow=flow)
        preds[method] = pred
        rows.append({
            "triplet_id":     tid,
            "sequence":       sequence,
            "method":         method,
            "flow_magnitude": flow_mag,
            "MSE":            compute_mse(I1_gt, pred),
            "PSNR":           compute_psnr(I1_gt, pred),
            "SSIM":           compute_ssim(I1_gt, pred),
        })

    if do_vis:
        _, uncertainty = gpfi_interpolate(I0, I2, precomputed_flow=flow)

    return rows, preds, I1_gt, uncertainty


# CLI

def parse_args():
    parser = argparse.ArgumentParser(
        description="Middlebury Other-Color-AllFrames Evaluation")
    parser.add_argument("--data-root", type=str, default=DEFAULT_DATA_ROOT,
                        help=f"Path to other-data directory (default: {DEFAULT_DATA_ROOT})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory for CSVs (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--figures-dir", type=str, default=DEFAULT_FIGURES_DIR,
                        help=f"Output directory for figures (default: {DEFAULT_FIGURES_DIR})")
    parser.add_argument("--vis", action="store_true",
                        help="Save visual comparison figures and GPFI uncertainty maps")
    parser.add_argument("--vis-samples", type=int, default=5,
                        help="Number of triplets to visualise when --vis is set (default: 5)")
    return parser.parse_args()


# Main

def main():
    args        = parse_args()
    data_root   = Path(args.data_root)
    output_dir  = Path(args.output_dir)
    figures_dir = Path(args.figures_dir)

    print("=" * 70)
    print("  Spline Frame Interpolation: Middlebury Other Evaluation")
    print("=" * 70)
    if _CUDA:
        try:
            import torch
            print(f"  GPU: {torch.cuda.get_device_name(0)}  (CUDA enabled)")
        except Exception:
            pass
    print(f"  Data root:   {data_root.resolve()}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Figures dir: {figures_dir}")

    # Step 1: Discover triplets
    print("\n[1] Scanning dataset...")
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root.resolve()}\n"
            "Pass the correct path with --data-root"
        )
    triplets = build_triplets(data_root)
    n = len(triplets)
    if n == 0:
        raise RuntimeError("No valid triplets found. Check --data-root.")
    print(f"    Found {n} evaluation triplets across "
          f"{len({t[0].split('/')[0] for t in triplets})} sequences:")
    seq_counts: Dict[str, int] = {}
    for tid, *_ in triplets:
        seq = tid.split("/")[0]
        seq_counts[seq] = seq_counts.get(seq, 0) + 1
    for seq, cnt in sorted(seq_counts.items()):
        print(f"      {seq:<20} {cnt} triplets")

    vis_ids = {t[0] for t in triplets[:args.vis_samples]} if args.vis else set()

    # Step 2: Evaluate
    print(f"\n[2] Evaluating {len(METHODS)} methods on {n} triplets...")
    t_start = time.perf_counter()

    all_rows: List[dict] = []
    vis_results: Dict[str, tuple] = {}

    iterable = tqdm(triplets, unit="triplet") if _TQDM else triplets
    for i, (tid, I0_path, I1_gt_path, I2_path) in enumerate(iterable):
        if not _TQDM and (i % 5 == 0 or i == 0):
            print(f"    [{i+1}/{n}] {tid}")
        do_vis = tid in vis_ids
        rows, preds, I1_gt, uncertainty = eval_triplet(
            tid, I0_path, I1_gt_path, I2_path, do_vis=do_vis
        )
        all_rows.extend(rows)
        if do_vis and preds is not None:
            vis_results[tid] = (preds, I1_gt, uncertainty)

    elapsed = time.perf_counter() - t_start
    print(f"    Completed in {elapsed:.1f}s  ({elapsed/n*1000:.1f} ms/triplet)")

    # Step 3: Visualisations
    if args.vis and vis_results:
        print("\n[3] Saving visual comparisons...")
        for tid, (preds, I1_gt, uncertainty) in vis_results.items():
            plot_frame_comparison(I1_gt, preds, tid, uncertainty,
                                  figures_dir=figures_dir)
            if uncertainty is not None:
                plot_uncertainty_map(uncertainty, tid, figures_dir=figures_dir)
                if "gpfi" in preds:
                    plot_uncertainty_overlay(preds["gpfi"], uncertainty, tid,
                                             figures_dir=figures_dir)

    # Step 4: Summarise
    print("\n[4] Computing summary statistics...")
    df = pd.DataFrame(all_rows)
    df.rename(columns={"triplet_id": "frame_index"}, inplace=True)

    # Restore sequence column for per-sequence plots
    df["sequence"] = df["frame_index"].str.split("/").str[0]

    summary = summarise_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path    = output_dir / "results.csv"
    summary_csv = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv)
    print(f"    Raw results  -> {csv_path}")
    print(f"    Summary      -> {summary_csv}")

    # Step 5: Print results
    print("\n" + "=" * 70)
    print("  RESULTS  (Middlebury Other-Color-AllFrames)")
    print("=" * 70)
    print(f"\n  Evaluated on {n} triplets  |  "
          f"{len(seq_counts)} sequences  |  {len(METHODS)} methods\n")
    print(f"  {'Method':<42}  {'PSNR (dB)':>14}  {'SSIM':>12}  {'MSE':>10}")
    print("  " + "-" * 82)
    for method in METHODS:
        if method not in summary.index:
            continue
        meta   = METHODS_META[method]
        psnr_m = summary.loc[method, ("PSNR", "mean")]
        psnr_s = summary.loc[method, ("PSNR", "std")]
        ssim_m = summary.loc[method, ("SSIM", "mean")]
        ssim_s = summary.loc[method, ("SSIM", "std")]
        mse_m  = summary.loc[method, ("MSE",  "mean")]
        print(f"  {meta['name']:<42}  "
              f"{psnr_m:6.2f}±{psnr_s:5.2f}  "
              f"{ssim_m:.4f}±{ssim_s:.4f}  "
              f"{mse_m:8.2f}")
    print()

    # Per-sequence PSNR table
    print("  Per-Sequence Mean PSNR (dB):")
    seq_psnr = df.groupby(["sequence", "method"])["PSNR"].mean().unstack("method")
    header = f"  {'Sequence':<20}" + "".join(
        f"  {METHODS_META[m]['name'].split()[0]:>8}" for m in METHODS
        if m in seq_psnr.columns
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for seq in sorted(seq_psnr.index):
        row_str = f"  {seq:<20}"
        for m in METHODS:
            if m not in seq_psnr.columns:
                continue
            val = seq_psnr.loc[seq, m]
            row_str += f"  {val:8.2f}"
        print(row_str)
    print()

    # Step 6: Figures
    print("[5] Generating figures...")
    plot_average_metrics_bar(df,         figures_dir=figures_dir)
    plot_box_plots(df,                   figures_dir=figures_dir)
    plot_psnr_histogram(df,              figures_dir=figures_dir)
    plot_summary_table(summary,          figures_dir=figures_dir)
    plot_ssim_psnr_scatter(df,           figures_dir=figures_dir)
    plot_psnr_cdf(df,                    figures_dir=figures_dir)
    plot_improvement_over_linear(df,     figures_dir=figures_dir)
    plot_per_sequence_bar(df,            figures_dir=figures_dir)
    plot_per_sequence_ssim(df,           figures_dir=figures_dir)
    print(f"    Figures saved to {figures_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
