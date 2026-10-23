"""
Vimeo-90K evaluation.

Runs every method in interpolation.METHODS on each triplet (optical flow is
computed once per triplet and shared), scores it with MSE / PSNR / SSIM,
writes the CSVs and draws the figures.

Usage:
  python main.py                              # full evaluation (3782 triplets)
  python main.py --max-triplets 100           # quick run on 100 triplets
  python main.py --max-triplets 50 --vis      # 50 triplets + visual comparisons
  python main.py --workers 8                  # use 8 parallel worker processes
  python main.py --output-dir outputs_vimeo90k --figures-dir figures_vimeo90k
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

from data_utils import load_triplet, load_triplet_list, save_frame
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
    plot_motion_stratified,
    plot_psnr_cdf,
    plot_psnr_histogram,
    plot_ssim_psnr_scatter,
    plot_summary_table,
    plot_uncertainty_map,
    plot_uncertainty_overlay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

NUM_VIS_SAMPLES = 10


# Worker function (must be at module level for pickling)

def _eval_triplet(args):
    """Process a single triplet. Returns list of metric dicts."""
    tid, dataset_root_str, do_vis, save_vis = args
    dataset_root = Path(dataset_root_str)

    try:
        I0, I1_gt, I2 = load_triplet(tid, dataset_root)
    except FileNotFoundError:
        return [], None, None, None

    flow = _compute_flow(I0, I2)
    flow_mag = float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))

    preds: Dict[str, np.ndarray] = {}
    rows = []
    uncertainty = None

    for method in METHODS:
        pred = interpolate(method, I0, I2, precomputed_flow=flow)
        preds[method] = pred
        rows.append({
            "triplet_id":     tid,
            "method":         method,
            "flow_magnitude": flow_mag,
            "MSE":            compute_mse(I1_gt, pred),
            "PSNR":           compute_psnr(I1_gt, pred),
            "SSIM":           compute_ssim(I1_gt, pred),
        })

    if do_vis:
        _, uncertainty = gpfi_interpolate(I0, I2, precomputed_flow=flow)

    return rows, preds, I1_gt, uncertainty


def parse_args():
    parser = argparse.ArgumentParser(description="Spline Frame Interpolation Evaluation")
    parser.add_argument("--max-triplets", type=int, default=None,
                        help="Limit triplets to evaluate (default: all 3782)")
    parser.add_argument("--vis", action="store_true",
                        help="Save visual comparison figures for sample triplets")
    parser.add_argument("--dataset-root", type=str,
                        default="vimeo_interp_test/vimeo_interp_test",
                        help="Path to vimeo_interp_test root")
    parser.add_argument("--workers", type=int,
                        default=min(mp.cpu_count(), 8),
                        help="Parallel worker processes (default: min(cpu_count, 8))")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for CSV results (default: outputs)")
    parser.add_argument("--figures-dir", type=str, default="figures",
                        help="Directory for figures (default: figures)")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    output_dir   = Path(args.output_dir)
    figures_dir  = Path(args.figures_dir)
    workers = 1 if (_CUDA and args.workers > 1) else args.workers

    print("=" * 70)
    print("  Spline-Based Video Frame Interpolation: Vimeo-90K Evaluation")
    print("=" * 70)
    if _CUDA:
        try:
            import torch
            print(f"  GPU: {torch.cuda.get_device_name(0)}  (CUDA warp + FFT enabled)")
        except Exception:
            pass
    print(f"  Workers:     {workers}")
    print(f"  Output dir:  {output_dir}")
    print(f"  Figures dir: {figures_dir}")

    # Step 1: Load triplet list
    print("\n[1] Loading dataset...")
    triplet_ids = load_triplet_list(dataset_root, max_triplets=args.max_triplets)
    n = len(triplet_ids)
    print(f"    {n} triplets to evaluate")

    vis_set = set(triplet_ids[:NUM_VIS_SAMPLES]) if args.vis else set()

    # Step 2: Evaluate all methods
    print(f"\n[2] Running interpolation methods ({workers} worker(s))...")
    t_start = time.perf_counter()

    task_args = [
        (tid, str(dataset_root), tid in vis_set, args.vis)
        for tid in triplet_ids
    ]

    all_rows: List[dict] = []
    vis_results: Dict[str, tuple] = {}

    if workers == 1:
        iterable = tqdm(task_args, total=n, unit="triplet") if _TQDM else task_args
        for i, task in enumerate(iterable):
            if not _TQDM and ((i + 1) % 100 == 0 or i == 0):
                print(f"    {i+1}/{n}  ({task[0]})")
            rows, preds, I1_gt, uncertainty = _eval_triplet(task)
            all_rows.extend(rows)
            if preds is not None and task[0] in vis_set:
                vis_results[task[0]] = (preds, I1_gt, uncertainty)
    else:
        completed = 0
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context("spawn")) as pool:
            futures = {pool.submit(_eval_triplet, t): t[0] for t in task_args}
            iter_futures = (
                tqdm(as_completed(futures), total=n, unit="triplet")
                if _TQDM else as_completed(futures)
            )
            for fut in iter_futures:
                tid = futures[fut]
                rows, preds, I1_gt, uncertainty = fut.result()
                all_rows.extend(rows)
                completed += 1
                if not _TQDM and completed % 100 == 0:
                    print(f"    {completed}/{n}")
                if preds is not None and tid in vis_set:
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
            for method, pred in preds.items():
                safe = tid.replace("/", "_")
                save_frame(pred, output_dir / method / f"{safe}.png")

    # Step 4: Summarise
    print("\n[4] Computing summary statistics...")
    df = pd.DataFrame(all_rows)
    df.rename(columns={"triplet_id": "frame_index"}, inplace=True)

    if "flow_magnitude" in df.columns:
        _mag = df[df["method"] == "linear"]["flow_magnitude"]
        p33, p67 = float(_mag.quantile(0.33)), float(_mag.quantile(0.67))
        print(f"    Motion bins: slow <= {p33:.2f} | moderate <= {p67:.2f} | "
              f"fast > {p67:.2f} px/frame")
        df["motion_bin"] = pd.cut(
            df["flow_magnitude"],
            bins=[-np.inf, p33, p67, np.inf],
            labels=["Slow", "Moderate", "Fast"],
        )

    summary = summarise_metrics(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path     = output_dir / "results.csv"
    summary_csv  = output_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_csv)
    print(f"    Raw results  -> {csv_path}  (saved)")
    print(f"    Summary      -> {summary_csv}  (saved)")

    # Step 5: Print results table
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(f"\n  Evaluated on {n} triplets\n")

    for method in METHODS:
        if method not in summary.index:
            continue
        meta   = METHODS_META[method]
        psnr_m = summary.loc[method, ("PSNR", "mean")]
        psnr_s = summary.loc[method, ("PSNR", "std")]
        ssim_m = summary.loc[method, ("SSIM", "mean")]
        ssim_s = summary.loc[method, ("SSIM", "std")]
        mse_m  = summary.loc[method, ("MSE",  "mean")]
        print(f"  {meta['name']:42s}  PSNR: {psnr_m:.2f}±{psnr_s:.2f} dB  "
              f"SSIM: {ssim_m:.4f}±{ssim_s:.4f}  MSE: {mse_m:.2f}")
    print()

    if "motion_bin" in df.columns:
        print("  Motion-Stratified Mean PSNR (dB):")
        print(f"  {'Method':<42}  {'Slow':>8}  {'Moderate':>10}  {'Fast':>8}")
        print("  " + "-" * 74)
        for method in METHODS:
            if method not in summary.index:
                continue
            sub = df[df["method"] == method]
            vals = []
            for bn in ["Slow", "Moderate", "Fast"]:
                bsub = sub[sub["motion_bin"] == bn]["PSNR"]
                vals.append(f"{bsub.mean():.2f}" if not bsub.empty else "  N/A")
            print(f"  {METHODS_META[method]['name']:<42}  "
                  f"{vals[0]:>8}  {vals[1]:>10}  {vals[2]:>8}")
        print()

    # Step 6: Generate all figures
    print("[5] Generating figures...")
    plot_average_metrics_bar(df,      figures_dir=figures_dir)
    plot_box_plots(df,                figures_dir=figures_dir)
    plot_psnr_histogram(df,           figures_dir=figures_dir)
    plot_summary_table(summary,       figures_dir=figures_dir)
    plot_motion_stratified(df,        figures_dir=figures_dir)
    plot_ssim_psnr_scatter(df,        figures_dir=figures_dir)
    plot_psnr_cdf(df,                 figures_dir=figures_dir)
    plot_improvement_over_linear(df,  figures_dir=figures_dir)
    print(f"    Figures saved to {figures_dir}/")
    print("\nDone!")


if __name__ == "__main__":
    main()
