# Results

All numbers come from `outputs/results.csv` (Vimeo-90K) and `outputs_middlebury/results.csv` (Middlebury).

## Setup

| | Vimeo-90K | Middlebury Other |
|---|---|---|
| Data | Triplet test set | Other-Color-AllFrames |
| Resolution | 448 × 256 | 640 × 480 or 584 × 388 |
| Triplets | 3,782 | 60 (10 sequences × 6) |
| Task | predict im2 from im1 and im3 | predict frame N+1 from frames N and N+2 |
| Optical flow | OpenCV DIS, MEDIUM preset (CPU) | same |
| GPU | RTX 5070 Ti (warping and FFT) | same |
| Run time | 1,267.7 s, 335 ms per triplet | 96.6 s, 1,611 ms per triplet |

Dimetrodon and Venus have only two frames, so they are skipped on Middlebury.

## Vimeo-90K

| Method | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| Linear | 288.72 ± 301.25 | 25.61 ± 4.60 | 0.7526 ± 0.1707 |
| Cubic spline | 288.72 ± 301.25 | 25.61 ± 4.60 | 0.7526 ± 0.1707 |
| B-spline | 288.72 ± 301.25 | 25.61 ± 4.60 | 0.7526 ± 0.1707 |
| MATSI | 51.68 ± 53.11 | 33.05 ± 4.50 | 0.9489 ± 0.0302 |
| GPFI | 51.68 ± 53.11 | 33.05 ± 4.50 | 0.9489 ± 0.0302 |
| VTPSI | 51.88 ± 53.16 | 33.03 ± 4.49 | 0.9486 ± 0.0303 |
| ASSI | 51.99 ± 53.14 | 33.00 ± 4.47 | 0.9479 ± 0.0306 |
| FDSI | 70.99 ± 67.04 | 31.47 ± 4.33 | 0.9235 ± 0.0486 |

MATSI is 7.44 dB PSNR and 0.196 SSIM above linear.

### By motion speed

Triplets are grouped by mean optical-flow magnitude, split at the 33rd and 67th percentiles: slow ≤ 3.84, moderate ≤ 6.40, fast > 6.40 px.

| Method | Slow | Moderate | Fast | Max − min |
|---|---|---|---|---|
| Linear / Cubic / B-spline | 28.24 | 25.63 | 22.94 | 5.30 |
| MATSI | 33.04 | 33.41 | 32.69 | 0.72 |
| GPFI | 33.04 | 33.41 | 32.69 | 0.72 |
| VTPSI | 33.02 | 33.38 | 32.67 | 0.71 |
| ASSI | 32.99 | 33.34 | 32.67 | 0.67 |
| FDSI | 31.97 | 31.38 | 31.05 | 0.92 |

## Middlebury Other

| Method | MSE ↓ | PSNR (dB) ↑ | SSIM ↑ |
|---|---|---|---|
| Linear | 344.84 ± 249.77 | 24.00 ± 3.67 | 0.6904 ± 0.1839 |
| Cubic spline | 344.84 ± 249.77 | 24.00 ± 3.67 | 0.6904 ± 0.1839 |
| B-spline | 344.84 ± 249.77 | 24.00 ± 3.67 | 0.6904 ± 0.1839 |
| MATSI | 88.92 ± 70.11 | 30.43 ± 4.54 | 0.9092 ± 0.0582 |
| GPFI | 88.92 ± 70.11 | 30.43 ± 4.54 | 0.9092 ± 0.0582 |
| VTPSI | 89.07 ± 70.02 | 30.41 ± 4.52 | 0.9088 ± 0.0578 |
| ASSI | 89.00 ± 69.96 | 30.41 ± 4.50 | 0.9088 ± 0.0576 |
| FDSI | 107.62 ± 78.38 | 29.07 ± 3.52 | 0.8680 ± 0.0749 |

MATSI is 6.43 dB PSNR and 0.219 SSIM above linear.

### PSNR per sequence (dB)

| Sequence | Linear | MATSI | GPFI | VTPSI | ASSI | FDSI |
|---|---|---|---|---|---|---|
| Beanbags | 24.60 | 24.96 | 24.96 | 24.96 | 24.96 | 25.13 |
| DogDance | 23.60 | 26.40 | 26.40 | 26.42 | 26.44 | 27.01 |
| Grove2 | 20.71 | 30.64 | 30.64 | 30.50 | 30.51 | 26.78 |
| Grove3 | 18.32 | 25.35 | 25.35 | 25.33 | 25.33 | 23.74 |
| Hydrangea | 23.18 | 32.12 | 32.12 | 32.10 | 32.10 | 30.80 |
| MiniCooper | 21.69 | 28.34 | 28.34 | 28.35 | 28.35 | 28.92 |
| RubberWhale | 32.36 | 40.42 | 40.42 | 40.33 | 40.27 | 35.22 |
| Urban2 | 24.05 | 31.50 | 31.50 | 31.50 | 31.50 | 30.18 |
| Urban3 | 24.17 | 32.57 | 32.57 | 32.55 | 32.55 | 29.39 |
| Walking | 27.27 | 32.04 | 32.04 | 32.06 | 32.06 | 33.52 |

The largest gains over linear are on Grove2 (+9.93 dB) and Hydrangea (+8.94 dB). FDSI has the best score on Beanbags, DogDance, MiniCooper and Walking.

## Vimeo-90K compared with Middlebury

| Method | Vimeo PSNR | Middlebury PSNR | Difference |
|---|---|---|---|
| Linear | 25.61 | 24.00 | −1.61 |
| MATSI / GPFI | 33.05 | 30.43 | −2.62 |
| VTPSI | 33.03 | 30.41 | −2.62 |
| ASSI | 33.00 | 30.41 | −2.59 |
| FDSI | 31.47 | 29.07 | −2.40 |

The ranking of the methods is the same on both datasets, and every method scores lower on Middlebury.

## Trained networks on Vimeo-90K

SplineWarp is trained on nothing. The other scores are as reported in each paper:

| Method | PSNR (dB) | SSIM |
|---|---|---|
| **SplineWarp (ours, MATSI)** | **33.05** | **0.949** |
| Super SloMo (CVPR 2018) | 33.15 | 0.948 |
| RIFE (ECCV 2022) | 35.61 | 0.978 |
| EMA-VFI (CVPR 2023) | 36.64 | 0.980 |
| AMT (CVPR 2023) | 36.67 | 0.981 |
| VFIMamba (NeurIPS 2024) | 36.92 | 0.982 |

MATSI (33.05 dB) is 2.56 dB below RIFE and 3.87 dB below VFIMamba.

## Notes

1. Linear, cubic spline and B-spline give identical outputs on every triplet. A natural cubic spline through two points is the straight line between them.
2. Warping along optical flow accounts for almost all of the gain. MATSI, GPFI, VTPSI and ASSI are within 0.05 dB of each other on Vimeo-90K.
3. GPFI gives almost the same frame as MATSI (at most 0.007 dB apart on any triplet) because the GP weights at t = 0.5 are ½ and ½. Its uncertainty map is the only extra output.
4. FDSI is 1.58 dB below MATSI on Vimeo-90K, probably because warping the half-resolution wavelet bands is less accurate than warping the full image.

## Figures

Vimeo-90K, in `figures/`: `average_metrics_bar.png`, `box_plots.png`, `psnr_histogram.png`, `summary_table.png`, `motion_stratified.png`, `ssim_psnr_scatter.png`, `psnr_cdf.png`, `improvement_over_linear.png`, and GPFI maps in `gpfi/`.

Middlebury, in `figures_middlebury/`: the same set without the motion plot, plus `per_sequence_psnr.png`, `per_sequence_ssim.png`, comparison grids in `comparisons/` and GPFI maps in `gpfi/`.
