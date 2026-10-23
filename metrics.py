"""
Quantitative evaluation metrics for interpolated frames.

MSE and PSNR (dB) are computed over all pixels and channels; SSIM uses
scikit-image on the RGB image. Inputs are uint8 RGB frames.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity

MAX_PIXEL_VALUE = 255.0
TABLE_FLOAT_PRECISION = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


# Per-image metrics

def compute_mse(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Mean Squared Error averaged over all pixels and channels.

    MSE = (1/N) Σ (I_p − Î_p)²
    """
    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Shape mismatch: true={img_true.shape}, pred={img_pred.shape}."
        )
    diff = img_true.astype(np.float64) - img_pred.astype(np.float64)
    return float(np.mean(diff * diff))


def compute_psnr(
    img_true: np.ndarray,
    img_pred: np.ndarray,
    max_val: float = MAX_PIXEL_VALUE,
) -> float:
    """Peak Signal-to-Noise Ratio in decibels.

    PSNR = 10 log₁₀(MAX² / MSE)

    Returns +inf when MSE = 0 (perfect reconstruction).
    """
    mse = compute_mse(img_true, img_pred)
    if mse == 0.0:
        return float("inf")
    return float(10.0 * np.log10((max_val ** 2) / mse))


def compute_ssim(img_true: np.ndarray, img_pred: np.ndarray) -> float:
    """Structural Similarity Index computed over RGB channels.

    Uses the scikit-image implementation with data_range = 255 and
    channel_axis = 2 for multi-channel images.
    """
    if img_true.shape != img_pred.shape:
        raise ValueError(
            f"Shape mismatch: true={img_true.shape}, pred={img_pred.shape}."
        )
    return float(
        structural_similarity(
            img_true, img_pred,
            channel_axis=2,
            data_range=MAX_PIXEL_VALUE,
        )
    )


# Aggregation

def summarise_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Grouped mean and std for each interpolation method.

    Returns a DataFrame indexed by method with MultiIndex columns:
        (MSE, mean), (MSE, std), (PSNR, mean), …
    """
    required = {"method", "MSE", "PSNR", "SSIM"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    summary = df.groupby("method")[["MSE", "PSNR", "SSIM"]].agg(["mean", "std"])

    with pd.option_context("display.float_format", f"{{:.{TABLE_FLOAT_PRECISION}f}}".format):
        LOGGER.info("\n%s", summary.to_string())

    return summary
