"""
Data loading utilities for the Vimeo-90K triplet dataset.

The Vimeo-90K test set contains 3,782 triplets at 448x256 resolution.
Each triplet has:
  - input/XXXXX/YYYY/im1.png  (frame at t=0)
  - input/XXXXX/YYYY/im3.png  (frame at t=1)
  - target/XXXXX/YYYY/im2.png (ground-truth at t=0.5)

All frames are stored as RGB uint8 numpy arrays of shape (H, W, 3).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

DATASET_ROOT = Path("vimeo_interp_test") / "vimeo_interp_test"
TRIPLET_LIST = DATASET_ROOT / "tri_testlist.txt"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)


def load_triplet_list(
    dataset_root: Path = DATASET_ROOT,
    list_file: Path = TRIPLET_LIST,
    max_triplets: Optional[int] = None,
) -> List[str]:
    """Load the list of triplet IDs from tri_testlist.txt.

    Returns list of strings like '00001/0389'.
    """
    lf = list_file if list_file.is_absolute() else dataset_root.parent / list_file.name
    if not lf.exists():
        lf = dataset_root / "tri_testlist.txt"
    triplets = [
        line.strip() for line in lf.read_text().splitlines() if line.strip()
    ]
    if max_triplets is not None:
        triplets = triplets[:max_triplets]
    LOGGER.info("Loaded %d triplet IDs from %s", len(triplets), lf)
    return triplets


def load_triplet(
    triplet_id: str,
    dataset_root: Path = DATASET_ROOT,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a single triplet (I0, I1_gt, I2) as RGB uint8 arrays.

    Parameters
    ----------
    triplet_id : str
        e.g. '00001/0389'

    Returns
    -------
    I0, I1_gt, I2 : np.ndarray of shape (H, W, 3), dtype uint8
    """
    input_dir = dataset_root / "input" / triplet_id
    target_dir = dataset_root / "target" / triplet_id

    im1_path = input_dir / "im1.png"
    im3_path = input_dir / "im3.png"
    im2_path = target_dir / "im2.png"

    for p in (im1_path, im3_path, im2_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing file: {p}")

    I0 = cv2.cvtColor(cv2.imread(str(im1_path)), cv2.COLOR_BGR2RGB)
    I2 = cv2.cvtColor(cv2.imread(str(im3_path)), cv2.COLOR_BGR2RGB)
    I1_gt = cv2.cvtColor(cv2.imread(str(im2_path)), cv2.COLOR_BGR2RGB)

    return I0.astype(np.uint8), I1_gt.astype(np.uint8), I2.astype(np.uint8)


def save_frame(frame: np.ndarray, path: Path) -> None:
    """Save an RGB uint8 frame as a PNG file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
