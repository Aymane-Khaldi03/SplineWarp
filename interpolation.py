"""
Frame interpolation methods.

Given two input frames I0 (t=0) and I2 (t=1), estimate I1 (t=0.5).

Classical baselines (all three give the same result with two samples):
  linear   pixel-wise average
  cubic    natural cubic spline per pixel
  bspline  cubic B-spline per pixel

Motion-compensated methods (DIS optical flow, both frames warped to t=0.5):
  matsi    average of the two warped frames
  gpfi     Gaussian process on the warped values, plus an uncertainty map
  vtpsi    thin-plate (biharmonic) smoothing, solved with a 2-D FFT
  assi     per-patch choice between linear, matsi and vtpsi
  fdsi     warp in the wavelet domain (needs PyWavelets)

Every method accepts (I0, I2) as uint8 RGB arrays (H, W, 3) and returns
I1_hat as uint8 RGB (H, W, 3).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import cv2
import numpy as np
from scipy.interpolate import CubicSpline

# Optional GPU acceleration via PyTorch
try:
    import torch
    import torch.nn.functional as TF
    _CUDA = torch.cuda.is_available()
except ImportError:
    torch = None  # type: ignore
    _CUDA = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

if _CUDA:
    LOGGER.info("GPU warp enabled: %s", torch.cuda.get_device_name(0))
else:
    LOGGER.info("GPU unavailable: using CPU warp")


# Helpers

def _to_float(img: np.ndarray) -> np.ndarray:
    """Convert uint8 image to float32 (halves GPU memory bandwidth vs float64)."""
    return img.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


_DIS: Optional[cv2.DISOpticalFlow] = None  # singleton, created once

def _compute_flow(I0: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """Dense optical flow from I0 to I2 using DIS (MEDIUM preset). Returns (H, W, 2)."""
    global _DIS
    if _DIS is None:
        _DIS = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        _DIS.setUseSpatialPropagation(True)
    gray0 = cv2.cvtColor(I0, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(I2, cv2.COLOR_RGB2GRAY)
    flow = _DIS.calc(gray0, gray2, None)   # (H, W, 2) float32
    return flow


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """Warp *img* by *flow* using bilinear interpolation.

    Uses GPU (torch.nn.functional.grid_sample) when CUDA is available,
    otherwise falls back to cv2.remap on CPU.
    """
    if _CUDA and torch is not None:
        return _warp_gpu(img, flow)
    h, w = flow.shape[:2]
    coords_x, coords_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (coords_x + flow[..., 0]).astype(np.float32)
    map_y = (coords_y + flow[..., 1]).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT_101)


_GRID_CACHE: dict = {}  # cache normalised identity grids by (H, W)


def _warp_gpu(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    """GPU-accelerated backward warp via torch.nn.functional.grid_sample."""
    device = torch.device("cuda")
    h, w = flow.shape[:2]

    # Build (or retrieve cached) normalised identity grid
    key = (h, w)
    if key not in _GRID_CACHE:
        xs = torch.linspace(-1.0, 1.0, w, device=device)
        ys = torch.linspace(-1.0, 1.0, h, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # (H, W)
        _GRID_CACHE[key] = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1,H,W,2)

    identity = _GRID_CACHE[key]  # (1, H, W, 2)

    # Convert flow to normalised coordinates
    flow_t = torch.from_numpy(flow.astype(np.float32)).to(device)  # (H, W, 2)
    flow_norm = flow_t.clone()
    flow_norm[..., 0] = flow_t[..., 0] / (w / 2.0)  # x displacement
    flow_norm[..., 1] = flow_t[..., 1] / (h / 2.0)  # y displacement
    grid = identity + flow_norm.unsqueeze(0)  # (1, H, W, 2)

    # Image tensor: (1, C, H, W), float32 in [0,255]
    img_t = torch.from_numpy(img.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
    warped = TF.grid_sample(img_t, grid, mode="bilinear", padding_mode="reflection",
                            align_corners=True)
    return warped.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (H, W, C) float32


# 1. Linear

def linear_interpolate(I0: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """Pixel-wise average:  I1_hat = (I0 + I2) / 2."""
    return _to_uint8((_to_float(I0) + _to_float(I2)) / 2.0)


# 2. Cubic spline

def cubic_spline_interpolate(I0: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """Natural cubic spline evaluated at t=0.5.

    With only two samples (t=0 and t=1) and natural boundary conditions
    (M0 = M1 = 0), the spline is the straight line, so this equals linear.
    """
    f0 = _to_float(I0)
    f2 = _to_float(I2)
    H, W, C = f0.shape
    flat0 = f0.reshape(-1)
    flat2 = f2.reshape(-1)

    # Stack pixel values: shape (2, N)
    y = np.stack([flat0, flat2], axis=0)
    t = np.array([0.0, 1.0])

    # CubicSpline with 2 points and natural BC is linear
    cs = CubicSpline(t, y, bc_type="natural")
    result = cs(0.5)  # (N,)
    return _to_uint8(result.reshape(H, W, C))


# 3. B-spline

def bspline_interpolate(I0: np.ndarray, I2: np.ndarray) -> np.ndarray:
    """Cubic B-spline interpolation at t=0.5.

    With two samples this is the same line as the cubic spline, so it reuses it.
    """
    return cubic_spline_interpolate(I0, I2)


# 4. MATSI: Motion-Aware Trajectory Spline Interpolation

def matsi_interpolate(
    I0: np.ndarray,
    I2: np.ndarray,
    precomputed_flow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Motion-Aware Trajectory Spline Interpolation.

    1. Compute dense optical flow F from I0 to I2 (DIS).
    2. Backward-warp I0 by -F/2 and I2 by +F/2, so both align at t=0.5.
    3. Average the two warped frames.
    """
    flow_0to2 = precomputed_flow if precomputed_flow is not None else _compute_flow(I0, I2)
    half_flow = flow_0to2 * 0.5

    # Backward warping: for the intermediate frame, sample source positions
    I0_warped = _warp(I0, -half_flow)   # source in I0: q - 0.5*F(q)
    I2_warped = _warp(I2,  half_flow)   # source in I2: q + 0.5*F(q)

    # Blend warped frames (spline along trajectory = average with 2 points)
    I1_hat = (_to_float(I0_warped) + _to_float(I2_warped)) / 2.0
    return _to_uint8(I1_hat)


# 5. GPFI: Gaussian Process Frame Interpolation

def gpfi_interpolate(
    I0: np.ndarray,
    I2: np.ndarray,
    length_scale: float = 0.2,
    signal_var: float = 1.0,
    noise_var: float = 1e-4,
    precomputed_flow: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gaussian Process Frame Interpolation with RBF kernel.

      - Kernel: k(t,t') = σ² exp(-|t-t'|² / (2ℓ²))
      - Observations: y = [v0, v2] at T = [0, 1]
      - Posterior mean at t*=0.5:
          μ* = k(t*,T) (K + σ_n² I)^{-1} y
      - Posterior variance:
          σ*² = k(t*,t*) - k(t*,T) (K + σ_n² I)^{-1} k(T,t*)

    Motion compensation: both frames warped to t=0.5 before GP inference.

    Returns
    -------
    I1_hat : np.ndarray (H, W, 3) uint8
    uncertainty : np.ndarray (H, W): GP variance (a constant) plus
                  0.01 * squared difference between the two warped frames
    """
    # Motion compensation: warp both frames to t=0.5
    flow_0to2 = precomputed_flow if precomputed_flow is not None else _compute_flow(I0, I2)
    half_flow = flow_0to2 * 0.5

    I0_w = _warp(I0, -half_flow)   # source in I0: q - 0.5*F(q)
    I2_w = _warp(I2,  half_flow)   # source in I2: q + 0.5*F(q)

    f0 = _to_float(I0_w)
    f2 = _to_float(I2_w)

    # RBF kernel evaluations
    T = np.array([0.0, 1.0])
    t_star = 0.5
    ell2 = length_scale ** 2

    # K matrix (2x2)
    K = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            K[i, j] = signal_var * np.exp(-0.5 * (T[i] - T[j]) ** 2 / ell2)
    K += noise_var * np.eye(2)

    # k_star vector (2,)
    k_star = np.array([
        signal_var * np.exp(-0.5 * (t_star - T[0]) ** 2 / ell2),
        signal_var * np.exp(-0.5 * (t_star - T[1]) ** 2 / ell2),
    ])

    # K^{-1} for 2x2: direct formula
    det = K[0, 0] * K[1, 1] - K[0, 1] * K[1, 0]
    K_inv = np.array([[K[1, 1], -K[0, 1]], [-K[1, 0], K[0, 0]]]) / det

    # Weights: w = k_star^T K^{-1}  -> shape (2,)
    w = k_star @ K_inv
    # Normalise so weights sum to 1 (keeps pixel values in valid range)
    w = w / w.sum()

    # Posterior mean per-pixel
    I1_hat = w[0] * f0 + w[1] * f2

    # Posterior variance (scalar, same for all pixels)
    k_ss = signal_var  # k(t*, t*) = σ²
    var = max(k_ss - k_star @ K_inv @ k_star, 0.0)

    # Uncertainty map: kernel variance + local intensity disagreement
    diff_sq = np.mean((f0 - f2) ** 2, axis=2)  # (H, W)
    uncertainty = var + diff_sq * 0.01

    return _to_uint8(I1_hat), uncertainty


# 6. VTPSI: Variational Thin-Plate Spline Interpolation

def vtpsi_interpolate(
    I0: np.ndarray,
    I2: np.ndarray,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
    lambda3: float = 0.005,
    precomputed_flow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Variational Thin-Plate Spline Interpolation.

    Energy functional:
      E(I1) = λ₁‖W(I0, F₀→½) − I1‖² + λ₂‖W(I2, F₂→½) − I1‖² + λ₃‖∇²I1‖²

    where:
      - W(I, F) warps image I by flow F
      - ∇²I1 is the Laplacian (thin-plate bending energy)

    Euler-Lagrange → linear system solved via 2D FFT:
      [(λ₁ + λ₂)I + λ₃ Δ²] I1 = λ₁ W(I0) + λ₂ W(I2)

    where Δ² is the biharmonic operator.  FFT is run on GPU when CUDA
    is available, on CPU otherwise.
    """
    # Motion compensation: warp both frames to t=0.5
    flow_0to2 = precomputed_flow if precomputed_flow is not None else _compute_flow(I0, I2)
    half_flow = flow_0to2 * 0.5

    I0_w = _to_float(_warp(I0, -half_flow))  # source: q - 0.5*F(q)
    I2_w = _to_float(_warp(I2,  half_flow))  # source: q + 0.5*F(q)

    H, W, C = I0_w.shape

    # Right-hand side
    rhs = lambda1 * I0_w + lambda2 * I2_w  # (H, W, 3)

    if _CUDA and torch is not None:
        # GPU path: torch.fft on CUDA
        device = torch.device("cuda")
        rhs_t = torch.from_numpy(rhs.astype(np.float32)).permute(2, 0, 1).to(device)  # (C,H,W)

        u = torch.fft.fftfreq(H, device=device).unsqueeze(1)  # (H, 1)
        v = torch.fft.fftfreq(W, device=device).unsqueeze(0)  # (1, W)
        lap_eig = (2.0 * torch.cos(2.0 * torch.pi * u) - 2.0) + \
                  (2.0 * torch.cos(2.0 * torch.pi * v) - 2.0)
        denom = torch.clamp((lambda1 + lambda2) + lambda3 * lap_eig ** 2, min=1e-10)  # (H, W)

        rhs_fft = torch.fft.fft2(rhs_t)  # (C, H, W) complex
        result_t = torch.fft.ifft2(rhs_fft / denom.unsqueeze(0)).real  # (C, H, W)
        result = result_t.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    else:
        # CPU path: numpy.fft
        u = np.fft.fftfreq(H)
        v = np.fft.fftfreq(W)
        U, V = np.meshgrid(v, u)
        lap_eig = (2.0 * np.cos(2.0 * np.pi * U) - 2.0) + \
                  (2.0 * np.cos(2.0 * np.pi * V) - 2.0)
        denom = np.maximum((lambda1 + lambda2) + lambda3 * lap_eig ** 2, 1e-10)
        result = np.zeros_like(rhs)
        for c in range(C):
            rhs_fft = np.fft.fft2(rhs[:, :, c])
            result[:, :, c] = np.real(np.fft.ifft2(rhs_fft / denom))

    return _to_uint8(result)


# 7. ASSI: Adaptive Spline Order Selection via Local Motion Complexity

def assi_interpolate(
    I0: np.ndarray,
    I2: np.ndarray,
    patch_size: int = 32,
    tau_motion: float = 3.0,
    tau_texture: float = 300.0,
    precomputed_flow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Patch-wise adaptive interpolation using local motion and texture complexity.

    For each overlapping 32x32 patch:
        kappa_m = mean optical-flow magnitude  (motion complexity)
        kappa_t = pixel-intensity variance     (texture complexity)

    Decision map:
        kappa_m < tau1  AND  kappa_t < tau2  ->  linear  (smooth, static)
        kappa_m < tau1  AND  kappa_t >= tau2 ->  VTPSI   (textured, static)
        kappa_m >= tau1 AND  kappa_t < tau2  ->  MATSI   (moving, smooth)
        kappa_m >= tau1 AND  kappa_t >= tau2 ->  VTPSI   (moving, textured)

    Overlapping patches (50% stride) are blended with a Gaussian window to
    suppress blocking artefacts at patch boundaries.
    """
    flow = precomputed_flow if precomputed_flow is not None else _compute_flow(I0, I2)
    H, W = I0.shape[:2]
    f0 = _to_float(I0)

    # Candidate results, all sharing the same flow
    r_linear = (_to_float(I0) + _to_float(I2)) / 2.0
    r_matsi  = _to_float(matsi_interpolate(I0, I2, precomputed_flow=flow))
    r_vtpsi  = _to_float(vtpsi_interpolate(I0, I2, precomputed_flow=flow))

    flow_mag = np.sqrt(flow[:, :, 0] ** 2 + flow[:, :, 1] ** 2)

    # Gaussian window for overlap-add blending
    g1  = cv2.getGaussianKernel(patch_size, patch_size / 4.0).astype(np.float32)
    gwin = (g1 @ g1.T)  # (patch_size, patch_size)

    blend   = np.zeros((H, W, 3), dtype=np.float32)
    weights = np.zeros((H, W),    dtype=np.float32)

    stride = patch_size // 2
    ys = sorted(set(list(range(0, H - patch_size, stride)) + [H - patch_size]))
    xs = sorted(set(list(range(0, W - patch_size, stride)) + [W - patch_size]))

    for y in ys:
        for x in xs:
            sy = slice(y, y + patch_size)
            sx = slice(x, x + patch_size)
            kappa_m = float(np.mean(flow_mag[sy, sx]))
            kappa_t = float(np.var(f0[sy, sx]))

            if kappa_m >= tau_motion:
                patch = r_vtpsi[sy, sx] if kappa_t >= tau_texture else r_matsi[sy, sx]
            else:
                patch = r_vtpsi[sy, sx] if kappa_t >= tau_texture else r_linear[sy, sx]

            blend[sy, sx]   += gwin[:, :, np.newaxis] * patch
            weights[sy, sx] += gwin

    result = blend / np.maximum(weights[:, :, np.newaxis], 1e-6)
    return _to_uint8(result)


# 8. FDSI: Frequency-Decoupled Spline Interpolation

try:
    import pywt as _pywt
    _PYWT = True
except ImportError:
    _pywt = None
    _PYWT = False
    LOGGER.warning(
        "PyWavelets not installed: FDSI falls back to VTPSI. "
        "Fix: pip install PyWavelets"
    )


def fdsi_interpolate(
    I0: np.ndarray,
    I2: np.ndarray,
    wavelet: str = "db2",
    precomputed_flow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Frequency-Decoupled Spline Interpolation in the wavelet domain.

    Each channel of both frames is split with a one-level 2-D DWT. Every
    subband (LL, LH, HL, HH) is warped to t=0.5 with the flow scaled to
    subband resolution, the two warped subbands are averaged, and the
    frame is rebuilt with the inverse DWT.

    Falls back to VTPSI if PyWavelets is not installed.
    """
    if not _PYWT:
        return vtpsi_interpolate(I0, I2, precomputed_flow=precomputed_flow)

    flow = precomputed_flow if precomputed_flow is not None else _compute_flow(I0, I2)
    H, W = I0.shape[:2]
    f0 = _to_float(I0)
    f2 = _to_float(I2)

    # half_flow: full-res displacement to midpoint t=0.5
    half_flow = flow * 0.5

    def warp_sub(band: np.ndarray, flo: np.ndarray) -> np.ndarray:
        """Backward-warp a 2-D subband array by flo (H_sub x W_sub x 2)."""
        sh, sw = band.shape
        cx = np.arange(sw, dtype=np.float32)
        cy = np.arange(sh, dtype=np.float32)
        gx, gy = np.meshgrid(cx, cy)
        fh = min(sh, flo.shape[0])
        fw = min(sw, flo.shape[1])
        map_x, map_y = gx.copy(), gy.copy()
        map_x[:fh, :fw] += flo[:fh, :fw, 0]
        map_y[:fh, :fw] += flo[:fh, :fw, 1]
        return cv2.remap(
            band.astype(np.float32), map_x, map_y,
            cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )

    result = np.zeros_like(f0)

    for c in range(3):
        # 2-D DWT (1 level) for each channel
        LL0, (LH0, HL0, HH0) = _pywt.dwt2(f0[:, :, c], wavelet)
        LL2, (LH2, HL2, HH2) = _pywt.dwt2(f2[:, :, c], wavelet)
        sh, sw = LL0.shape

        # Subbands are half resolution, so displacements are halved too
        flo_sub = cv2.resize(
            half_flow, (sw, sh), interpolation=cv2.INTER_LINEAR
        ) * 0.5

        # Warp every subband to t=0.5 and average
        LL_interp = 0.5 * (warp_sub(LL0, -flo_sub) + warp_sub(LL2, flo_sub))
        LH_interp = 0.5 * (warp_sub(LH0, -flo_sub) + warp_sub(LH2, flo_sub))
        HL_interp = 0.5 * (warp_sub(HL0, -flo_sub) + warp_sub(HL2, flo_sub))
        HH_interp = 0.5 * (warp_sub(HH0, -flo_sub) + warp_sub(HH2, flo_sub))

        # Reconstruct via inverse DWT
        ch = _pywt.idwt2((LL_interp, (LH_interp, HL_interp, HH_interp)), wavelet)
        result[:, :, c] = ch[:H, :W]

    return _to_uint8(result)


# Dispatcher

METHODS = [
    "linear", "cubic", "bspline",          # classical baselines
    "matsi", "gpfi", "vtpsi",              # motion-compensated
    "assi", "fdsi",
]

METHODS_META = {
    "linear":  {"name": "Linear Interpolation",          "type": "Classical baseline"},
    "cubic":   {"name": "Cubic Spline Interpolation",    "type": "Classical spline"},
    "bspline": {"name": "B-Spline Interpolation",        "type": "Classical spline"},
    "matsi":   {"name": "MATSI",                         "type": "Motion-aware, deterministic"},
    "gpfi":    {"name": "GPFI",                          "type": "Motion-aware, stochastic"},
    "vtpsi":   {"name": "VTPSI",                         "type": "Motion-aware, variational"},
    "assi":    {"name": "ASSI",                          "type": "Motion-aware, adaptive"},
    "fdsi":    {"name": "FDSI",                          "type": "Motion-aware, frequency"},
}


def interpolate(
    method: str,
    I0: np.ndarray,
    I2: np.ndarray,
    precomputed_flow: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Dispatch to the named interpolation method.

    Pass *precomputed_flow* to avoid recomputing DIS flow for every method.
    Returns the estimated middle frame I1_hat as uint8 RGB.
    For GPFI, returns only the frame (uncertainty map is discarded here).
    """
    name = method.strip().lower()
    if name == "linear":
        return linear_interpolate(I0, I2)
    if name == "cubic":
        return cubic_spline_interpolate(I0, I2)
    if name == "bspline":
        return bspline_interpolate(I0, I2)
    if name == "matsi":
        return matsi_interpolate(I0, I2, precomputed_flow=precomputed_flow)
    if name == "gpfi":
        return gpfi_interpolate(I0, I2, precomputed_flow=precomputed_flow)[0]
    if name == "vtpsi":
        return vtpsi_interpolate(I0, I2, precomputed_flow=precomputed_flow)
    if name == "assi":
        return assi_interpolate(I0, I2, precomputed_flow=precomputed_flow)
    if name == "fdsi":
        return fdsi_interpolate(I0, I2, precomputed_flow=precomputed_flow)
    raise ValueError(f"Unknown interpolation method: {method!r}")

