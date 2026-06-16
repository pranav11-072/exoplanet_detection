"""
Preprocessing Module
Detrends, cleans, and normalises raw TESS light curves before transit detection.
"""

import numpy as np
from scipy.signal import savgol_filter, medfilt
from scipy.interpolate import UnivariateSpline
import logging

logger = logging.getLogger(__name__)


def sigma_clip(flux: np.ndarray, sigma: float = 4.0, max_iter: int = 5) -> np.ndarray:
    """Return a boolean mask of True = good (not clipped) data points."""
    mask = np.ones(len(flux), dtype=bool)
    for _ in range(max_iter):
        med = np.median(flux[mask])
        std = np.std(flux[mask])
        new_mask = np.abs(flux - med) < sigma * std
        if np.all(new_mask == mask):
            break
        mask = new_mask
    return mask


def remove_outliers(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
                    sigma: float = 4.5) -> tuple:
    """Remove outlier data points via sigma clipping."""
    mask = sigma_clip(flux, sigma=sigma)
    logger.info(f"Outlier removal: kept {mask.sum()}/{len(mask)} points")
    return time[mask], flux[mask], flux_err[mask]


def detrend_savgol(flux: np.ndarray, window_length: int = None,
                   polyorder: int = 3) -> tuple:
    """
    Detrend using a Savitzky-Golay filter (good for preserving transit shapes).

    Returns
    -------
    flux_detrended, trend
    """
    if window_length is None:
        # ~2-day window for typical TESS 2-min cadence (~720 points/day)
        window_length = min(len(flux) // 5, 501)
        if window_length % 2 == 0:
            window_length += 1
        window_length = max(window_length, polyorder + 2)

    trend = savgol_filter(flux, window_length=window_length, polyorder=polyorder)
    flux_detrended = flux / trend
    return flux_detrended, trend


def detrend_spline(time: np.ndarray, flux: np.ndarray, knot_spacing: float = 1.5) -> tuple:
    """
    Detrend using a cubic spline with knots every `knot_spacing` days.
    Better for long-period trends.

    Returns
    -------
    flux_detrended, trend
    """
    knots = np.arange(time[0] + knot_spacing, time[-1] - knot_spacing, knot_spacing)
    try:
        spline = UnivariateSpline(time, flux, t=knots, k=3)
        trend = spline(time)
        flux_detrended = flux / trend
    except Exception:
        logger.warning("Spline detrending failed, falling back to Savitzky-Golay")
        flux_detrended, trend = detrend_savgol(flux)
    return flux_detrended, trend


def median_filter_gaps(time: np.ndarray, flux: np.ndarray,
                       gap_threshold: float = 0.5) -> tuple:
    """
    Fill gaps in time series via linear interpolation so BLS works on a
    near-uniform grid. Returns original indices mask so results map back.
    """
    dt = np.median(np.diff(time))
    gaps = np.where(np.diff(time) > gap_threshold)[0]
    if len(gaps) == 0:
        return time, flux
    logger.info(f"Found {len(gaps)} time gap(s) in light curve")
    return time, flux  # BLS handles gaps natively; just return as-is


def normalise(flux: np.ndarray) -> np.ndarray:
    """Normalise flux to median = 1.0."""
    med = np.median(flux)
    if med == 0:
        return flux
    return flux / med


def preprocess(lc_dict: dict, method: str = "savgol") -> dict:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    lc_dict : dict
        Output from ingestion module
    method : str
        Detrending method: 'savgol' or 'spline'

    Returns
    -------
    dict with cleaned and detrended light curve data
    """
    time = lc_dict["time"].copy()
    flux = lc_dict["flux"].copy()
    flux_err = lc_dict["flux_err"].copy()

    # Step 1: normalise
    flux = normalise(flux)

    # Step 2: remove outliers
    time, flux, flux_err = remove_outliers(time, flux, flux_err, sigma=4.5)

    # Step 3: detrend
    if method == "spline":
        flux_clean, trend = detrend_spline(time, flux)
    else:
        flux_clean, trend = detrend_savgol(flux)

    # Step 4: second sigma-clip on residuals
    mask = sigma_clip(flux_clean, sigma=5.0)
    time = time[mask]
    flux_clean = flux_clean[mask]
    flux_err = flux_err[mask]
    trend = trend[mask]

    result = {**lc_dict}
    result.update({
        "time": time,
        "flux_raw": flux[mask],
        "flux": flux_clean,
        "flux_err": flux_err,
        "trend": trend,
        "detrend_method": method,
        "n_points": len(time),
    })

    logger.info(f"Preprocessing complete: {len(time)} clean data points")
    return result
