"""
BLS Transit Detection Module
Implements Box Least Squares (BLS) algorithm to find periodic transit signals
in detrended TESS light curves.
"""

import numpy as np
from astropy.timeseries import BoxLeastSquares
from astropy import units as u
import logging

logger = logging.getLogger(__name__)


def run_bls(time: np.ndarray, flux: np.ndarray, flux_err: np.ndarray,
            period_min: float = 0.5, period_max: float = 27.0,
            n_periods: int = 10000, duration_grid: list = None) -> dict:
    """
    Run Box Least Squares on a detrended light curve.

    Parameters
    ----------
    time : np.ndarray
        Time array in days
    flux : np.ndarray
        Detrended, normalised flux (median ~ 1.0)
    flux_err : np.ndarray
        Flux uncertainties
    period_min : float
        Minimum period to search (days)
    period_max : float
        Maximum period to search (days). Default: length of data.
    n_periods : int
        Number of periods to sample
    duration_grid : list
        Transit durations to test (days). Default: auto-generated.

    Returns
    -------
    dict with BLS results including best period, depth, duration, power spectrum
    """
    if period_max > (time[-1] - time[0]):
        period_max = (time[-1] - time[0]) * 0.9

    periods = np.exp(np.linspace(np.log(period_min), np.log(period_max), n_periods))

    if duration_grid is None:
        duration_grid = np.array([0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])

    logger.info(f"Running BLS: period range [{period_min:.2f}, {period_max:.2f}] days, "
                f"{n_periods} periods, {len(duration_grid)} durations")

    bls = BoxLeastSquares(time * u.day, flux, dy=flux_err)
    result = bls.power(periods * u.day, duration_grid * u.day, objective="snr")

    # Best period
    best_idx = np.argmax(result.power)
    best_period = result.period[best_idx].value
    best_power = result.power[best_idx]
    best_duration = result.duration[best_idx].value
    best_t0 = result.transit_time[best_idx].value
    best_depth = result.depth[best_idx]

    # Stats at best period
    stats = bls.compute_stats(best_period * u.day, best_duration * u.day, best_t0 * u.day)

    snr = float(stats["depth"][0] / stats["depth"][1]) if stats["depth"][1] > 0 else 0.0

    # Phase-folded data at best period
    phase = ((time - best_t0) % best_period) / best_period
    phase[phase > 0.5] -= 1.0  # centre transit at phase 0

    logger.info(f"BLS best: period={best_period:.4f}d, depth={best_depth:.5f}, "
                f"duration={best_duration*24:.2f}h, SNR={snr:.1f}")

    return {
        "periods": result.period.value,
        "power": result.power,
        "best_period": best_period,
        "best_power": float(best_power),
        "best_duration": best_duration,
        "best_t0": best_t0,
        "best_depth": float(best_depth),
        "snr": snr,
        "phase": phase,
        "stats": stats,
    }


def extract_transit_candidates(bls_result: dict, time: np.ndarray,
                               flux: np.ndarray, n_candidates: int = 3) -> list:
    """
    Extract top N transit candidates from BLS power spectrum using peak finding.

    Parameters
    ----------
    bls_result : dict
        Output from run_bls()
    time, flux : np.ndarray
        Cleaned light curve arrays
    n_candidates : int
        Maximum number of candidates to return

    Returns
    -------
    List of candidate dicts, sorted by BLS power (descending)
    """
    periods = bls_result["periods"]
    power = np.array(bls_result["power"])
    candidates = []

    # Simple peak finding with minimum period spacing
    remaining_power = power.copy()
    for _ in range(n_candidates):
        if np.max(remaining_power) < 0.01:
            break
        peak_idx = np.argmax(remaining_power)
        peak_period = periods[peak_idx]

        # Mask harmonics and aliases within 20% of this period
        for harm in [0.5, 1.0, 2.0, 3.0]:
            alias = peak_period * harm
            mask = np.abs(periods - alias) / alias < 0.2
            remaining_power[mask] = 0

        candidates.append({
            "period": float(peak_period),
            "power": float(power[peak_idx]),
            "rank": len(candidates) + 1,
        })

    return candidates


def phase_fold(time: np.ndarray, flux: np.ndarray, period: float,
               t0: float, n_bins: int = 100) -> dict:
    """
    Phase-fold a light curve and optionally bin.

    Returns
    -------
    dict with phase, flux, and binned versions
    """
    phase = ((time - t0) % period) / period
    phase[phase > 0.5] -= 1.0
    sort_idx = np.argsort(phase)
    phase_sorted = phase[sort_idx]
    flux_sorted = flux[sort_idx]

    # Bin
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    bin_flux = np.full(n_bins, np.nan)
    for i in range(n_bins):
        mask = (phase_sorted >= bins[i]) & (phase_sorted < bins[i + 1])
        if mask.sum() > 0:
            bin_flux[i] = np.median(flux_sorted[mask])

    return {
        "phase": phase_sorted,
        "flux": flux_sorted,
        "bin_phase": bin_centers,
        "bin_flux": bin_flux,
    }
