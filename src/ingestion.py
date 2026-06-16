"""
Data Ingestion Module
Downloads and loads TESS light curves from MAST archive via lightkurve.
"""

import numpy as np
import lightkurve as lk
from astropy.io import fits
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_tess_lc(target: str, sector: int = None, exptime: str = "short") -> dict:
    """
    Download a TESS light curve for a given target.

    Parameters
    ----------
    target : str
        Target name or TIC ID (e.g. 'TIC 261136679' or 'TOI-700')
    sector : int, optional
        TESS sector number. If None, uses all available sectors.
    exptime : str
        Exposure time: 'short' (2-min), 'long' (10-min), or 'fast' (20-sec)

    Returns
    -------
    dict with keys: time, flux, flux_err, quality, sector, target
    """
    logger.info(f"Searching for {target} in TESS archive...")
    search = lk.search_lightcurve(target, mission="TESS", exptime=exptime)
    if len(search) == 0:
        raise ValueError(f"No TESS light curves found for target: {target}")

    if sector is not None:
        search = search[search.table["#sector"] == sector]
        if len(search) == 0:
            raise ValueError(f"No light curve found for sector {sector}")

    logger.info(f"Found {len(search)} light curve(s). Downloading first...")
    lc = search[0].download()
    lc = lc.remove_nans().remove_outliers(sigma=5)

    return {
        "time": lc.time.value,
        "flux": lc.flux.value,
        "flux_err": lc.flux_err.value if lc.flux_err is not None else np.ones_like(lc.flux.value) * 1e-4,
        "quality": lc.quality.value if hasattr(lc, "quality") else np.zeros(len(lc.time)),
        "sector": search[0].table["#sector"][0],
        "target": target,
        "cadence": exptime,
    }


def load_fits_file(filepath: str) -> dict:
    """
    Load a TESS FITS light curve file from disk.

    Parameters
    ----------
    filepath : str
        Path to the .fits or .fit file

    Returns
    -------
    dict with keys: time, flux, flux_err, quality, sector, target
    """
    logger.info(f"Loading FITS file: {filepath}")
    with fits.open(filepath) as hdul:
        # Try PDCSAP flux first, fallback to SAP
        data = hdul[1].data
        time = data["TIME"]
        if "PDCSAP_FLUX" in data.names:
            flux = data["PDCSAP_FLUX"]
            flux_err = data["PDCSAP_FLUX_ERR"]
        elif "SAP_FLUX" in data.names:
            flux = data["SAP_FLUX"]
            flux_err = data["SAP_FLUX_ERR"]
        else:
            flux = data["FLUX"]
            flux_err = np.ones_like(flux) * 1e-4

        quality = data["QUALITY"] if "QUALITY" in data.names else np.zeros(len(time))
        header = hdul[0].header
        target = header.get("OBJECT", "Unknown")
        sector = header.get("SECTOR", 0)

    # Remove NaNs and bad quality flags
    mask = np.isfinite(time) & np.isfinite(flux) & (quality == 0)
    flux_norm = flux[mask] / np.nanmedian(flux[mask])

    return {
        "time": time[mask],
        "flux": flux_norm,
        "flux_err": flux_err[mask] / np.nanmedian(flux[mask]),
        "quality": quality[mask],
        "sector": sector,
        "target": target,
        "cadence": "unknown",
    }


def generate_synthetic_lc(signal_type: str = "transit", n_points: int = 1000) -> dict:
    """
    Generate a synthetic light curve for testing/demo purposes.

    Parameters
    ----------
    signal_type : str
        One of: 'transit', 'binary', 'stellar_activity', 'blend', 'noise'
    n_points : int
        Number of data points

    Returns
    -------
    dict with keys: time, flux, flux_err, quality, sector, target
    """
    np.random.seed(42)
    time = np.linspace(0, 27.0, n_points)  # 27-day TESS sector
    noise = np.random.normal(0, 5e-4, n_points)
    flux = np.ones(n_points) + noise

    if signal_type == "transit":
        period = np.random.uniform(2, 20)
        depth = np.random.uniform(0.001, 0.02)
        duration_frac = 0.05
        for i, t in enumerate(time):
            phase = (t % period) / period
            if abs(phase - 0.5) < duration_frac / 2:
                u = (phase - 0.5) / (duration_frac / 2)
                flux[i] -= depth * (1 - u**2)  # limb-darkened transit shape

    elif signal_type == "binary":
        period = np.random.uniform(1, 5)
        depth_pri = np.random.uniform(0.05, 0.3)
        depth_sec = depth_pri * np.random.uniform(0.3, 0.8)
        for i, t in enumerate(time):
            phase = (t % period) / period
            if phase < 0.1 or phase > 0.9:
                flux[i] -= depth_pri
            elif 0.4 < phase < 0.6:
                flux[i] -= depth_sec

    elif signal_type == "stellar_activity":
        period = np.random.uniform(5, 25)
        amp = np.random.uniform(0.003, 0.015)
        flux += amp * np.sin(2 * np.pi * time / period)
        flux += (amp / 3) * np.sin(4 * np.pi * time / period + 0.5)
        # Add a flare
        flare_t = time[n_points // 3]
        flux += 0.02 * np.exp(-((time - flare_t) ** 2) / 0.01)

    elif signal_type == "blend":
        period = np.random.uniform(3, 15)
        depth = np.random.uniform(0.02, 0.08)
        dilution = np.random.uniform(0.3, 0.7)
        for i, t in enumerate(time):
            phase = (t % period) / period
            if abs(phase - 0.5) < 0.06:
                flux[i] -= depth * dilution

    # noise: just the noise baseline (already set)

    flux_err = np.ones(n_points) * 5e-4

    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "quality": np.zeros(n_points),
        "sector": 1,
        "target": f"Synthetic_{signal_type}",
        "cadence": "synthetic",
        "true_type": signal_type,
    }
