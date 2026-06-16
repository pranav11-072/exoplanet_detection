"""
Parameter Estimation Module
Derives physical planetary parameters from BLS transit detection results.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

# Solar constants
R_SUN_KM = 695700.0
R_EARTH_KM = 6371.0
R_JUP_KM = 71492.0
AU_KM = 1.496e8


def estimate_planet_radius(depth: float, r_star_rsun: float = 1.0) -> dict:
    """
    Estimate planet radius from transit depth.

    depth = (Rp / Rs)^2  =>  Rp = Rs * sqrt(depth)

    Parameters
    ----------
    depth : float
        Transit depth (fractional flux loss, e.g. 0.01 = 1%)
    r_star_rsun : float
        Stellar radius in solar radii (default: 1.0 = Sun-like)

    Returns
    -------
    dict with Rp in Earth radii and Jupiter radii
    """
    if depth <= 0:
        return {"r_planet_rearth": None, "r_planet_rjup": None, "rp_rs_ratio": None}

    rp_rs = np.sqrt(depth)
    r_planet_km = rp_rs * r_star_rsun * R_SUN_KM

    return {
        "rp_rs_ratio": rp_rs,
        "r_planet_km": r_planet_km,
        "r_planet_rearth": r_planet_km / R_EARTH_KM,
        "r_planet_rjup": r_planet_km / R_JUP_KM,
    }


def estimate_semi_major_axis(period_days: float, m_star_msun: float = 1.0) -> dict:
    """
    Estimate semi-major axis from orbital period using Kepler's third law.

    a^3 / P^2 = G * M_star / (4 * pi^2)  =>  a = (G M P^2 / 4pi^2)^(1/3)

    Parameters
    ----------
    period_days : float
        Orbital period in days
    m_star_msun : float
        Stellar mass in solar masses

    Returns
    -------
    dict with semi-major axis in AU and km
    """
    period_yr = period_days / 365.25
    a_au = (m_star_msun * period_yr**2) ** (1.0 / 3.0)  # Kepler's 3rd law (solar units)
    return {
        "a_au": a_au,
        "a_km": a_au * AU_KM,
    }


def estimate_impact_parameter(duration_days: float, period_days: float,
                              rp_rs: float, r_star_rsun: float = 1.0,
                              m_star_msun: float = 1.0) -> float:
    """
    Estimate the transit impact parameter b.

    b = a/Rs * cos(i)
    T = (Rs/pi*a) * sqrt((1 + rp_rs)^2 - b^2) * P / sqrt(1 - e^2)  (circular)

    Parameters
    ----------
    duration_days : float
        Transit duration in days
    period_days : float
        Orbital period in days
    rp_rs : float
        Planet-to-star radius ratio
    r_star_rsun : float
        Stellar radius in solar radii
    m_star_msun : float
        Stellar mass in solar masses

    Returns
    -------
    float : impact parameter (0 = central transit, 1 = grazing)
    """
    a_info = estimate_semi_major_axis(period_days, m_star_msun)
    a_rs = a_info["a_au"] * AU_KM / (r_star_rsun * R_SUN_KM)

    sin_term = np.sin(np.pi * duration_days / period_days)
    inner = (1 + rp_rs)**2 - (a_rs * sin_term)**2
    b = np.sqrt(max(inner, 0.0))
    return float(np.clip(b, 0, 1))


def estimate_equilibrium_temperature(a_au: float, t_star_k: float = 5778.0,
                                     r_star_rsun: float = 1.0,
                                     albedo: float = 0.3) -> float:
    """
    Estimate planet equilibrium temperature (blackbody approximation).

    T_eq = T_star * (R_star / 2a)^0.5 * (1 - A)^0.25

    Parameters
    ----------
    a_au : float
        Semi-major axis in AU
    t_star_k : float
        Stellar effective temperature in K
    r_star_rsun : float
        Stellar radius in solar radii
    albedo : float
        Bond albedo (default: 0.3 — Earth-like)

    Returns
    -------
    float : equilibrium temperature in K
    """
    r_star_au = r_star_rsun * R_SUN_KM / AU_KM
    t_eq = t_star_k * np.sqrt(r_star_au / (2 * a_au)) * (1 - albedo) ** 0.25
    return float(t_eq)


def insolation_flux(a_au: float, l_star_lsun: float = 1.0) -> float:
    """
    Stellar insolation at planet's orbit in Earth units (S_Earth = 1361 W/m^2).

    S = L_star / a^2  (in solar units, 1 AU from Sun gives S=1)
    """
    return float(l_star_lsun / a_au**2)


def estimate_all_parameters(bls_result: dict, r_star_rsun: float = 1.0,
                             m_star_msun: float = 1.0,
                             t_star_k: float = 5778.0,
                             l_star_lsun: float = 1.0) -> dict:
    """
    Full parameter estimation from BLS results and stellar properties.

    Parameters
    ----------
    bls_result : dict
        Output from bls_detection.run_bls()
    r_star_rsun, m_star_msun, t_star_k, l_star_lsun : float
        Stellar parameters (default: solar)

    Returns
    -------
    dict with all estimated planetary parameters
    """
    period = bls_result["best_period"]
    depth = bls_result["best_depth"]
    duration = bls_result["best_duration"]  # days
    snr = bls_result["snr"]

    radius_info = estimate_planet_radius(depth, r_star_rsun)
    orbit_info = estimate_semi_major_axis(period, m_star_msun)

    rp_rs = radius_info.get("rp_rs_ratio", 0) or 0
    b = estimate_impact_parameter(duration, period, rp_rs, r_star_rsun, m_star_msun)
    t_eq = estimate_equilibrium_temperature(orbit_info["a_au"], t_star_k, r_star_rsun)
    insol = insolation_flux(orbit_info["a_au"], l_star_lsun)

    params = {
        # Transit observables
        "period_days": round(period, 4),
        "epoch_bjd": round(bls_result["best_t0"], 4),
        "depth_ppm": round(depth * 1e6, 1),
        "depth_percent": round(depth * 100, 4),
        "duration_hours": round(duration * 24, 2),
        "snr": round(snr, 1),

        # Planetary properties
        "rp_rs_ratio": round(rp_rs, 4) if rp_rs else None,
        "r_planet_rearth": round(radius_info["r_planet_rearth"], 2) if radius_info["r_planet_rearth"] else None,
        "r_planet_rjup": round(radius_info["r_planet_rjup"], 3) if radius_info["r_planet_rjup"] else None,

        # Orbital properties
        "semi_major_axis_au": round(orbit_info["a_au"], 4),
        "impact_parameter": round(b, 3),

        # Habitability indicators
        "equilibrium_temp_k": round(t_eq, 1),
        "insolation_earth_units": round(insol, 3),
        "in_habitable_zone": bool(0.25 < insol < 1.5),

        # Planet category
        "planet_category": categorise_planet(
            radius_info.get("r_planet_rearth"), period
        ),
    }

    logger.info(f"Parameters estimated: P={period:.2f}d, Rp={params['r_planet_rearth']} R⊕, "
                f"a={params['semi_major_axis_au']} AU, T_eq={t_eq:.0f} K")
    return params


def categorise_planet(r_earth: float, period_days: float) -> str:
    """Classify planet by radius (Fulton gap informed)."""
    if r_earth is None:
        return "Unknown"
    if r_earth < 1.5:
        return "Super-Earth / Rocky"
    elif r_earth < 2.5:
        return "Sub-Neptune"
    elif r_earth < 4.0:
        return "Neptune-like"
    elif r_earth < 11.2:
        cat = "Hot Jupiter" if period_days < 10 else "Gas Giant"
        return cat
    else:
        return "Ultra-Hot Jupiter / Giant"
