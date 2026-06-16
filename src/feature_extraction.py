"""
Feature Extraction Module
Extracts discriminating features from BLS results and phase-folded light curves
for input to the CNN classifier.
"""

import numpy as np
from scipy.stats import kurtosis, skew
import logging

logger = logging.getLogger(__name__)


def extract_transit_shape_features(bin_flux: np.ndarray, bin_phase: np.ndarray,
                                   depth: float, duration: float) -> dict:
    """
    Extract shape-based features from a phase-folded, binned transit.

    Features capture:
    - Transit depth and duration ratios
    - Ingress/egress symmetry (planetary vs V-shaped binary)
    - Presence of secondary eclipse (binary)
    - Out-of-transit variability
    """
    mask = np.isfinite(bin_flux)
    if mask.sum() < 10:
        return {}

    flux = bin_flux[mask]
    phase = bin_phase[mask]

    # In-transit mask (~20% of phase around 0)
    in_transit = np.abs(phase) < 0.1
    out_of_transit = np.abs(phase) > 0.2

    flux_in = flux[in_transit] if in_transit.sum() > 0 else np.array([1.0])
    flux_out = flux[out_of_transit] if out_of_transit.sum() > 0 else np.array([1.0])

    # Secondary eclipse at phase ±0.5
    sec_mask = np.abs(np.abs(phase) - 0.5) < 0.1
    flux_sec = flux[sec_mask] if sec_mask.sum() > 0 else np.array([1.0])

    measured_depth = 1.0 - np.median(flux_in)
    secondary_depth = 1.0 - np.median(flux_sec)
    oot_rms = np.std(flux_out) if len(flux_out) > 1 else 0.0
    oot_mean = np.mean(flux_out)

    # Shape: U (planetary) vs V (binary) - ratio of mid-transit to edge
    mid_mask = np.abs(phase) < 0.03
    edge_mask = (np.abs(phase) > 0.06) & (np.abs(phase) < 0.1)
    mid_depth = 1.0 - np.mean(flux[mid_mask]) if mid_mask.sum() > 0 else measured_depth
    edge_depth = 1.0 - np.mean(flux[edge_mask]) if edge_mask.sum() > 0 else measured_depth * 0.5
    shape_ratio = mid_depth / (edge_depth + 1e-10)  # >1 = U-shaped (planetary), ~1 = V (binary)

    # Odd-even depth difference (binary sign)
    left_mask = (phase > -0.1) & (phase < -0.02)
    right_mask = (phase > 0.02) & (phase < 0.1)
    left_depth = 1.0 - np.mean(flux[left_mask]) if left_mask.sum() > 0 else measured_depth
    right_depth = 1.0 - np.mean(flux[right_mask]) if right_mask.sum() > 0 else measured_depth
    asymmetry = abs(left_depth - right_depth) / (measured_depth + 1e-10)

    return {
        "measured_depth": measured_depth,
        "secondary_depth": secondary_depth,
        "depth_ratio": secondary_depth / (measured_depth + 1e-10),
        "oot_rms": oot_rms,
        "oot_mean": oot_mean,
        "shape_ratio": shape_ratio,
        "asymmetry": asymmetry,
        "skewness": float(skew(flux_in)) if len(flux_in) > 3 else 0.0,
        "kurtosis_val": float(kurtosis(flux_in)) if len(flux_in) > 3 else 0.0,
    }


def extract_bls_features(bls_result: dict) -> dict:
    """Extract statistical features from the BLS power spectrum."""
    power = np.array(bls_result["power"])
    best_power = bls_result["best_power"]
    median_power = np.median(power)
    std_power = np.std(power)

    return {
        "bls_snr": bls_result["snr"],
        "bls_peak_power": best_power,
        "bls_peak_to_median": best_power / (median_power + 1e-10),
        "bls_peak_to_std": (best_power - median_power) / (std_power + 1e-10),
        "bls_period": bls_result["best_period"],
        "bls_depth": bls_result["best_depth"],
        "bls_duration_hours": bls_result["best_duration"] * 24,
        "period_log": np.log10(bls_result["best_period"]),
        "depth_log": np.log10(max(bls_result["best_depth"], 1e-6)),
    }


def extract_lc_features(flux: np.ndarray) -> dict:
    """Extract global light curve statistical features."""
    return {
        "lc_std": float(np.std(flux)),
        "lc_skew": float(skew(flux)),
        "lc_kurtosis": float(kurtosis(flux)),
        "lc_p05": float(np.percentile(flux, 5)),
        "lc_p95": float(np.percentile(flux, 95)),
        "lc_peak_to_peak": float(np.ptp(flux)),
        "lc_n_points": len(flux),
    }


def build_feature_vector(lc_data: dict, bls_result: dict, folded: dict) -> np.ndarray:
    """
    Combine all features into a single normalised feature vector for the classifier.

    Also returns a 1D local view of the phase-folded transit (for CNN 1D input).

    Returns
    -------
    feature_vector : np.ndarray, shape (n_features,)
    folded_1d : np.ndarray, shape (128,)  — interpolated phase-folded transit
    feature_names : list of str
    """
    shape_feats = extract_transit_shape_features(
        folded["bin_flux"], folded["bin_phase"],
        bls_result["best_depth"], bls_result["best_duration"]
    )
    bls_feats = extract_bls_features(bls_result)
    lc_feats = extract_lc_features(lc_data["flux"])

    all_feats = {**bls_feats, **lc_feats, **shape_feats}
    feature_names = sorted(all_feats.keys())
    feature_vector = np.array([all_feats.get(k, 0.0) for k in feature_names])
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1e6, neginf=-1e6)

    # 1D folded transit for CNN: interpolate to fixed 128 bins
    bin_flux = folded["bin_flux"]
    bin_phase = folded["bin_phase"]
    mask = np.isfinite(bin_flux)
    if mask.sum() > 5:
        from scipy.interpolate import interp1d
        interp = interp1d(bin_phase[mask], bin_flux[mask], kind="linear",
                          fill_value=1.0, bounds_error=False)
        phase_grid = np.linspace(-0.5, 0.5, 128)
        folded_1d = interp(phase_grid)
    else:
        folded_1d = np.ones(128)

    logger.info(f"Extracted {len(feature_names)} features for classification")
    return feature_vector, folded_1d, feature_names
