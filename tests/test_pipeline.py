"""
tests/test_pipeline.py
Basic unit tests for the exoplanet detection pipeline.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_ingestion_synthetic():
    """Synthetic light curve generation produces correct structure."""
    from src.ingestion import generate_synthetic_lc
    for sig_type in ["transit", "binary", "stellar_activity", "blend", "noise"]:
        lc = generate_synthetic_lc(sig_type, n_points=200)
        assert "time" in lc and "flux" in lc and "flux_err" in lc
        assert len(lc["time"]) == 200
        assert lc["true_type"] == sig_type


def test_preprocessing():
    """Preprocessing cleans and detrends a synthetic LC."""
    from src.ingestion import generate_synthetic_lc
    from src.preprocessing import preprocess
    lc = generate_synthetic_lc("transit", n_points=300)
    clean = preprocess(lc, method="savgol")
    assert "flux" in clean
    assert clean["n_points"] <= 300
    assert np.abs(np.median(clean["flux"]) - 1.0) < 0.01, "Flux not normalised to ~1"


def test_bls_finds_period():
    """BLS recovers the correct period from a synthetic transit."""
    from src.ingestion import generate_synthetic_lc
    from src.preprocessing import preprocess
    from src.bls_detection import run_bls
    np.random.seed(7)
    lc = generate_synthetic_lc("transit", n_points=600)
    clean = preprocess(lc)
    t, f, fe = clean["time"], clean["flux"], clean["flux_err"]
    bls = run_bls(t, f, fe, n_periods=2000)
    assert bls["snr"] > 0
    assert 0.1 < bls["best_period"] < 27.0


def test_feature_extraction():
    """Feature extraction returns the right shapes."""
    from src.ingestion import generate_synthetic_lc
    from src.preprocessing import preprocess
    from src.bls_detection import run_bls, phase_fold
    from src.feature_extraction import build_feature_vector
    lc = generate_synthetic_lc("transit", n_points=400)
    clean = preprocess(lc)
    t, f, fe = clean["time"], clean["flux"], clean["flux_err"]
    bls = run_bls(t, f, fe, n_periods=1000)
    folded = phase_fold(t, f, bls["best_period"], bls["best_t0"])
    feat_vec, folded_1d, feat_names = build_feature_vector(clean, bls, folded)
    assert len(feat_vec) == len(feat_names)
    assert len(folded_1d) == 128
    assert np.all(np.isfinite(feat_vec))


def test_rule_based_classifier():
    """Rule-based classifier returns valid probabilities for all classes."""
    from src.classifier import rule_based_classify, LABELS
    bls_mock = {"snr": 20.0, "best_depth": 0.01, "best_period": 5.0}
    shape_mock = {"depth_ratio": 0.05, "shape_ratio": 1.8, "asymmetry": 0.1,
                  "oot_rms": 0.001, "secondary_depth": 0.0005}
    result = rule_based_classify(bls_mock, shape_mock)
    assert result["classification"] in LABELS
    assert 0.0 <= result["confidence"] <= 1.0
    assert abs(sum(result["probabilities"].values()) - 1.0) < 1e-5


def test_parameter_estimation():
    """Parameter estimation returns physically reasonable values."""
    from src.parameters import estimate_all_parameters
    bls_mock = {
        "best_period": 3.52, "best_depth": 0.0145, "best_duration": 0.115,
        "best_t0": 2459000.0, "snr": 28.4
    }
    params = estimate_all_parameters(bls_mock)
    assert params["period_days"] == 3.52
    assert params["r_planet_rearth"] is not None
    assert params["r_planet_rearth"] > 0
    assert 0 < params["semi_major_axis_au"] < 1.0
    assert params["equilibrium_temp_k"] > 200


def test_full_pipeline_synthetic():
    """End-to-end pipeline on a synthetic transit LC."""
    from src.pipeline import ExoplanetPipeline
    pipe = ExoplanetPipeline()
    result = pipe.run_synthetic("transit")
    assert "classification" in result
    assert "probabilities" in result
    assert "parameters" in result
    assert "bls" in result
    assert result["bls"]["snr"] > 0
    assert len(result["probabilities"]) == 5


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
