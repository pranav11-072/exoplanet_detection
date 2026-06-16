"""
Detection Pipeline
End-to-end orchestrator: ingest → preprocess → BLS → features → classify → parameters.
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

from .ingestion import load_fits_file, generate_synthetic_lc
from .preprocessing import preprocess
from .bls_detection import run_bls, phase_fold
from .feature_extraction import build_feature_vector, extract_transit_shape_features, extract_bls_features
from .classifier import rule_based_classify, LABELS
from .parameters import estimate_all_parameters

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class ExoplanetPipeline:
    """
    Full TESS exoplanet detection pipeline.

    Usage
    -----
    >>> pipe = ExoplanetPipeline()
    >>> result = pipe.run_fits("path/to/lightcurve.fits")
    >>> result = pipe.run_synthetic("transit")
    >>> result = pipe.run_batch(["file1.fits", "file2.fits"])
    """

    def __init__(self, model_path: str = None, use_rule_based: bool = True):
        """
        Parameters
        ----------
        model_path : str, optional
            Path to trained Keras model (.keras). If None, uses rule-based classifier.
        use_rule_based : bool
            If True and model_path is None, use rule-based fallback.
        """
        self.model = None
        self.use_rule_based = use_rule_based

        if model_path and os.path.exists(model_path):
            try:
                from .classifier import load_model
                self.model = load_model(model_path)
                logger.info("CNN model loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load model: {e}. Using rule-based classifier.")
        else:
            logger.info("No CNN model found. Using rule-based classifier.")

    def run(self, lc_dict: dict, stellar_params: dict = None) -> dict:
        """
        Run the full pipeline on a pre-loaded light curve dict.

        Parameters
        ----------
        lc_dict : dict
            Output from ingestion module
        stellar_params : dict, optional
            Stellar properties: r_star_rsun, m_star_msun, t_star_k, l_star_lsun

        Returns
        -------
        dict with all pipeline outputs
        """
        if stellar_params is None:
            stellar_params = {"r_star_rsun": 1.0, "m_star_msun": 1.0,
                              "t_star_k": 5778.0, "l_star_lsun": 1.0}

        target = lc_dict.get("target", "Unknown")
        logger.info(f"\n{'='*50}\nProcessing: {target}\n{'='*50}")

        # Step 1: Preprocess
        logger.info("[1/5] Preprocessing...")
        lc_clean = preprocess(lc_dict, method="savgol")

        # Step 2: BLS detection
        logger.info("[2/5] Running BLS transit detection...")
        t = lc_clean["time"]
        f = lc_clean["flux"]
        fe = lc_clean["flux_err"]
        period_max = min((t[-1] - t[0]) * 0.9, 27.0)
        bls_result = run_bls(t, f, fe, period_min=0.5, period_max=period_max)

        # Step 3: Phase fold at best period
        logger.info("[3/5] Phase folding...")
        folded = phase_fold(t, f, bls_result["best_period"], bls_result["best_t0"])

        # Step 4: Feature extraction
        logger.info("[4/5] Extracting features...")
        feature_vector, folded_1d, feature_names = build_feature_vector(lc_clean, bls_result, folded)
        shape_feats = extract_transit_shape_features(
            folded["bin_flux"], folded["bin_phase"],
            bls_result["best_depth"], bls_result["best_duration"]
        )

        # Step 5: Classify
        logger.info("[5/5] Classifying signal...")
        if self.model is not None:
            from .classifier import predict
            classification = predict(self.model, folded_1d, feature_vector)
            classification["method"] = "CNN"
        else:
            classification = rule_based_classify(bls_result, shape_feats)

        # Step 6: Estimate parameters (if transit-like)
        params = estimate_all_parameters(bls_result, **stellar_params)

        result = {
            "target": target,
            "sector": lc_dict.get("sector", 0),
            "n_points": lc_clean["n_points"],
            "classification": classification["classification"],
            "confidence": round(classification["confidence"], 4),
            "probabilities": {k: round(v, 4) for k, v in classification["probabilities"].items()},
            "classifier_method": classification.get("method", "rule_based"),
            "bls": {
                "period_days": round(bls_result["best_period"], 4),
                "depth": round(bls_result["best_depth"], 6),
                "duration_hours": round(bls_result["best_duration"] * 24, 2),
                "snr": round(bls_result["snr"], 1),
                "peak_power": round(bls_result["best_power"], 4),
                "t0_bjd": round(bls_result["best_t0"], 4),
            },
            "parameters": params,
            "shape_features": {k: round(v, 4) if isinstance(v, float) else v
                               for k, v in shape_feats.items()},
            # Raw arrays for plotting
            "_lc": {"time": t.tolist(), "flux": f.tolist()},
            "_bls_periods": bls_result["periods"].tolist(),
            "_bls_power": [float(p) for p in bls_result["power"]],
            "_folded": {"phase": folded["phase"].tolist(), "flux": folded["flux"].tolist(),
                        "bin_phase": folded["bin_phase"].tolist(),
                        "bin_flux": [float(v) if np.isfinite(v) else None
                                     for v in folded["bin_flux"]]},
        }

        self._print_summary(result)
        return result

    def run_fits(self, filepath: str, stellar_params: dict = None) -> dict:
        """Run pipeline on a FITS file."""
        lc_dict = load_fits_file(filepath)
        return self.run(lc_dict, stellar_params)

    def run_synthetic(self, signal_type: str = "transit",
                      stellar_params: dict = None) -> dict:
        """Run pipeline on a synthetic light curve (for testing/demo)."""
        lc_dict = generate_synthetic_lc(signal_type)
        return self.run(lc_dict, stellar_params)

    def run_batch(self, filepaths: list, output_csv: str = "results/batch_results.csv",
                  stellar_params: dict = None) -> pd.DataFrame:
        """
        Process multiple FITS files and export results to CSV.

        Parameters
        ----------
        filepaths : list of str
        output_csv : str
        stellar_params : dict, optional

        Returns
        -------
        pd.DataFrame with one row per star
        """
        rows = []
        n = len(filepaths)
        for i, fp in enumerate(filepaths):
            logger.info(f"\nBatch [{i+1}/{n}]: {fp}")
            try:
                result = self.run_fits(fp, stellar_params)
                rows.append({
                    "filename": os.path.basename(fp),
                    "target": result["target"],
                    "sector": result["sector"],
                    "classification": result["classification"],
                    "confidence": result["confidence"],
                    "period_days": result["bls"]["period_days"],
                    "depth_ppm": result["parameters"]["depth_ppm"],
                    "duration_hours": result["bls"]["duration_hours"],
                    "snr": result["bls"]["snr"],
                    "r_planet_rearth": result["parameters"].get("r_planet_rearth"),
                    "semi_major_axis_au": result["parameters"].get("semi_major_axis_au"),
                    "equilibrium_temp_k": result["parameters"].get("equilibrium_temp_k"),
                    "in_habitable_zone": result["parameters"].get("in_habitable_zone"),
                    "planet_category": result["parameters"].get("planet_category"),
                    "transit_prob": result["probabilities"].get("Exoplanet Transit", 0),
                })
            except Exception as e:
                logger.error(f"Failed on {fp}: {e}")
                rows.append({"filename": os.path.basename(fp), "error": str(e)})

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(output_csv) if os.path.dirname(output_csv) else ".", exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info(f"\nBatch complete. Results saved to {output_csv}")
        transit_count = (df["classification"] == "Exoplanet Transit").sum()
        logger.info(f"Transit candidates: {transit_count}/{len(df)}")
        return df

    def _print_summary(self, result: dict):
        print("\n" + "="*50)
        print(f"  Target  : {result['target']}")
        print(f"  Result  : {result['classification']} ({result['confidence']*100:.1f}%)")
        print(f"  Period  : {result['bls']['period_days']} days")
        print(f"  Depth   : {result['parameters']['depth_ppm']} ppm")
        print(f"  Duration: {result['bls']['duration_hours']} hours")
        print(f"  SNR     : {result['bls']['snr']}")
        if result["parameters"].get("r_planet_rearth"):
            print(f"  Rp      : {result['parameters']['r_planet_rearth']} R⊕")
        if result["parameters"].get("in_habitable_zone"):
            print("  ★ In habitable zone!")
        print("="*50 + "\n")
