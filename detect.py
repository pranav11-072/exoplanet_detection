"""
detect.py
Command-line interface for running the exoplanet detection pipeline.

Usage:
    # Analyse a FITS file
    python detect.py --fits path/to/lightcurve.fits

    # Run on a synthetic demo
    python detect.py --demo transit

    # Batch mode
    python detect.py --batch path/to/fits_dir/ --output results/batch.csv

    # With stellar parameters
    python detect.py --demo transit --r-star 0.8 --m-star 0.85 --t-star 5200
"""

import argparse
import json
import sys
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(
        description="AI Exoplanet Detection System — TESS light curve analyser",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--fits", help="Path to a TESS FITS light curve file")
    group.add_argument("--demo", choices=["transit", "binary", "stellar_activity", "blend", "noise"],
                       help="Run on a synthetic demo light curve")
    group.add_argument("--batch", help="Directory containing FITS files for batch processing")

    parser.add_argument("--model", default=None, help="Path to trained CNN model (.keras)")
    parser.add_argument("--output", default=None, help="Output CSV path (batch mode)")
    parser.add_argument("--json", action="store_true", help="Print result as JSON")

    # Stellar parameters
    parser.add_argument("--r-star", type=float, default=1.0, metavar="R_SUN",
                        help="Stellar radius in solar radii (default: 1.0)")
    parser.add_argument("--m-star", type=float, default=1.0, metavar="M_SUN",
                        help="Stellar mass in solar masses (default: 1.0)")
    parser.add_argument("--t-star", type=float, default=5778.0, metavar="K",
                        help="Stellar effective temperature in K (default: 5778)")
    parser.add_argument("--l-star", type=float, default=1.0, metavar="L_SUN",
                        help="Stellar luminosity in solar units (default: 1.0)")

    args = parser.parse_args()

    stellar = {
        "r_star_rsun": args.r_star,
        "m_star_msun": args.m_star,
        "t_star_k": args.t_star,
        "l_star_lsun": args.l_star,
    }

    from src.pipeline import ExoplanetPipeline
    pipe = ExoplanetPipeline(model_path=args.model)

    if args.fits:
        if not os.path.exists(args.fits):
            print(f"Error: File not found: {args.fits}", file=sys.stderr)
            sys.exit(1)
        result = pipe.run_fits(args.fits, stellar_params=stellar)

        if args.json:
            # Remove raw array data for cleaner JSON output
            for k in ["_lc", "_bls_periods", "_bls_power", "_folded"]:
                result.pop(k, None)
            print(json.dumps(result, indent=2))
        else:
            _print_pretty(result)

    elif args.demo:
        print(f"\nRunning demo: {args.demo} light curve")
        result = pipe.run_synthetic(args.demo, stellar_params=stellar)
        if args.json:
            for k in ["_lc", "_bls_periods", "_bls_power", "_folded"]:
                result.pop(k, None)
            print(json.dumps(result, indent=2))
        else:
            _print_pretty(result)

    elif args.batch:
        if not os.path.isdir(args.batch):
            print(f"Error: Directory not found: {args.batch}", file=sys.stderr)
            sys.exit(1)

        fits_files = [
            os.path.join(args.batch, f)
            for f in os.listdir(args.batch)
            if f.lower().endswith((".fits", ".fit"))
        ]

        if not fits_files:
            print(f"No FITS files found in: {args.batch}", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(fits_files)} FITS files")
        output_csv = args.output or "results/batch_results.csv"
        df = pipe.run_batch(fits_files, output_csv=output_csv, stellar_params=stellar)
        print(f"\nResults saved to: {output_csv}")
        print(df[["target", "classification", "confidence", "period_days", "snr"]].to_string())


def _print_pretty(result: dict):
    """Print a human-readable summary."""
    print("\n" + "━"*52)
    print(f"  {'AI EXOPLANET DETECTION RESULT':^48}")
    print("━"*52)
    print(f"  Target          : {result['target']}")
    print(f"  Sector          : {result['sector']}")
    print(f"  Classification  : {result['classification']}")
    print(f"  Confidence      : {result['confidence']*100:.1f}%")
    print(f"  Classifier      : {result['classifier_method']}")
    print("─"*52)
    print(f"  BLS Period      : {result['bls']['period_days']} days")
    print(f"  Epoch (T0)      : {result['bls']['t0_bjd']} BJD")
    print(f"  Transit Depth   : {result['parameters']['depth_ppm']} ppm ({result['parameters']['depth_percent']}%)")
    print(f"  Duration        : {result['bls']['duration_hours']} hours")
    print(f"  BLS SNR         : {result['bls']['snr']}")
    print("─"*52)
    p = result["parameters"]
    if p.get("r_planet_rearth"):
        print(f"  Planet Radius   : {p['r_planet_rearth']} R⊕  ({p['r_planet_rjup']} R♃)")
    print(f"  Semi-major axis : {p['semi_major_axis_au']} AU")
    print(f"  Equilibrium T   : {p['equilibrium_temp_k']} K")
    print(f"  Insolation      : {p['insolation_earth_units']} S⊕")
    print(f"  Habitable zone  : {'✓ YES' if p['in_habitable_zone'] else '✗ No'}")
    if p.get("planet_category"):
        print(f"  Planet type     : {p['planet_category']}")
    print("─"*52)
    print("  Class probabilities:")
    for label, prob in sorted(result["probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {label:<22} {bar:<20} {prob*100:.1f}%")
    print("━"*52 + "\n")


if __name__ == "__main__":
    main()
