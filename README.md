# 🪐 AI-enabled Detection of Exoplanets from Noisy Astronomical Light Curves

**ISRO BAH 2026 — CosmicAI Team**

A complete end-to-end pipeline that automatically detects, classifies, and characterises
exoplanet transit signals from TESS light curves using BLS + CNN.

---

## Pipeline Architecture

```
TESS Light Curves
      │
      ▼
Data Ingestion (lightkurve / FITS)
      │
      ▼
Preprocessing (Savitzky-Golay detrending, sigma-clipping)
      │
      ▼
BLS Transit Detection (astropy BoxLeastSquares)
      │
      ▼
Feature Extraction (shape, BLS, LC statistics)
      │
      ▼
CNN Classification (1D-CNN + Dense hybrid)
      │
      ▼
Parameter Estimation (Rp, a, Teq, habitability)
      │
      ▼
Streamlit Dashboard / CSV Export
```

### Classification Classes
| Class | Description |
|---|---|
| Exoplanet Transit | U-shaped, symmetric, small depth periodic dip |
| Eclipsing Binary | Deep dip with secondary eclipse, V-shaped |
| Stellar Blend | Diluted background eclipsing binary |
| Stellar Activity | Spot modulation, rotation, flares |
| Noise / Instrument | Detector artefacts, scattered light |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/your-team/exoplanet-detection
cd exoplanet-detection
pip install -r requirements.txt
```

### 2. Run the Streamlit dashboard

```bash
streamlit run app.py
```
Open your browser at **http://localhost:8501**

---

## All Run Commands

### Streamlit Web App (recommended)
```bash
# Start the interactive dashboard
streamlit run app.py

# Custom port
streamlit run app.py --server.port 8080
```

### Command-line Interface

```bash
# Analyse a demo transit signal
python detect.py --demo transit

# Analyse an eclipsing binary demo
python detect.py --demo binary

# Analyse a FITS file
python detect.py --fits path/to/tess_lightcurve.fits

# With custom stellar parameters
python detect.py --demo transit --r-star 0.8 --m-star 0.85 --t-star 5200 --l-star 0.4

# Output as JSON
python detect.py --demo transit --json

# Batch mode: analyse a directory of FITS files
python detect.py --batch path/to/fits_directory/ --output results/batch.csv

# Use trained CNN model
python detect.py --demo transit --model models/exoplanet_cnn.keras
```

### Train the CNN model

```bash
# Quick training (1000 synthetic samples, 50 epochs)
python train_model.py

# Full training (5000 samples, 100 epochs)
python train_model.py --samples 5000 --epochs 100

# Custom output path
python train_model.py --samples 2000 --epochs 75 --output models/my_model.keras

# Then use with detect.py
python detect.py --demo transit --model models/exoplanet_cnn.keras
```

### Run tests

```bash
# All tests
python -m pytest tests/ -v

# Specific test
python -m pytest tests/test_pipeline.py::test_full_pipeline_synthetic -v

# With coverage
pip install pytest-cov
python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### Docker

```bash
# Build and run with Docker Compose (recommended)
docker-compose up --build

# Or manually
docker build -t exoplanet-detection .
docker run -p 8501:8501 exoplanet-detection

# With volume mounts to persist models/results
docker run -p 8501:8501 \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/results:/app/results \
  exoplanet-detection
```

### Python API

```python
from src.pipeline import ExoplanetPipeline

# Initialise (with optional CNN model)
pipe = ExoplanetPipeline(model_path="models/exoplanet_cnn.keras")

# Run on a synthetic demo
result = pipe.run_synthetic("transit")
print(result["classification"])        # "Exoplanet Transit"
print(result["confidence"])            # 0.92
print(result["parameters"]["r_planet_rearth"])  # 11.8

# Run on a real FITS file
result = pipe.run_fits("tess_lc.fits", stellar_params={
    "r_star_rsun": 0.8,
    "m_star_msun": 0.85,
    "t_star_k": 5200,
    "l_star_lsun": 0.4,
})

# Batch processing
import glob
files = glob.glob("data/*.fits")
df = pipe.run_batch(files, output_csv="results/batch_results.csv")

# Download real TESS data
from src.ingestion import download_tess_lc
lc = download_tess_lc("TOI-700", sector=11)
result = pipe.run(lc)
```

---

## Project Structure

```
exoplanet_detection/
├── app.py                  # Streamlit dashboard
├── detect.py               # CLI entry point
├── train_model.py          # CNN training script
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── README.md
├── src/
│   ├── __init__.py
│   ├── ingestion.py        # TESS/FITS data loading
│   ├── preprocessing.py    # Detrending, sigma-clipping
│   ├── bls_detection.py    # Box Least Squares transit search
│   ├── feature_extraction.py  # Shape, BLS, LC features
│   ├── classifier.py       # CNN model + rule-based fallback
│   ├── parameters.py       # Planet parameter estimation
│   └── pipeline.py         # End-to-end orchestrator
├── models/                 # Trained model weights (created after training)
├── results/                # Batch output CSVs
├── data/                   # Place FITS files here for batch mode
└── tests/
    └── test_pipeline.py
```

---

## Features

- **Automatic TESS ingestion** via `lightkurve` (MAST archive) or local FITS files
- **Robust detrending**: Savitzky-Golay or spline-based trend removal
- **BLS periodogram**: `astropy` Box Least Squares with SNR estimation
- **Dual-input CNN**: 1D-CNN on phase-folded transit + dense branch on 20+ engineered features
- **Rule-based fallback**: works without a trained model
- **Full parameter estimation**: Rp (R⊕/R♃), semi-major axis, Teq, insolation, habitability
- **Interactive Streamlit dashboard** with Plotly charts
- **Batch processing** of 20,000+ light curves with CSV export
- **Docker support** for containerised deployment

---

## Technologies

| Category | Library |
|---|---|
| Language | Python 3.11 |
| ML | TensorFlow, Keras, Scikit-Learn |
| Astronomy | Lightkurve, Astropy, Astroquery |
| Signal processing | BLS (astropy), SciPy |
| Visualisation | Plotly, Matplotlib, Streamlit |
| Data | NumPy, Pandas |
| Deployment | Docker, Docker Compose |

---

## Estimated Cost

| Item | Cost |
|---|---|
| Software / libraries | ₹0 (open source) |
| TESS dataset | ₹0 (public MAST archive) |
| Cloud compute (optional) | ₹0 – ₹3,000 |
| **Total** | **₹0 – ₹3,000** |
