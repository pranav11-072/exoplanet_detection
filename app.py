"""
app.py — Streamlit Dashboard for AI Exoplanet Detection System
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import tempfile
import os
import json
import io

# Page config
st.set_page_config(
    page_title="AI Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stSidebar"] { background: #0d1117; }
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }
.metric-card {
    background: #161b22; border-radius: 10px; padding: 16px 20px;
    border: 1px solid #30363d; margin-bottom: 10px;
}
.metric-title { font-size: 12px; color: #8b949e; margin-bottom: 4px; }
.metric-value { font-size: 24px; font-weight: 600; color: #f0f6fc; }
.badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 13px; font-weight: 600; margin-right: 6px;
}
.badge-transit { background: #1a4731; color: #3fb950; }
.badge-binary  { background: #3d1f00; color: #f78166; }
.badge-blend   { background: #271550; color: #d2a8ff; }
.badge-activity{ background: #3d2900; color: #e3b341; }
.badge-noise   { background: #21262d; color: #8b949e; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ──────────────────────────────────────────────────────────────────
BADGE_CLASS = {
    "Exoplanet Transit": "badge-transit",
    "Eclipsing Binary": "badge-binary",
    "Stellar Blend": "badge-blend",
    "Stellar Activity": "badge-activity",
    "Noise": "badge-noise",
}

COLOR_MAP = {
    "Exoplanet Transit": "#3fb950",
    "Eclipsing Binary": "#f78166",
    "Stellar Blend": "#d2a8ff",
    "Stellar Activity": "#e3b341",
    "Noise": "#8b949e",
}


@st.cache_resource
def get_pipeline():
    from src.pipeline import ExoplanetPipeline
    return ExoplanetPipeline()


def run_analysis(lc_dict, stellar_params):
    pipe = get_pipeline()
    return pipe.run(lc_dict, stellar_params)


def plot_lightcurve(time, flux, title="Light Curve"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time, y=flux, mode="markers",
        marker=dict(size=2, color="#388bfd", opacity=0.7),
        name="Flux"
    ))
    fig.update_layout(
        title=title, height=280,
        xaxis_title="Time (days)", yaxis_title="Relative Flux",
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_bls(periods, power, best_period):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=periods, y=power, mode="lines",
        line=dict(color="#e3b341", width=1), name="BLS power"
    ))
    fig.add_vline(x=best_period, line_color="#3fb950", line_dash="dash",
                  annotation_text=f"P={best_period:.3f}d", annotation_font_color="#3fb950")
    fig.update_layout(
        title="BLS Periodogram", height=220,
        xaxis_title="Period (days)", yaxis_title="BLS Power",
        xaxis_type="log",
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_folded(phase, flux, bin_phase, bin_flux, period, depth):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=phase, y=flux, mode="markers",
        marker=dict(size=2, color="#58a6ff", opacity=0.4), name="Data"
    ))
    # Binned
    valid = [i for i, v in enumerate(bin_flux) if v is not None]
    bp = [bin_phase[i] for i in valid]
    bf = [bin_flux[i] for i in valid]
    fig.add_trace(go.Scatter(
        x=bp, y=bf, mode="lines",
        line=dict(color="#3fb950", width=2), name="Binned"
    ))
    fig.update_layout(
        title=f"Phase-folded Transit (P = {period:.3f} d)", height=240,
        xaxis_title="Phase", yaxis_title="Flux",
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        margin=dict(l=50, r=20, t=40, b=40),
    )
    return fig


def plot_probabilities(probs):
    labels = list(probs.keys())
    values = [v * 100 for v in probs.values()]
    colors = [COLOR_MAP.get(l, "#8b949e") for l in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside"
    ))
    fig.update_layout(
        height=200, xaxis_title="Probability (%)",
        plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
        font=dict(color="#c9d1d9"),
        margin=dict(l=140, r=60, t=20, b=30),
        showlegend=False, xaxis_range=[0, 115]
    )
    return fig


def badge_html(cls):
    c = BADGE_CLASS.get(cls, "badge-noise")
    return f'<span class="badge {c}">{cls}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🪐 AI Exoplanet Detector")
    st.markdown("---")
    st.markdown("### Input")
    input_mode = st.radio("Data source", ["Demo (synthetic)", "Upload FITS file"])

    demo_type = None
    uploaded_file = None

    if input_mode == "Demo (synthetic)":
        demo_type = st.selectbox("Demo signal type", [
            "transit", "binary", "stellar_activity", "blend", "noise"
        ])
    else:
        uploaded_file = st.file_uploader("Upload FITS light curve", type=["fits", "fit"])

    st.markdown("---")
    st.markdown("### Stellar parameters")
    r_star = st.number_input("Radius (R☉)", 0.1, 10.0, 1.0, 0.05)
    m_star = st.number_input("Mass (M☉)", 0.1, 10.0, 1.0, 0.05)
    t_star = st.number_input("Teff (K)", 2500, 30000, 5778, 100)
    l_star = st.number_input("Luminosity (L☉)", 0.001, 100.0, 1.0, 0.1)

    st.markdown("---")
    analyse_btn = st.button("🔭 Analyse", use_container_width=True, type="primary")


# ── Main ──────────────────────────────────────────────────────────────────────
st.title("AI-enabled Detection of Exoplanets from TESS Light Curves")
st.caption("BLS transit detection + CNN classification pipeline | ISRO BAH 2026")

if not analyse_btn:
    # Landing info
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Pipeline stages", "6")
    col2.metric("Classes detected", "5")
    col3.metric("Classifier", "CNN + Rule-based")
    col4.metric("Data source", "TESS / MAST")

    st.markdown("---")
    st.markdown("""
    ### How to use
    1. Choose a **demo signal type** or **upload a FITS file** in the sidebar
    2. Optionally set the **stellar parameters** for your target star
    3. Click **Analyse** — the full BLS + CNN pipeline will run
    4. View the light curve, BLS periodogram, phase-folded transit, and AI classification
    5. **Export** results as CSV

    ### Pipeline
    `TESS Light Curve → Ingest → Preprocess → BLS Detection → Feature Extraction → CNN Classify → Parameter Estimation`

    ### Signal classes
    | Class | Description |
    |---|---|
    | 🟢 Exoplanet Transit | U-shaped, symmetric, small depth |
    | 🔴 Eclipsing Binary | Deep dip, secondary eclipse, V-shaped |
    | 🟣 Stellar Blend | Diluted background EB |
    | 🟡 Stellar Activity | Spot modulation, flares |
    | ⬜ Noise | Instrumental artefacts |
    """)
    st.stop()


# Run analysis
stellar_params = {
    "r_star_rsun": r_star, "m_star_msun": m_star,
    "t_star_k": t_star, "l_star_lsun": l_star
}

with st.spinner("Running detection pipeline…"):
    try:
        if input_mode == "Demo (synthetic)":
            from src.ingestion import generate_synthetic_lc
            from src.preprocessing import preprocess
            lc_dict = generate_synthetic_lc(demo_type)
            result = run_analysis(lc_dict, stellar_params)
        else:
            if uploaded_file is None:
                st.error("Please upload a FITS file or switch to demo mode")
                st.stop()
            with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            from src.ingestion import load_fits_file
            lc_dict = load_fits_file(tmp_path)
            result = run_analysis(lc_dict, stellar_params)
            os.unlink(tmp_path)
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        st.exception(e)
        st.stop()

# ── Results ───────────────────────────────────────────────────────────────────
cls = result["classification"]
conf = result["confidence"]

# Header row
col_l, col_r = st.columns([2, 1])
with col_l:
    st.markdown(f"## {badge_html(cls)}", unsafe_allow_html=True)
    st.markdown(f"**Confidence:** {conf*100:.1f}%  •  "
                f"**Target:** {result['target']}  •  "
                f"**Sector:** {result['sector']}  •  "
                f"**Points:** {result['n_points']:,}")

with col_r:
    # Export button
    export_dict = {k: v for k, v in result.items()
                   if not k.startswith("_")}
    df_export = pd.DataFrame([{
        "target": result["target"],
        "classification": cls,
        "confidence": conf,
        **result["bls"],
        **{k: v for k, v in result["parameters"].items() if not isinstance(v, bool)},
        "in_habitable_zone": result["parameters"].get("in_habitable_zone"),
    }])
    csv_bytes = df_export.to_csv(index=False).encode()
    st.download_button("📥 Export CSV", csv_bytes, f"{result['target']}_result.csv",
                       "text/csv", use_container_width=True)
    json_bytes = json.dumps({k: v for k, v in result.items() if not k.startswith("_")},
                             indent=2).encode()
    st.download_button("📥 Export JSON", json_bytes, f"{result['target']}_result.json",
                       "application/json", use_container_width=True)

st.markdown("---")

# ── Plots ─────────────────────────────────────────────────────────────────────
lc_data = result["_lc"]
st.plotly_chart(
    plot_lightcurve(lc_data["time"], lc_data["flux"],
                    f"Light Curve — {result['target']}"),
    use_container_width=True
)

col_bls, col_fold = st.columns(2)
with col_bls:
    st.plotly_chart(
        plot_bls(result["_bls_periods"], result["_bls_power"],
                 result["bls"]["period_days"]),
        use_container_width=True
    )
with col_fold:
    fd = result["_folded"]
    st.plotly_chart(
        plot_folded(fd["phase"], fd["flux"], fd["bin_phase"], fd["bin_flux"],
                    result["bls"]["period_days"], result["bls"]["depth"]),
        use_container_width=True
    )

st.markdown("---")

# ── Classification panel ──────────────────────────────────────────────────────
col_cl, col_pr = st.columns([1, 1])

with col_cl:
    st.subheader("Classification probabilities")
    st.plotly_chart(plot_probabilities(result["probabilities"]), use_container_width=True)

with col_pr:
    st.subheader("BLS transit signal")
    b = result["bls"]
    c1, c2 = st.columns(2)
    c1.metric("Period", f"{b['period_days']} d")
    c2.metric("Depth", f"{result['parameters']['depth_ppm']} ppm")
    c1.metric("Duration", f"{b['duration_hours']} h")
    c2.metric("BLS SNR", f"{b['snr']}")

st.markdown("---")

# ── Parameters ────────────────────────────────────────────────────────────────
st.subheader("Estimated planetary parameters")
p = result["parameters"]
cols = st.columns(4)
cols[0].metric("Planet radius", f"{p.get('r_planet_rearth', '—')} R⊕" if p.get('r_planet_rearth') else "—")
cols[1].metric("Semi-major axis", f"{p.get('semi_major_axis_au', '—')} AU")
cols[2].metric("Equilibrium T", f"{p.get('equilibrium_temp_k', '—')} K")
cols[3].metric("Insolation", f"{p.get('insolation_earth_units', '—')} S⊕")

c1, c2, c3 = st.columns(3)
c1.metric("Planet type", p.get("planet_category", "—"))
c2.metric("Habitable zone", "✓ Yes" if p.get("in_habitable_zone") else "✗ No")
c3.metric("Impact parameter", p.get("impact_parameter", "—"))

# ── Raw data table ────────────────────────────────────────────────────────────
with st.expander("Raw parameter table"):
    flat = {**result["bls"], **{k: v for k, v in p.items()}}
    st.dataframe(pd.DataFrame([flat]), use_container_width=True)
