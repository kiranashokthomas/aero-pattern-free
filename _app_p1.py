"""
AeroPattern Free — Open-Source Aircraft Maintenance Pattern Recognition
=======================================================================
Completely free for anyone to use, modify, and deploy.
No paid services required. Runs locally or on free hosting.

License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pathlib import Path
import sys
import io

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_generator import generate_sample_fleet_data, generate_maintenance_log
from utils.ml_models import (
    train_rul_model,
    train_failure_classifier,
    detect_anomalies,
    find_patterns_kmeans,
    get_feature_columns,
)
from utils.schema import STANDARD_COLUMNS, get_required_columns
from utils.data_prep import (
    suggest_column_mapping,
    suggest_sensor_mapping,
    apply_mapping,
    validate_dataframe,
    clean_dataframe,
    create_failure_imminent_from_rul,
)

# -----------------------------------------------------------------------------
# Page config & style
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AeroPattern Free — Aircraft Maintenance Pattern Recognition",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0B3D91; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Session state defaults
# -----------------------------------------------------------------------------
if "prepared_df" not in st.session_state:
    st.session_state.prepared_df = None
if "raw_df" not in st.session_state:
    st.session_state.raw_df = None
if "validation_report" not in st.session_state:
    st.session_state.validation_report = None
if "data_ready" not in st.session_state:
    st.session_state.data_ready = False

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=64)
    st.title("AeroPattern Free")
    st.caption("Open-source • MIT License • Free forever")

    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "🧹 Data Preparation",
            "📊 Explore Data",
            "🔮 Predict RUL",
            "⚠️ Failure Risk",
            "🔍 Anomaly Detection",
            "🧩 Pattern Clusters",
            "📋 Maintenance Log Demo",
            "ℹ️ About & GitHub",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")
    if st.session_state.data_ready and st.session_state.prepared_df is not None:
        df_s = st.session_state.prepared_df
        st.success("Data ready for analysis")
        st.caption(f"{len(df_s):,} rows · {df_s['engine_id'].nunique() if 'engine_id' in df_s.columns else '?'} units")
    else:
        st.warning("No prepared data yet")
        st.caption("Go to **Data Preparation** first")

# -----------------------------------------------------------------------------
# Helper: get analysis dataframe (only if prepared)
# -----------------------------------------------------------------------------
def require_prepared_data():
    if not st.session_state.data_ready or st.session_state.prepared_df is None:
        st.warning("Please prepare your data first on the **🧹 Data Preparation** page.")
        st.stop()
    return st.session_state.prepared_df


# =============================================================================
# HOME
# =============================================================================
if page == "🏠 Home":
    st.markdown('<p class="main-header">✈️ AeroPattern Free</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Pattern recognition for aircraft maintenance data — free for anyone.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Status", "100% Free")
    c2.metric("License", "MIT")
    c3.metric("Hosting", "Self-host or free cloud")

    st.markdown("---")
    st.subheader("How it works")
    st.markdown("""
    1. **Data Preparation** — Upload any CSV from your airline systems.  
       The tool helps you **map columns**, **validate**, and **clean** the data automatically.  
       No more hours of manual Excel formatting.
    2. **Explore** — See distributions and trends.
    3. **Predict** — Remaining Useful Life, failure risk, anomalies, and behavioral patterns.
    """)

    st.info("👉 Start at **🧹 Data Preparation** in the sidebar. You can use the built-in sample fleet or upload your own file.")

    st.subheader("What the models expect (after mapping)")
    st.markdown("""
    | Column | Required? | Purpose |
    |--------|-----------|---------|
    | `engine_id` | Yes | Unique unit / engine / component ID |
    | `cycle` | Yes | Operating cycle or sequential time step |
    | `sensor_1` … `sensor_N` | Recommended | Numeric sensor readings |
    | `operational_setting_1/2/3` | Optional | Flight conditions |
    | `RUL` | Optional | Remaining Useful Life (for supervised prediction) |
    | `failure_imminent` | Optional | 0/1 label (or auto-created from RUL) |
    """)

# =============================================================================
# DATA PREPARATION  (the new core step)
# =============================================================================
elif page == "🧹 Data Preparation":
    st.header("🧹 Data Preparation")
    st.markdown(
        "Upload raw data from any airline system. Map columns, validate quality, "
        "and produce a clean dataset ready for pattern recognition — without manual Excel work."
    )

    source = st.radio(
        "Data source",
        ["Use built-in sample fleet (recommended to try first)", "Upload my own CSV"],
        horizontal=True,
    )

    raw_df = None

    if source.startswith("Use built-in"):
        n_eng = st.slider("Number of sample engines", 5, 30, 15)
        if st.button("Load sample fleet", type="primary"):
            with st.spinner("Generating realistic sample data…"):
                raw_df = generate_sample_fleet_data(n_engines=n_eng, seed=42)
                st.session_state.raw_df = raw_df
                st.session_state.prepared_df = None
                st.session_state.data_ready = False
                st.session_state.validation_report = None
                st.success(f"Loaded sample fleet: {len(raw_df):,} rows, {raw_df['engine_id'].nunique()} engines")
        if st.session_state.raw_df is not None and source.startswith("Use built-in"):
            raw_df = st.session_state.raw_df
    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv", "txt"])
        if uploaded is not None:
            try:
                # Try common separators
                content = uploaded.getvalue()
                for sep in [",", ";", "\t", "|"]:
                    try:
                        raw_df = pd.read_csv(io.BytesIO(content), sep=sep)
                        if len(raw_df.columns) > 1:
                            break
                    except Exception:
                        continue
                if raw_df is None or len(raw_df.columns) <= 1:
                    raw_df = pd.read_csv(io.BytesIO(content))
                st.session_state.raw_df = raw_df
                st.session_state.prepared_df = None
                st.session_state.data_ready = False
                st.session_state.validation_report = None
                st.success(f"Uploaded: **{len(raw_df):,}** rows × **{len(raw_df.columns)}** columns")
            except Exception as e:
                st.error(f"Could not read file: {e}")
        if st.session_state.raw_df is not None and not source.startswith("Use built-in"):
            raw_df = st.session_state.raw_df

    if raw_df is None:
        st.info("Load or upload data to begin mapping and validation.")
        st.stop()

    st.markdown("---")
    st.subheader("1. Preview raw data")
    st.dataframe(raw_df.head(8), width="stretch")
    st.caption(f"Columns found: {', '.join(raw_df.columns.astype(str).tolist())}")

    # ---------- Column mapping ----------
    st.markdown("---")
    st.subheader("2. Map columns to standard schema")

    suggestions = suggest_column_mapping(raw_df.columns.tolist())
    already = [v for v in suggestions.values() if v]

    st.markdown("**Core identifiers & targets**")
    col_map = {}
    cols_ui = st.columns(2)
    std_names = list(STANDARD_COLUMNS.keys())
    for i, std in enumerate(std_names):
        with cols_ui[i % 2]:
            options = ["— not mapped —"] + raw_df.columns.tolist()
            default_idx = 0
            if suggestions.get(std) in raw_df.columns:
                default_idx = options.index(suggestions[std])
            choice = st.selectbox(
                f"{std} {'*' if STANDARD_COLUMNS[std]['required'] else ''}",
                options,
                index=default_idx,
                help=STANDARD_COLUMNS[std]["description"],
                key=f"map_{std}",
            )
            col_map[std] = None if choice == "— not mapped —" else choice

    # Sensor mapping
    st.markdown("**Sensor / measurement columns**")
    st.caption("Select which raw columns are sensors. They will be renamed sensor_1, sensor_2, …")
    sensor_candidates = suggest_sensor_mapping(raw_df.columns.tolist(), already)
    # Also offer all remaining columns
    remaining = [c for c in raw_df.columns if c not in [v for v in col_map.values() if v]]
    default_sensors = sensor_candidates[:14] if sensor_candidates else remaining[:8]

    selected_sensors = st.multiselect(
        "Raw columns to treat as sensors",
        options=remaining,
        default=[c for c in default_sensors if c in remaining],
        help="Numeric readings (temperature, pressure, vibration, RPM, etc.)",
    )

    sensor_map = {f"sensor_{i+1}": col for i, col in enumerate(selected_sensors)}

    # ---------- Apply & validate ----------
    st.markdown("---")
    st.subheader("3. Validate & clean")

    if st.button("Apply mapping, validate and clean", type="primary"):
        with st.spinner("Mapping, cleaning and validating…"):
            mapped = apply_mapping(raw_df, col_map, sensor_map)
            cleaned = clean_dataframe(mapped)
            cleaned = create_failure_imminent_from_rul(cleaned, threshold=30)
            report = validate_dataframe(cleaned)

            st.session_state.prepared_df = cleaned
            st.session_state.validation_report = report
            st.session_state.data_ready = report["status"] != "error"

    # Show report if available
    if st.session_state.validation_report is not None:
        report = st.session_state.validation_report
        status = report["status"]
        if status == "ok":
            st.success("✅ " + " | ".join(report["messages"]))
        elif status == "warning":
            st.warning("⚠️ Validation completed with warnings:")
            for m in report["messages"]:
                st.write(f"- {m}")
        else:
            st.error("❌ Validation failed — fix the issues below before analysis:")
            for m in report["messages"]:
                st.write(f"- {m}")

        stats = report.get("stats", {})
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Rows", f"{stats.get('n_rows', 0):,}")
        m2.metric("Engines / units", stats.get("n_engines", "—"))
        m3.metric("Sensors mapped", stats.get("n_sensors", 0))
        m4.metric("Status", status.upper())

        if st.session_state.prepared_df is not None:
            st.markdown("**Cleaned data preview**")
