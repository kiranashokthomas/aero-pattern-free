"""
AeroPattern Free — Open-Source Aircraft Maintenance Pattern Recognition
Completely free. MIT License.
"""

from pathlib import Path
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))

from utils.data_generator import generate_sample_fleet_data, generate_maintenance_log
from utils.data_prep import (
    apply_mapping,
    clean_dataframe,
    create_failure_imminent_from_rul,
    read_uploaded_file,
    suggest_column_mapping,
    suggest_sensor_mapping,
    validate_dataframe,
)
from utils.ml_models import (
    detect_anomalies,
    find_patterns_kmeans,
    get_feature_columns,
    train_failure_classifier,
    train_rul_model,
)
from utils.schema import STANDARD_COLUMNS

st.set_page_config(
    page_title="AeroPattern Free — Aircraft Maintenance Pattern Recognition",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #0B3D91; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.1rem; color: #555; margin-bottom: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)

for key, default in [
    ("prepared_df", None),
    ("raw_df", None),
    ("validation_report", None),
    ("data_ready", False),
    ("rul_result", None),
    ("fail_result", None),
    ("data_fingerprint", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def _fingerprint(df: pd.DataFrame) -> str:
    return f"{len(df)}|{list(df.columns)}|{df.head(3).to_json()}"


def require_prepared_data():
    if not st.session_state.data_ready or st.session_state.prepared_df is None:
        st.warning("Please prepare your data first on the **🧹 Data Preparation** page.")
        st.stop()
    return st.session_state.prepared_df


with st.sidebar:
    st.image("https://img.icons8.com/color/96/airplane-mode-on.png", width=64)
    st.title("AeroPattern Free")
    st.caption("Open-source · MIT · Free forever")

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
        n_eng = df_s["engine_id"].nunique() if "engine_id" in df_s.columns else "?"
        st.caption(f"{len(df_s):,} rows · {n_eng} units")
    else:
        st.warning("No prepared data yet")
        st.caption("Go to **Data Preparation** first")


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
    st.markdown(
        """
1. **Data Preparation** — Upload CSV or Excel from any airline system.
   Map columns automatically, validate quality, and clean — no manual Excel work.
2. **Explore** — Distributions and per-unit trends.
3. **Predict** — Remaining Useful Life, failure risk, anomalies, and behavioral patterns.
        """
    )
    st.info(
        "👉 Start at **🧹 Data Preparation**. Use the built-in sample fleet or upload your own file."
    )

    st.subheader("Expected schema (after mapping)")
    st.markdown(
        """
| Column | Required? | Purpose |
|--------|-----------|---------|
| `engine_id` | Yes | Unit / engine / component ID |
| `cycle` | Yes | Operating cycle or time step |
| `sensor_1` … `sensor_N` | Recommended | Numeric sensor readings |
| `operational_setting_1/2/3` | Optional | Flight conditions |
| `RUL` | Optional | Remaining Useful Life |
| `failure_imminent` | Optional | 0/1 (auto-created from RUL) |
        """
    )

elif page == "🧹 Data Preparation":
    st.header("🧹 Data Preparation")
    st.markdown(
        "Upload raw data from any airline system. Map columns, validate quality, "
        "and produce a clean dataset ready for pattern recognition."
    )

    source = st.radio(
        "Data source",
        ["Use built-in sample fleet (recommended to try first)", "Upload my own file"],
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
                st.session_state.rul_result = None
                st.session_state.fail_result = None
                st.success(
                    f"Loaded sample fleet: {len(raw_df):,} rows, "
                    f"{raw_df['engine_id'].nunique()} engines"
                )
        if st.session_state.raw_df is not None and source.startswith("Use built-in"):
            raw_df = st.session_state.raw_df
    else:
        uploaded = st.file_uploader(
            "Upload CSV, TSV, or Excel (.xlsx)",
            type=["csv", "txt", "tsv", "xlsx", "xls"],
        )
        if uploaded is not None:
            try:
                raw_df = read_uploaded_file(uploaded.getvalue(), uploaded.name)
                st.session_state.raw_df = raw_df
                st.session_state.prepared_df = None
                st.session_state.data_ready = False
                st.session_state.validation_report = None
                st.session_state.rul_result = None
                st.session_state.fail_result = None
                st.success(
                    f"Uploaded: **{len(raw_df):,}** rows × **{len(raw_df.columns)}** columns"
                )
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
    st.caption(f"Columns: {', '.join(map(str, raw_df.columns.tolist()))}")

    st.markdown("---")
    st.subheader("2. Map columns to standard schema")

    suggestions = suggest_column_mapping(raw_df.columns.tolist())
    already = [v for v in suggestions.values() if v]

    st.markdown("**Core identifiers & targets**")
    col_map = {}
    cols_ui = st.columns(2)
    for i, std in enumerate(STANDARD_COLUMNS.keys()):
        with cols_ui[i % 2]:
            options = ["— not mapped —"] + list(raw_df.columns)
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

    st.markdown("**Sensor / measurement columns**")
    st.caption("Select numeric sensor columns (EGT, oil pressure, vibration, RPM, etc.).")
    remaining = [c for c in raw_df.columns if c not in [v for v in col_map.values() if v]]
    sensor_candidates = suggest_sensor_mapping(raw_df.columns.tolist(), already)
    default_sensors = [c for c in sensor_candidates[:14] if c in remaining] or remaining[:8]

    selected_sensors = st.multiselect(
        "Raw columns to treat as sensors",
        options=remaining,
        default=default_sensors,
        help="Temperature, pressure, vibration, RPM, etc.",
    )
    sensor_map = {f"sensor_{i + 1}": col for i, col in enumerate(selected_sensors)}

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
            st.session_state.rul_result = None
            st.session_state.fail_result = None
            st.session_state.data_fingerprint = _fingerprint(cleaned)

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
            st.error("❌ Validation failed — fix issues before analysis:")
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
            st.dataframe(st.session_state.prepared_df.head(10), width="stretch")
            csv_buf = st.session_state.prepared_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download cleaned CSV",
                data=csv_buf,
                file_name="aeropattern_cleaned.csv",
                mime="text/csv",
            )
            if st.session_state.data_ready:
                st.success("Data is ready. Use the analysis pages in the sidebar.")
            else:
                st.error("Map required columns correctly before analysis.")

elif page == "📊 Explore Data":
    st.header("📊 Explore Fleet Data")
    df = require_prepared_data()
    st.write(f"**Shape:** {df.shape[0]:,} rows × {df.shape[1]} columns")
    if "engine_id" in df.columns:
        st.write(f"**Engines / units:** {df['engine_id'].nunique()}")
    tab1, tab2, tab3 = st.tabs(["Overview", "Sensor distributions", "Per-unit view"])
    with tab1:
        st.dataframe(df.describe(include="all").T, width="stretch")
    with tab2:
        sensors = [c for c in df.columns if c.startswith("sensor_")]
        if not sensors:
            st.info("No sensor columns available.")
        else:
            sensor_choice = st.selectbox("Sensor", sensors)
            fig = px.histogram(df, x=sensor_choice, nbins=40, title=f"Distribution of {sensor_choice}")
            st.plotly_chart(fig, width="stretch")
    with tab3:
        if "engine_id" in df.columns and "cycle" in df.columns:
            eng = st.selectbox("Unit / Engine", sorted(df["engine_id"].dropna().unique()))
            eng_df = df[df["engine_id"] == eng]
            sensors = [c for c in df.columns if c.startswith("sensor_")]
            if sensors:
                sensor = st.selectbox("Sensor to plot", sensors, key="eng_sensor")
                fig = px.line(eng_df, x="cycle", y=sensor, title=f"{eng} — {sensor} over cycles")
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("No sensors to plot.")
        else:
            st.info("Need engine_id and cycle columns.")

elif page == "🔮 Predict RUL":
    st.header("🔮 Remaining Useful Life (RUL) Prediction")
    df = require_prepared_data()
    if "RUL" not in df.columns:
        st.error("No **RUL** column. Map a remaining-life column in Data Preparation.")
        st.stop()
    fp = _fingerprint(df)
    if st.session_state.rul_result is None or st.session_state.data_fingerprint != fp:
        with st.spinner("Training Random Forest RUL model…"):
            try:
                st.session_state.rul_result = train_rul_model(df)
                st.session_state.data_fingerprint = fp
            except Exception as e:
                st.error(str(e))
                st.stop()
    result = st.session_state.rul_result
    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (cycles)", f"{metrics['mae']:.1f}")
    c2.metric("RMSE", f"{metrics['rmse']:.1f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")
    c4.metric("Test samples", metrics["n_test"])
    st.subheader("Feature Importance")
    imp = result["feature_importance"].head(15)
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", labels={"x": "Importance", "y": "Feature"}, title="Top features driving RUL prediction")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")
    st.subheader("Predicted vs Actual RUL (test set)")
    plot_df = pd.DataFrame({"Actual RUL": result["y_test"].values, "Predicted RUL": result["preds"]})
    fig2 = px.scatter(plot_df, x="Actual RUL", y="Predicted RUL", trendline="ols", title="Model performance on held-out data")
    max_v = max(plot_df["Actual RUL"].max(), plot_df["Predicted RUL"].max())
    fig2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v, line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig2, width="stretch")
    st.download_button("⬇️ Download test predictions (CSV)", data=plot_df.to_csv(index=False).encode("utf-8"), file_name="rul_predictions.csv", mime="text/csv")

elif page == "⚠️ Failure Risk":
    st.header("⚠️ Imminent Failure Risk Classification")
    df = require_prepared_data()
    if "failure_imminent" not in df.columns:
        st.error("No **failure_imminent** column. Auto-create it from RUL in Data Preparation.")
        st.stop()
    fp = _fingerprint(df)
    if st.session_state.fail_result is None or st.session_state.data_fingerprint != fp:
        with st.spinner("Training failure classifier…"):
            try:
                st.session_state.fail_result = train_failure_classifier(df)
                st.session_state.data_fingerprint = fp
            except Exception as e:
                st.error(str(e))
                st.stop()
    result = st.session_state.fail_result
    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Precision (fail)", f"{m['precision_fail']:.1%}")
    c3.metric("Recall (fail)", f"{m['recall_fail']:.1%}")
    c4.metric("F1 (fail)", f"{m['f1_fail']:.1%}")
    st.subheader("Risk probability distribution (test set)")
    proba_df = pd.DataFrame({"Probability of imminent failure": result["proba"], "Actual": result["y_test"].map({0: "Healthy", 1: "Imminent failure"})})
    fig = px.histogram(proba_df, x="Probability of imminent failure", color="Actual", barmode="overlay", nbins=30, opacity=0.7)
    st.plotly_chart(fig, width="stretch")

elif page == "🔍 Anomaly Detection":
    st.header("🔍 Anomaly Detection (Isolation Forest)")
    df = require_prepared_data()
    contamination = st.slider("Expected anomaly rate", 0.01, 0.15, 0.05, 0.01)
    try:
        with st.spinner("Running Isolation Forest…"):
            anomalous = detect_anomalies(df, contamination=contamination)
    except Exception as e:
        st.error(str(e))
        st.stop()
    n_anom = int(anomalous["is_anomaly"].sum())
    st.metric("Anomalies detected", f"{n_anom:,}  ({n_anom / len(anomalous):.1%})")
    if "engine_id" in anomalous.columns:
        top = anomalous[anomalous["is_anomaly"] == 1].groupby("engine_id").size().sort_values(ascending=False).head(10)
        if len(top):
            st.subheader("Units with most anomalous readings")
            st.bar_chart(top)
    st.subheader("Anomaly score distribution")
    fig = px.histogram(anomalous, x="anomaly_score", color="is_anomaly", color_discrete_map={0: "#2ecc71", 1: "#e74c3c"}, labels={"is_anomaly": "Anomaly"}, nbins=50)
    st.plotly_chart(fig, width="stretch")
    show_cols = [c for c in ["engine_id", "cycle", "anomaly_score"] + get_feature_columns(df)[:5] if c in anomalous.columns]
    st.dataframe(anomalous[anomalous["is_anomaly"] == 1][show_cols].sort_values("anomaly_score").head(20), width="stretch")
    st.download_button("⬇️ Download anomaly results (CSV)", data=anomalous.to_csv(index=False).encode("utf-8"), file_name="anomaly_results.csv", mime="text/csv")

elif page == "🧩 Pattern Clusters":
    st.header("🧩 Behavioral Pattern Clusters")
    df = require_prepared_data()
    n_clusters = st.slider("Number of patterns to discover", 2, 8, 4)
    try:
        with st.spinner("Clustering…"):
            clustered, kmeans = find_patterns_kmeans(df, n_clusters=n_clusters)
    except Exception as e:
        st.error(str(e))
        st.stop()
    st.subheader("Cluster sizes")
    counts = clustered["pattern_cluster"].value_counts().sort_index()
    st.bar_chart(counts)
    feats = get_feature_columns(df)
    if len(feats) >= 2:
        sensor_x = st.selectbox("X axis", feats, index=0)
        sensor_y = st.selectbox("Y axis", feats, index=min(1, len(feats) - 1))
        sample = clustered.sample(min(3000, len(clustered)), random_state=42)
        fig = px.scatter(sample, x=sensor_x, y=sensor_y, color="pattern_cluster", title="Pattern clusters (sampled)", opacity=0.6)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Need at least two feature columns to plot clusters.")

elif page == "📋 Maintenance Log Demo":
    st.header("📋 Sample Maintenance Work-Order Log")
    st.caption("Secondary demo. Main analysis uses prepared sensor/cycle data.")
    log = generate_maintenance_log(n_events=60, seed=42)
    st.dataframe(log, width="stretch")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(log["component"].value_counts().reset_index(), x="component", y="count", title="Events by component")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(log["action"].value_counts().reset_index(), x="action", y="count", title="Events by action type")
        st.plotly_chart(fig, width="stretch")

elif page == "ℹ️ About & GitHub":
    st.header("ℹ️ About AeroPattern Free")
    st.markdown(
        """
**AeroPattern Free** is an open-source pattern-recognition tool for aircraft maintenance data.

### Why Data Preparation exists
Real airline data almost never arrives ready for analysis. Different systems use different
column names, separators, and quality levels. This app maps, validates, and cleans automatically.

### Run locally
```bash
git clone https://github.com/kiranashokthomas/aero-pattern-free.git
cd aero-pattern-free
pip install -r requirements.txt
streamlit run app.py
```

### Free hosting
- Streamlit Community Cloud
- Hugging Face Spaces
- Any free-tier host or your own machine

### License
MIT — free for commercial and non-commercial use.

### Disclaimer
Educational / research tool only. **Not** certified for operational airworthiness decisions.
Always follow approved maintenance programs and regulations (FAA, EASA, etc.).
        """
    )
    st.success("Built to be free. Fork it, improve it, share it.")
