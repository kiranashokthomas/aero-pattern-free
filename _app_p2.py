            st.dataframe(st.session_state.prepared_df.head(10), width="stretch")

            # Download cleaned CSV
            csv_buf = st.session_state.prepared_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download cleaned CSV",
                data=csv_buf,
                file_name="aeropattern_cleaned.csv",
                mime="text/csv",
            )

            if st.session_state.data_ready:
                st.success("Data is ready. You can now use the analysis pages in the sidebar.")
            else:
                st.error("Data is not ready for analysis until required columns are correctly mapped and validation passes.")

# =============================================================================
# EXPLORE
# =============================================================================
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

# =============================================================================
# PREDICT RUL
# =============================================================================
elif page == "🔮 Predict RUL":
    st.header("🔮 Remaining Useful Life (RUL) Prediction")
    df = require_prepared_data()

    if "RUL" not in df.columns:
        st.error("No **RUL** column in the prepared data. Map a remaining-life column (or create one) in Data Preparation.")
        st.stop()

    try:
        with st.spinner("Training Random Forest RUL model…"):
            result = train_rul_model(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    metrics = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE (cycles)", f"{metrics['mae']:.1f}")
    c2.metric("RMSE", f"{metrics['rmse']:.1f}")
    c3.metric("R²", f"{metrics['r2']:.3f}")
    c4.metric("Test samples", metrics["n_test"])

    st.subheader("Feature Importance")
    imp = result["feature_importance"].head(15)
    fig = px.bar(x=imp.values, y=imp.index, orientation="h",
                 labels={"x": "Importance", "y": "Feature"},
                 title="Top features driving RUL prediction")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

    st.subheader("Predicted vs Actual RUL (test set)")
    plot_df = pd.DataFrame({
        "Actual RUL": result["y_test"].values,
        "Predicted RUL": result["preds"],
    })
    fig2 = px.scatter(plot_df, x="Actual RUL", y="Predicted RUL",
                      trendline="ols", title="Model performance on held-out data")
    max_v = max(plot_df["Actual RUL"].max(), plot_df["Predicted RUL"].max())
    fig2.add_shape(type="line", x0=0, y0=0, x1=max_v, y1=max_v,
                   line=dict(dash="dash", color="gray"))
    st.plotly_chart(fig2, width="stretch")

# =============================================================================
# FAILURE RISK
# =============================================================================
elif page == "⚠️ Failure Risk":
    st.header("⚠️ Imminent Failure Risk Classification")
    df = require_prepared_data()

    if "failure_imminent" not in df.columns:
        st.error("No **failure_imminent** column. In Data Preparation you can auto-create it from RUL.")
        st.stop()

    try:
        with st.spinner("Training failure classifier…"):
            result = train_failure_classifier(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    m = result["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{m['accuracy']:.1%}")
    c2.metric("Precision (fail)", f"{m['precision_fail']:.1%}")
    c3.metric("Recall (fail)", f"{m['recall_fail']:.1%}")
    c4.metric("F1 (fail)", f"{m['f1_fail']:.1%}")

    st.subheader("Risk probability distribution (test set)")
    proba_df = pd.DataFrame({
        "Probability of imminent failure": result["proba"],
        "Actual": result["y_test"].map({0: "Healthy", 1: "Imminent failure"}),
    })
    fig = px.histogram(proba_df, x="Probability of imminent failure", color="Actual",
                       barmode="overlay", nbins=30, opacity=0.7)
    st.plotly_chart(fig, width="stretch")

# =============================================================================
# ANOMALY DETECTION
# =============================================================================
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
        top = (anomalous[anomalous["is_anomaly"] == 1]
               .groupby("engine_id").size()
               .sort_values(ascending=False).head(10))
        if len(top):
            st.subheader("Units with most anomalous readings")
            st.bar_chart(top)

    st.subheader("Anomaly score distribution")
    fig = px.histogram(anomalous, x="anomaly_score", color="is_anomaly",
                       color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
                       labels={"is_anomaly": "Anomaly"}, nbins=50)
    st.plotly_chart(fig, width="stretch")

    show_cols = [c for c in ["engine_id", "cycle", "anomaly_score"] + get_feature_columns(df)[:5]
                 if c in anomalous.columns]
    st.dataframe(
        anomalous[anomalous["is_anomaly"] == 1][show_cols]
        .sort_values("anomaly_score").head(20),
        width="stretch",
    )

# =============================================================================
# PATTERN CLUSTERS
# =============================================================================
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
        fig = px.scatter(sample, x=sensor_x, y=sensor_y, color="pattern_cluster",
                         title="Pattern clusters (sampled)", opacity=0.6)
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("Need at least two feature columns to plot clusters.")

# =============================================================================
# MAINTENANCE LOG DEMO
# =============================================================================
elif page == "📋 Maintenance Log Demo":
    st.header("📋 Sample Maintenance Work-Order Log")
    st.caption("Secondary demo (work orders). Main analysis uses the prepared sensor/cycle data.")
    log = generate_maintenance_log(n_events=60, seed=42)
    st.dataframe(log, width="stretch")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(log["component"].value_counts().reset_index(),
                     x="component", y="count", title="Events by component")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(log["action"].value_counts().reset_index(),
                     x="action", y="count", title="Events by action type")
        st.plotly_chart(fig, width="stretch")

# =============================================================================
# ABOUT & GITHUB
# =============================================================================
elif page == "ℹ️ About & GitHub":
    st.header("ℹ️ About AeroPattern Free")
    st.markdown("""
    **AeroPattern Free** is an open-source pattern-recognition tool for aircraft maintenance data.

    ### Why the Data Preparation step exists
    Real airline data almost never arrives in a perfect format. Different systems use different column names,
    separators, units, and quality levels. Manually fixing this in Excel is slow and error-prone.

    This app includes a **pre-analysis pipeline** that:
    - Suggests column mappings using known aliases + fuzzy matching
    - Lets you map any CSV columns to a standard schema
    - Validates types, missing values, and basic consistency
    - Cleans and formats the data
    - Only then unlocks the analysis pages

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
    """)
    st.success("Built to be free. Fork it, improve it, share it.")
