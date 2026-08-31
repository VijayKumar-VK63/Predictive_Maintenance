"""Streamlit dashboard for Predictive Maintenance Analytics."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path

from src.data import load_data, get_feature_target, get_feature_info
from src.predict import Predictor

st.set_page_config(
    page_title="Predictive Maintenance Analytics",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
)

RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "models"


@st.cache_data
def load_dataset():
    return load_data()


@st.cache_data
def load_results():
    """Load training results and metrics."""
    results = {}
    if (RESULTS_DIR / "model_comparison.csv").exists():
        results["comparison"] = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    if (RESULTS_DIR / "test_metrics.json").exists():
        with open(RESULTS_DIR / "test_metrics.json") as f:
            results["test_metrics"] = json.load(f)
    if (RESULTS_DIR / "threshold_analysis.csv").exists():
        results["threshold_analysis"] = pd.read_csv(RESULTS_DIR / "threshold_analysis.csv")
    if (RESULTS_DIR / "feature_importance.csv").exists():
        results["feature_importance"] = pd.read_csv(RESULTS_DIR / "feature_importance.csv")
    if (RESULTS_DIR / "training_summary.json").exists():
        with open(RESULTS_DIR / "training_summary.json") as f:
            results["summary"] = json.load(f)
    return results


@st.cache_resource
def get_predictor():
    try:
        return Predictor()
    except FileNotFoundError:
        return None


def render_kpi_cards(results):
    """Render KPI cards at the top of dashboard."""
    summary = results.get("summary", {})
    test_metrics = results.get("test_metrics", {})

    col1, col2, col3, col4, col5, col6 = st.columns(6)

    with col1:
        st.metric("Total Records", f"{summary.get('dataset_shape', [0])[0]:,}")
    with col2:
        fail_count = summary.get("target_distribution", {}).get(1, 0)
        st.metric("Failure Cases", f"{fail_count:,}")
    with col3:
        fail_rate = summary.get("target_distribution", {}).get(1, 0) / summary.get("dataset_shape", [1])[0]
        st.metric("Failure Rate", f"{fail_rate:.2%}")
    with col4:
        st.metric("Best Model", summary.get("best_model", "N/A"))
    with col5:
        st.metric("Test Recall", f"{test_metrics.get('recall', 0):.3f}")
    with col6:
        st.metric("Test F1", f"{test_metrics.get('f1', 0):.3f}")


def page_overview(df, results):
    st.header("📊 Dataset Overview")

    render_kpi_cards(results)

    st.subheader("Failure vs Non-Failure")
    fail_counts = df["Machine failure"].value_counts().sort_index()
    fig = px.bar(
        x=["Normal", "Failure"],
        y=fail_counts.values,
        labels={"x": "Class", "y": "Count"},
        color=["Normal", "Failure"],
        color_discrete_map={"Normal": "#2ecc71", "Failure": "#e74c3c"},
    )
    fig.update_layout(showlegend=False, height=350)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Machine Type Distribution")
        type_counts = df["Type"].value_counts().sort_index()
        fig = px.pie(
            values=type_counts.values,
            names=type_counts.index,
            color=type_counts.index,
            color_discrete_map={"L": "#3498db", "M": "#f39c12", "H": "#9b59b6"},
        )
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Failure Rate by Machine Type")
        fail_by_type = df.groupby("Type")["Machine failure"].mean().reset_index()
        fig = px.bar(
            fail_by_type,
            x="Type",
            y="Machine failure",
            color="Type",
            color_discrete_map={"L": "#3498db", "M": "#f39c12", "H": "#9b59b6"},
            labels={"Machine failure": "Failure Rate"},
        )
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Distributions")
    numeric_features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    selected_feature = st.selectbox("Select feature", numeric_features)

    fig = px.histogram(
        df,
        x=selected_feature,
        color="Machine failure",
        nbins=50,
        barmode="overlay",
        opacity=0.7,
        color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
        labels={"Machine failure": "Failure (0=Normal, 1=Failure)"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr_cols = numeric_features + ["Machine failure"]
    corr_matrix = df[corr_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        aspect="auto",
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)


def page_failure_analysis(df):
    st.header("🔍 Failure Analysis")

    st.subheader("Failure Modes Breakdown")
    failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
    mode_counts = df[df["Machine failure"] == 1][failure_modes].sum().sort_values(ascending=True)

    fig = px.bar(
        x=mode_counts.values,
        y=mode_counts.index,
        orientation="h",
        labels={"x": "Count", "y": "Failure Mode"},
        color=mode_counts.values,
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=350, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Failure Mode by Machine Type")
    mode_by_type = df[df["Machine failure"] == 1].groupby("Type")[failure_modes].sum().T
    fig = px.imshow(
        mode_by_type,
        text_auto=True,
        color_continuous_scale="Reds",
        aspect="auto",
        labels={"x": "Machine Type", "y": "Failure Mode", "color": "Count"},
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Operating Conditions at Failure")
    fail_df = df[df["Machine failure"] == 1]
    normal_df = df[df["Machine failure"] == 0]

    features = [
        "Air temperature [K]",
        "Process temperature [K]",
        "Rotational speed [rpm]",
        "Torque [Nm]",
        "Tool wear [min]",
    ]

    selected = st.multiselect("Select features to compare", features, default=features[:3])

    for feat in selected:
        fig = go.Figure()
        fig.add_trace(go.Box(y=normal_df[feat], name="Normal", marker_color="#2ecc71", boxpoints=False))
        fig.add_trace(go.Box(y=fail_df[feat], name="Failure", marker_color="#e74c3c", boxpoints=False))
        fig.update_layout(title=feat, height=300, showlegend=True)
        st.plotly_chart(fig, use_container_width=True)


def page_model_performance(results):
    st.header("📈 Model Performance")

    comparison = results.get("comparison")
    if comparison is None:
        st.warning("No model comparison data found. Run training first.")
        return

    st.subheader("Model Comparison Table")
    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
            }
        ).background_gradient(cmap="RdYlGn", subset=["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]),
        use_container_width=True,
    )

    st.subheader("Metric Comparison")
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"]
    fig = go.Figure()
    for metric in metrics:
        fig.add_trace(go.Bar(name=metric, x=comparison["Model"], y=comparison[metric]))
    fig.update_layout(barmode="group", height=450, yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    summary = results.get("summary", {})
    best_model = summary.get("best_model", "Random Forest")

    st.subheader(f"Best Model: {best_model} - Detailed Analysis")

    threshold_data = results.get("threshold_analysis")
    if threshold_data is not None:
        st.subheader("Threshold Analysis")

        col1, col2 = st.columns(2)

        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=threshold_data["threshold"], y=threshold_data["precision"], name="Precision"))
            fig.add_trace(go.Scatter(x=threshold_data["threshold"], y=threshold_data["recall"], name="Recall"))
            fig.add_trace(go.Scatter(x=threshold_data["threshold"], y=threshold_data["f1"], name="F1"))
            fig.update_layout(
                title="Metrics vs Threshold",
                xaxis_title="Threshold",
                yaxis_title="Score",
                height=400,
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=threshold_data["threshold"],
                y=threshold_data["predicted_failures"],
                name="Predicted Failures",
                fill="tozeroy",
            ))
            fig.update_layout(
                title="Predicted Failures vs Threshold",
                xaxis_title="Threshold",
                yaxis_title="Count",
                height=400,
            )
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Precision-Recall Tradeoff")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=threshold_data["recall"],
            y=threshold_data["precision"],
            mode="lines+markers",
            name="PR Curve",
            marker=dict(size=4, color=threshold_data["threshold"], colorscale="Viridis", showscale=True),
        ))
        fig.update_layout(
            xaxis_title="Recall",
            yaxis_title="Precision",
            height=400,
        )
        st.plotly_chart(fig, use_container_width=True)

        # Threshold selector
        st.subheader("Threshold Selector")
        selected_threshold = st.slider(
            "Classification Threshold",
            min_value=0.01,
            max_value=0.99,
            value=0.5,
            step=0.01,
            help="Lower threshold → catch more failures, more false alarms. Higher threshold → fewer false alarms, miss more failures.",
        )

        row = threshold_data.iloc[(threshold_data["threshold"] - selected_threshold).abs().argsort()[:1]]
        if not row.empty:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{row['precision'].values[0]:.3f}")
            with col2:
                st.metric("Recall", f"{row['recall'].values[0]:.3f}")
            with col3:
                st.metric("F1", f"{row['f1'].values[0]:.3f}")
            with col4:
                st.metric("Predicted Failures", f"{int(row['predicted_failures'].values[0])}")

        best_f1 = summary.get("selected_threshold_f1", 0.5)
        best_recall = summary.get("selected_threshold_recall_80", 0.5)
        st.info(f"""
        **Threshold Recommendations:**
        - **F1-optimal threshold**: {best_f1:.3f} (balances precision & recall)
        - **High-recall threshold (≥80% recall)**: {best_recall:.3f} (catches more failures, more false alarms)
        - **Default 0.5**: Conservative, may miss failures

        For predictive maintenance, we recommend the **high-recall threshold** because missing a failure
        is more costly than investigating a false alarm.
        """)


def page_feature_importance(results):
    st.header("🎯 Feature Importance")

    fi_df = results.get("feature_importance")
    if fi_df is None:
        st.warning("No feature importance data found. Run training first.")
        return

    if "importance" in fi_df.columns:
        value_col = "importance"
        title = "Feature Importances (Tree-based Model)"
    elif "coefficient" in fi_df.columns:
        value_col = "coefficient"
        title = "Model Coefficients (Logistic Regression)"
    else:
        st.error("Unexpected feature importance format")
        return

    top_n = st.slider("Number of features to show", 5, min(20, len(fi_df)), 15)
    plot_df = fi_df.head(top_n).sort_values(value_col, ascending=True)

    fig = px.bar(
        plot_df,
        x=value_col,
        y="feature",
        orientation="h",
        color=value_col,
        color_continuous_scale="RdBu" if value_col == "coefficient" else "Blues",
        labels={value_col: "Importance" if value_col == "importance" else "Coefficient", "feature": "Feature"},
        title=title,
    )
    fig.update_layout(height=500, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Feature Importance Table")
    st.dataframe(
        fi_df.style.format({value_col: "{:.6f}"}).background_gradient(cmap="Blues", subset=[value_col]),
        use_container_width=True,
    )


def page_predict(predictor):
    st.header("🔮 Predict Machine Failure")

    if predictor is None:
        st.error("Model not found. Please run `python src/train.py` first to train the model.")
        return

    st.markdown("Enter machine operating conditions to predict failure probability.")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"], help="L=Low, M=Medium, H=High")
        air_temp = st.number_input("Air Temperature [K]", min_value=290.0, max_value=310.0, value=300.0, step=0.1)
        process_temp = st.number_input("Process Temperature [K]", min_value=300.0, max_value=320.0, value=310.0, step=0.1)

    with col2:
        rotational_speed = st.number_input("Rotational Speed [rpm]", min_value=1000, max_value=3000, value=1500, step=10)
        torque = st.number_input("Torque [Nm]", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        tool_wear = st.number_input("Tool Wear [min]", min_value=0, max_value=300, value=100, step=1)

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.5,
        step=0.01,
        help="Lower = more sensitive (catch more failures, more false alarms). Higher = more conservative.",
    )

    if st.button("Predict", type="primary"):
        result = predictor.predict_single(
            machine_type=machine_type,
            air_temp=air_temp,
            process_temp=process_temp,
            rotational_speed=rotational_speed,
            torque=torque,
            tool_wear=tool_wear,
            threshold=threshold,
        )

        st.subheader("Prediction Result")

        prob = result["failure_probability"]
        pred = result["prediction"]
        risk = result["risk_category"]

        if pred == "Potential Failure":
            st.error(f"⚠️ **{pred}**")
        else:
            st.success(f"✅ **{pred}**")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Failure Probability", f"{prob:.1%}")
        with col2:
            st.metric("Threshold Used", f"{threshold:.2f}")
        with col3:
            if risk == "High Risk":
                st.metric("Risk Category", risk, delta="High", delta_color="inverse")
            elif risk == "Medium Risk":
                st.metric("Risk Category", risk, delta="Medium", delta_color="off")
            else:
                st.metric("Risk Category", risk, delta="Low", delta_color="normal")

        # Gauge chart
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=prob * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Failure Probability (%)"},
            delta={"reference": threshold * 100, "relative": False},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if prob > threshold else "darkgreen"},
                "steps": [
                    {"range": [0, 30], "color": "#2ecc71"},
                    {"range": [30, 70], "color": "#f39c12"},
                    {"range": [70, 100], "color": "#e74c3c"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": threshold * 100,
                },
            },
        ))
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "**Disclaimer:** This prediction is based on a statistical model trained on synthetic data. "
            "It should not be used as the sole basis for maintenance decisions. Always consult "
            "domain experts and consider additional factors."
        )


def page_batch_predict(predictor):
    st.header("📁 Batch Prediction")

    if predictor is None:
        st.error("Model not found. Please run `python src/train.py` first to train the model.")
        return

    st.markdown("Upload a CSV file with machine measurements to get batch predictions.")

    st.subheader("Required Columns")
    st.code("Type, Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min]")

    uploaded_file = st.file_uploader("Choose CSV file", type="csv")

    threshold = st.slider(
        "Classification Threshold",
        min_value=0.01,
        max_value=0.99,
        value=0.5,
        step=0.01,
        key="batch_threshold",
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head(), use_container_width=True)

            if st.button("Run Predictions", type="primary"):
                with st.spinner("Making predictions..."):
                    results_df = predictor.predict_batch(df, threshold=threshold)

                st.success(f"Predictions complete! {results_df['predicted_failure'].sum()} potential failures detected.")

                st.subheader("Results Preview")
                st.dataframe(results_df.head(20), use_container_width=True)

                # Summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(results_df))
                with col2:
                    st.metric("Predicted Failures", int(results_df["predicted_failure"].sum()))
                with col3:
                    st.metric("Failure Rate", f"{results_df['predicted_failure'].mean():.2%}")

                # Download button
                csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {e}")


def page_documentation():
    st.header("📚 Documentation")

    with open("README.md", "r") as f:
        readme_content = f.read()

    st.markdown(readme_content)


def main():
    st.sidebar.title("🔧 Predictive Maintenance")
    st.sidebar.caption("AI4I2020 Machine Failure Prediction")

    # Load data and results
    df = load_dataset()
    results = load_results()
    predictor = get_predictor()

    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        [
            "📊 Dataset Overview",
            "🔍 Failure Analysis",
            "📈 Model Performance",
            "🎯 Feature Importance",
            "🔮 Predict Machine Failure",
            "📁 Batch Prediction",
            "📚 Documentation",
        ],
    )

    if page == "📊 Dataset Overview":
        page_overview(df, results)
    elif page == "🔍 Failure Analysis":
        page_failure_analysis(df)
    elif page == "📈 Model Performance":
        page_model_performance(results)
    elif page == "🎯 Feature Importance":
        page_feature_importance(results)
    elif page == "🔮 Predict Machine Failure":
        page_predict(predictor)
    elif page == "📁 Batch Prediction":
        page_batch_predict(predictor)
    elif page == "📚 Documentation":
        page_documentation()

    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project Info")
    summary = results.get("summary", {})
    if summary:
        st.sidebar.write(f"**Best Model:** {summary.get('best_model', 'N/A')}")
        st.sidebar.write(f"**Test Recall:** {summary.get('test_metrics', {}).get('recall', 0):.3f}")
        st.sidebar.write(f"**Test F1:** {summary.get('test_metrics', {}).get('f1', 0):.3f}")
        st.sidebar.write(f"**ROC-AUC:** {summary.get('test_metrics', {}).get('roc_auc', 0):.3f}")

    st.sidebar.markdown("---")
    st.sidebar.caption("Built with Streamlit, scikit-learn, Plotly")


if __name__ == "__main__":
    main()