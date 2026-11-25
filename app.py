"""Breast Cancer Subtype Explorer: MSc Bioinformatics teaching dashboard."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

OUTPUT_DIR = Path("outputs")
CLASS_ORDER: list[str] = ["Basal", "Her2", "LumA", "LumB"]

# Accessible nature-inspired palette paired consistently with text labels
SUBTYPE_COLORS: dict[str, str] = {
    "Basal": "#C2410C",  # Terracotta
    "Her2": "#B45309",   # Warm ochre
    "LumA": "#0F766E",   # Deep teal
    "LumB": "#3730A3",   # Slate indigo
}

SUBTYPE_BG_TINTS: dict[str, str] = {
    "Basal": "#FFF7ED",
    "Her2": "#FFFBEB",
    "LumA": "#F0FDFA",
    "LumB": "#EEF2FF",
}

st.set_page_config(
    page_title="Breast Cancer Subtype Explorer",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def inject_custom_css() -> None:
    """Apply restrained, scoped styling for cards, badges, and responsive layout."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
        }
        .hero-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-left: 5px solid #0F766E;
            border-radius: 12px;
            padding: 1.35rem 1.6rem;
            margin-bottom: 1.1rem;
            box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
        }
        .hero-title {
            color: #0F172A;
            font-size: 1.75rem;
            font-weight: 700;
            margin: 0 0 0.35rem 0;
            line-height: 1.25;
        }
        .hero-subtitle {
            color: #334155;
            font-size: 0.98rem;
            margin: 0 0 0.8rem 0;
            line-height: 1.45;
        }
        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
        }
        .pill-badge {
            display: inline-block;
            background: #F1F5F9;
            color: #1E293B;
            border: 1px solid #CBD5E1;
            border-radius: 999px;
            padding: 0.18rem 0.65rem;
            font-size: 0.78rem;
            font-weight: 600;
        }
        .pill-badge-teal {
            background: #CCFBF1;
            color: #115E59;
            border-color: #99F6E4;
        }
        .applied-banner {
            background: #F0FDFA;
            border: 1px solid #99F6E4;
            border-radius: 10px;
            padding: 0.65rem 1rem;
            margin-bottom: 0.9rem;
            color: #0F172A;
            font-size: 0.92rem;
        }
        .metric-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1rem 1.15rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            height: 100%;
        }
        .metric-label {
            color: #475569;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.03em;
            margin-bottom: 0.25rem;
        }
        .metric-value {
            color: #0F172A;
            font-size: 1.7rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.25rem;
        }
        .metric-sub {
            color: #475569;
            font-size: 0.82rem;
            line-height: 1.35;
        }
        .sample-card {
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 12px;
            padding: 1.15rem 1.3rem;
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
            margin-bottom: 0.9rem;
        }
        .subtype-chip {
            display: inline-block;
            padding: 0.24rem 0.72rem;
            border-radius: 999px;
            font-weight: 700;
            font-size: 0.92rem;
            border: 1.5px solid;
        }
        .review-badge {
            display: inline-block;
            background: #FEF3C7;
            color: #92400E;
            border: 1px solid #F59E0B;
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 700;
            margin-top: 0.45rem;
        }
        .ok-badge {
            display: inline-block;
            background: #F1F5F9;
            color: #334155;
            border: 1px solid #CBD5E1;
            border-radius: 999px;
            padding: 0.22rem 0.7rem;
            font-size: 0.82rem;
            font-weight: 600;
            margin-top: 0.45rem;
        }
        .edu-footer {
            margin-top: 2rem;
            padding-top: 1rem;
            border-top: 1px solid #CBD5E1;
            color: #475569;
            font-size: 0.83rem;
            line-height: 1.5;
        }
        @media (max-width: 768px) {
            .hero-title { font-size: 1.38rem; }
            .metric-value { font-size: 1.42rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load precomputed outputs only; zero model training occurs in the app."""
    with (OUTPUT_DIR / "metrics.json").open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    test_preds = pd.read_csv(OUTPUT_DIR / "test_predictions.csv")
    cv_results = pd.read_csv(OUTPUT_DIR / "cv_results.csv")
    split_manifest = pd.read_csv(OUTPUT_DIR / "split_manifest.csv")
    return metrics, test_preds, cv_results, split_manifest


def make_confusion_matrix_figure(cm: list[list[int]], display_name: str) -> go.Figure:
    """Create a labelled 4x4 Plotly heatmap for the held-out test confusion matrix."""
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=CLASS_ORDER,
            y=CLASS_ORDER,
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 15},
            colorscale=[[0.0, "#F8FAFC"], [0.5, "#5EEAD4"], [1.0, "#0F766E"]],
            showscale=True,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Confusion Matrix (Held-Out Test)", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Predicted subtype", side="bottom", tickfont=dict(size=12, color="#0F172A"), automargin=True),
        yaxis=dict(title="Recorded subtype", autorange="reversed", tickfont=dict(size=12, color="#0F172A"), automargin=True),
        margin=dict(l=20, r=20, t=50, b=35),
        height=395,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def make_roc_figure(roc_ovr: dict, display_name: str) -> go.Figure:
    """Plot one-vs-rest ROC curves in canonical subtype order."""
    fig = go.Figure()
    for cls in CLASS_ORDER:
        cls_roc = roc_ovr["per_class"][cls]
        fig.add_trace(
            go.Scatter(
                x=cls_roc["fpr"],
                y=cls_roc["tpr"],
                mode="lines",
                name=f"{cls} ({cls_roc['auc']:.3f})",
                line=dict(color=SUBTYPE_COLORS[cls], width=2.5),
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance (0.500)",
            line=dict(color="#94A3B8", width=1.5, dash="dash"),
        )
    )
    fig.update_layout(
        title=dict(
            text=f"One-vs-Rest ROC (Macro AUC = {roc_ovr['macro_auc']:.3f})",
            font=dict(size=14, color="#0F172A"),
        ),
        xaxis=dict(title=dict(text="False positive rate", standoff=10), range=[0, 1], gridcolor="#E2E8F0", automargin=True),
        yaxis=dict(title="True positive rate", range=[0, 1.02], gridcolor="#E2E8F0", automargin=True),
        legend=dict(orientation="h", yanchor="top", y=-0.24, xanchor="center", x=0.5, font=dict(size=11)),
        margin=dict(l=20, r=20, t=50, b=105),
        height=395,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def make_cohort_vs_test_chart(cohort_counts: dict[str, int], test_counts: dict[str, int]) -> go.Figure:
    """Plot full cohort vs locked held-out test subtype counts."""
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Full cohort (n = 998)",
            x=CLASS_ORDER,
            y=[cohort_counts[c] for c in CLASS_ORDER],
            marker_color="#94A3B8",
            text=[cohort_counts[c] for c in CLASS_ORDER],
            textposition="outside",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Held-out test (n = 100)",
            x=CLASS_ORDER,
            y=[test_counts[c] for c in CLASS_ORDER],
            marker_color=[SUBTYPE_COLORS[c] for c in CLASS_ORDER],
            text=[test_counts[c] for c in CLASS_ORDER],
            textposition="outside",
        )
    )
    fig.update_layout(
        barmode="group",
        title=dict(text="Subtype Counts: Cohort vs. Test Split", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Recorded PAM50 subtype", automargin=True),
        yaxis=dict(title="Primary tumour samples", range=[0, max(cohort_counts.values()) * 1.2], gridcolor="#E2E8F0", automargin=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=20, r=20, t=72, b=30),
        height=345,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def make_sample_probability_chart(sample_row: pd.Series, threshold: float) -> go.Figure:
    """Horizontal bar chart of all four saved class probabilities in canonical order."""
    probs = [float(sample_row[f"proba_{cls}"]) for cls in CLASS_ORDER]
    fig = go.Figure(
        go.Bar(
            x=probs,
            y=CLASS_ORDER,
            orientation="h",
            marker_color=[SUBTYPE_COLORS[cls] for cls in CLASS_ORDER],
            text=[f"{p:.3f}" for p in probs],
            textposition="outside",
            hovertemplate="Subtype: %{y}<br>Saved model score: %{x:.4f}<extra></extra>",
        )
    )
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="#B45309",
        annotation_text=f"Cutoff ({threshold:.2f})",
        annotation_position="top",
    )
    fig.update_layout(
        title=dict(text="Saved Model Scores by Subtype", font=dict(size=14, color="#0F172A")),
        xaxis=dict(title="Saved predict_proba score", range=[0, 1.15], gridcolor="#E2E8F0", automargin=True),
        yaxis=dict(title="Subtype", categoryorder="array", categoryarray=list(reversed(CLASS_ORDER)), automargin=True),
        margin=dict(l=20, r=25, t=60, b=25),
        height=310,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
    )
    return fig


def render_metric_card(label: str, value: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    metrics, test_preds, cv_results, split_manifest = load_artifacts()

    model_keys = list(metrics["models"].keys())
    preferred_model = metrics["preferred_model"]
    baseline_key = metrics["baseline_model"]

    if "applied_model" not in st.session_state:
        st.session_state["applied_model"] = preferred_model

    # Compact sidebar for provenance and workflow orientation
    with st.sidebar:
        st.markdown("### Study Summary")
        st.markdown(
            f"- **Cohort:** {metrics['n_cohort_total']} primary tumours (code `01`)\n"
            f"- **Predictors:** {metrics['n_features']} gene expression columns\n"
            f"- **Splits (80/10/10):** {metrics['split_counts']['train']['total']} train / "
            f"{metrics['split_counts']['val']['total']} val / {metrics['split_counts']['test']['total']} test\n"
            f"- **CV rule:** 3-fold stratified CV on training split only\n"
            f"- **CV preferred:** `{preferred_model}`"
        )
        st.caption(
            "All metrics and probabilities are loaded from precomputed artifacts in `outputs/`. "
            "No model training or tuning occurs during dashboard interaction."
        )

    # 1. Restrained hero area
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Breast Cancer Subtype Explorer</div>
            <p class="hero-subtitle">
                Can gene expression measurements distinguish the recorded Basal, Her2, LumA, and LumB labels
                in previously unseen primary tumour samples within this TCGA-BRCA teaching study?
            </p>
            <div class="badge-row">
                <span class="pill-badge pill-badge-teal">TCGA-BRCA</span>
                <span class="pill-badge">4 subtypes</span>
                <span class="pill-badge">Teaching demonstration</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Compact evaluation form with model selector and Apply button
    with st.form("model_evaluation_form", border=True):
        form_col1, form_col2 = st.columns([3, 1], vertical_alignment="bottom")
        with form_col1:
            current_index = model_keys.index(st.session_state["applied_model"])
            staged_model = st.selectbox(
                "Choose classifier to evaluate",
                options=model_keys,
                index=current_index,
                format_func=lambda k: (
                    f"{metrics['models'][k]['display_name']}"
                    + ("  [CV Preferred]" if k == preferred_model else "")
                ),
            )
        with form_col2:
            submitted = st.form_submit_button("Apply", type="primary", use_container_width=True)
        st.caption(
            "Model choice is staged inside this form: displayed results stay on the currently applied model until you press **Apply**."
        )

    if submitted:
        st.session_state["applied_model"] = staged_model

    active_model = st.session_state["applied_model"]
    active_info = metrics["models"][active_model]
    active_test = active_info["test_metrics"]
    baseline_test = metrics["models"][baseline_key]["test_metrics"]

    st.markdown(
        f"""
        <div class="applied-banner">
            <strong>Applied model:</strong> {active_info['display_name']} (<code>{active_model}</code>)
            &nbsp;&middot;&nbsp; Selected before testing? {"Yes (highest 3-fold CV balanced accuracy)" if active_model == preferred_model else f"No (CV preferred was {preferred_model})"}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 3. Four aligned metric cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    n_test = active_test["n_samples"]
    with m_col1:
        render_metric_card(
            "Test Accuracy",
            f"{active_test['accuracy']:.1%}",
            f"{active_test['correct_count']} / {n_test} correct · Baseline: {baseline_test['accuracy']:.1%} ({baseline_test['correct_count']} / {n_test})",
        )
    with m_col2:
        render_metric_card(
            "Test Balanced Accuracy",
            f"{active_test['balanced_accuracy']:.1%}",
            f"Mean of 4 subtype recalls · Baseline: {baseline_test['balanced_accuracy']:.1%}",
        )
    with m_col3:
        render_metric_card(
            "Test Macro F1",
            f"{active_test['macro_f1']:.3f}",
            f"Equal weight per subtype · Baseline: {baseline_test['macro_f1']:.3f}",
        )
    with m_col4:
        render_metric_card(
            "Held-Out Test Samples",
            f"{n_test}",
            f"10% locked split of {metrics['n_cohort_total']} cohort samples (patient-disjoint)",
        )

    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

    # 4. Two main tabs
    tab_perf, tab_sample = st.tabs(["Model performance", "Explore a sample"])

    with tab_perf:
        perf_c1, perf_c2 = st.columns(2)
        with perf_c1:
            st.plotly_chart(
                make_confusion_matrix_figure(active_test["confusion_matrix"], active_info["display_name"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Rows show the recorded subtype and columns show the predicted subtype chosen by largest `predict_proba` mapped via `classes_`."
            )
        with perf_c2:
            st.plotly_chart(
                make_roc_figure(active_test["roc_ovr"], active_info["display_name"]),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "One-vs-rest ROC curves on the 100 held-out test samples. Each curve compares one subtype against the other three combined."
            )

        st.markdown("#### Active Model vs. Majority-Class Baseline (Held-Out Test, n = 100)")
        comp_rows = [
            {
                "Metric": "Accuracy",
                active_info["display_name"]: f"{active_test['accuracy']:.3f} ({active_test['correct_count']}/{n_test})",
                "Majority Baseline": f"{baseline_test['accuracy']:.3f} ({baseline_test['correct_count']}/{n_test})",
            },
            {
                "Metric": "Balanced Accuracy",
                active_info["display_name"]: f"{active_test['balanced_accuracy']:.3f}",
                "Majority Baseline": f"{baseline_test['balanced_accuracy']:.3f}",
            },
            {
                "Metric": "Macro F1",
                active_info["display_name"]: f"{active_test['macro_f1']:.3f}",
                "Majority Baseline": f"{baseline_test['macro_f1']:.3f}",
            },
        ]
        for cls in CLASS_ORDER:
            n_cls = metrics["split_counts"]["test"]["by_class"][cls]
            comp_rows.append(
                {
                    "Metric": f"Recall — {cls} (n={n_cls})",
                    active_info["display_name"]: f"{active_test['per_class_recall'][cls]:.3f}",
                    "Majority Baseline": f"{baseline_test['per_class_recall'][cls]:.3f}",
                }
            )
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Cohort vs. Held-Out Split Counts & Pre-Test CV Comparison")
        dist_col, cv_col = st.columns([0.95, 1.15])
        with dist_col:
            test_counts = metrics["split_counts"]["test"]["by_class"]
            st.plotly_chart(
                make_cohort_vs_test_chart(metrics["cohort_counts"], test_counts),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                f"Full cohort: {metrics['n_cohort_total']} samples. Split: "
                f"{metrics['split_counts']['train']['total']} train (80%), "
                f"{metrics['split_counts']['val']['total']} validation (10%), "
                f"{metrics['split_counts']['test']['total']} locked test (10%)."
            )

        with cv_col:
            cv_summary_rows = []
            for k in model_keys:
                cv_s = metrics["models"][k]["cv_summary"]
                cv_summary_rows.append(
                    {
                        "Model": k + (" ★" if k == preferred_model else ""),
                        "CV Bal. Acc": f"{cv_s['mean_balanced_accuracy']:.3f}",
                        "CV Acc": f"{cv_s['mean_accuracy']:.3f}",
                        "CV Macro F1": f"{cv_s['mean_macro_f1']:.3f}",
                        "Fit (s)": f"{cv_s['mean_fit_time_sec']:.2f}",
                    }
                )
            st.markdown("**3-Fold Stratified CV on Training Set Only (n = 798)**")
            st.dataframe(pd.DataFrame(cv_summary_rows), use_container_width=True, hide_index=True)
            st.caption(
                f"★ Model selection used mean CV balanced accuracy on the training folds only (`{preferred_model}` selected) **before** the locked test set was evaluated."
            )

    with tab_sample:
        model_test_df = test_preds[test_preds["model_name"] == active_model].copy()
        sample_ids = model_test_df["sample"].tolist()

        ctrl_col1, ctrl_col2 = st.columns([1.2, 1.3])
        with ctrl_col1:
            selected_sample = st.selectbox(
                "Select a held-out test sample",
                options=sample_ids,
                index=0,
                help="Outside the model form: updates only this sample panel immediately.",
            )
        with ctrl_col2:
            review_threshold = st.slider(
                "Flag for review below",
                min_value=0.25,
                max_value=0.95,
                value=0.60,
                step=0.01,
                help="Exploratory display aid only (strict '<' threshold on max saved score). Does not alter predictions or metrics.",
            )
            st.caption(
                "Exploratory display aid, not a validated confidence cutoff. Triggers only when `max_probability < slider_value`."
            )

        sample_row = model_test_df[model_test_df["sample"] == selected_sample].iloc[0]
        pred_cls = str(sample_row["predicted_label"])
        actual_cls = str(sample_row["actual_label"])
        is_correct = pred_cls == actual_cls
        max_prob = max(float(sample_row[f"proba_{c}"]) for c in CLASS_ORDER)
        # Strict inequality: equality does NOT trigger the badge
        review_needed = max_prob < review_threshold

        card_col, chart_col = st.columns([1.0, 1.6])
        with card_col:
            pred_color = SUBTYPE_COLORS[pred_cls]
            pred_bg = SUBTYPE_BG_TINTS[pred_cls]
            actual_color = SUBTYPE_COLORS[actual_cls]
            actual_bg = SUBTYPE_BG_TINTS[actual_cls]
            status_text = "Correct prediction" if is_correct else "Incorrect prediction"
            status_color = "#0F766E" if is_correct else "#B91C1C"

            badge_html = (
                f'<div class="review-badge">Review suggested (max score {max_prob:.3f} &lt; {review_threshold:.2f})</div>'
                if review_needed
                else f'<div class="ok-badge">No review flag (max score {max_prob:.3f} &ge; {review_threshold:.2f})</div>'
            )

            st.markdown(
                f"""
                <div class="sample-card">
                    <div class="metric-label">Sample ID</div>
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 0.75rem;">
                        <code>{selected_sample}</code>
                    </div>
                    <div class="metric-label">Predicted Subtype ({active_model})</div>
                    <div style="margin-bottom: 0.7rem;">
                        <span class="subtype-chip" style="color: {pred_color}; background: {pred_bg}; border-color: {pred_color};">
                            {pred_cls}
                        </span>
                    </div>
                    <div class="metric-label">Recorded Subtype</div>
                    <div style="margin-bottom: 0.7rem;">
                        <span class="subtype-chip" style="color: {actual_color}; background: {actual_bg}; border-color: {actual_color};">
                            {actual_cls}
                        </span>
                    </div>
                    <div style="font-weight: 700; color: {status_color}; font-size: 0.96rem;">
                        {status_text}
                    </div>
                    {badge_html}
                </div>
                """,
                unsafe_allow_html=True,
            )

        with chart_col:
            st.plotly_chart(
                make_sample_probability_chart(sample_row, review_threshold),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.caption(
                "Note: These bars show raw saved `predict_proba` scores for the applied model, not clinically calibrated probabilities."
            )

    # 7. Educational-use footer
    runtime_meta = metrics["runtime_metadata"]
    st.markdown(
        f"""
        <div class="edu-footer">
            <strong>Educational-Use Notice:</strong> This dashboard supports an MSc Bioinformatics classroom demonstration on the
            998-sample TCGA-BRCA primary tumour teaching table (<code>data/tcga_brca_teaching.csv</code>). Upstream gene selection
            and expression normalization history are unresolved in the source file; results must not be interpreted as independent
            clinical validation or biomarker discovery. &nbsp;|&nbsp;
            Offline pipeline runtime: {runtime_meta['total_runtime_sec']:.2f}s · Warnings logged: {runtime_meta['warnings_count']}.
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
