import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
import joblib

from burnout_recommender import (
    BurnoutSuggestionRecommender,
    add_engineered_features,
    load_developer_recommender_data,
)


st.set_page_config(
    page_title="PulseWatch",
    page_icon="PW",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}

        .block-container {
            max-width: 1180px;
            padding-top: 1.25rem;
            padding-bottom: 2rem;
        }

        .app-title {
            font-size: 2.15rem;
            line-height: 1.15;
            font-weight: 800;
            color: #17202a;
            margin-bottom: 0.25rem;
        }

        .app-subtitle {
            color: #526173;
            font-size: 1rem;
            margin-bottom: 1.25rem;
        }

        .recommendation-card {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-left: 4px solid #1f4e79;
            border-radius: 8px;
            padding: 0.85rem 1rem;
            margin-bottom: 0.7rem;
        }

        .recommendation-title {
            color: #17202a;
            font-weight: 750;
            margin-bottom: 0.2rem;
        }

        .recommendation-meta {
            color: #526173;
            font-size: 0.9rem;
        }

        .cluster-box {
            background: #ffffff;
            border: 1px solid #d9e2ec;
            border-radius: 8px;
            padding: 1rem;
            min-height: 103px;
        }

        .cluster-label {
            color: #526173;
            font-size: 0.85rem;
            font-weight: 700;
            margin-bottom: 0.35rem;
        }

        .cluster-value {
            color: #17202a;
            font-size: 1.25rem;
            font-weight: 800;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("burnout_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    return model, scaler, feature_names, label_encoder


@st.cache_resource
def load_shap_explainer(_model):
    return shap.TreeExplainer(_model)


@st.cache_resource
def load_recommender(feature_names_key):
    data = load_developer_recommender_data("Dataset/developer_burnout.csv")
    recommender = BurnoutSuggestionRecommender(feature_columns=list(feature_names_key))
    recommender.fit(data)
    return recommender


def build_input_frame(
    age,
    exp,
    sleep_h,
    exercise,
    work_h,
    screen,
    meetings,
    commits,
    bugs,
    caffeine,
    feature_names,
):
    base = pd.DataFrame(
        [
            {
                "age": age,
                "experience_years": exp,
                "daily_work_hours": work_h,
                "sleep_hours": sleep_h,
                "caffeine_intake": caffeine,
                "bugs_per_day": bugs,
                "commits_per_day": commits,
                "meetings_per_day": meetings,
                "screen_time": screen,
                "exercise_hours": exercise,
            }
        ]
    )
    engineered = add_engineered_features(base)
    return engineered[feature_names]


def probability_by_class(probabilities, label_encoder):
    return {
        label: float(probabilities[idx])
        for idx, label in enumerate(label_encoder.classes_)
    }


def continuous_risk_score(probabilities, label_encoder):
    prob_map = probability_by_class(probabilities, label_encoder)
    return (
        100 * prob_map.get("High", 0.0)
        + 55 * prob_map.get("Medium", 0.0)
        + 15 * prob_map.get("Low", 0.0)
    )


def risk_message(label):
    if label == "High":
        return st.error("High burnout risk. Prioritize workload reduction, recovery time, and manager support.")
    if label == "Medium":
        return st.warning("Medium burnout risk. Review workload, meetings, sleep, and delivery pressure.")
    return st.success("Low burnout risk. Current signals are close to a healthier behavior pattern.")


def shap_waterfall_figure(explainer, input_scaled, input_data, prediction_idx):
    shap_values = explainer.shap_values(input_scaled)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        values = shap_values[prediction_idx][0]
        base_value = expected_value[prediction_idx]
    elif np.ndim(shap_values) == 3:
        values = shap_values[0, :, prediction_idx]
        base_value = expected_value[prediction_idx] if np.ndim(expected_value) > 0 else expected_value
    else:
        values = shap_values[0]
        base_value = expected_value

    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=input_data.iloc[0].values,
        feature_names=input_data.columns.tolist(),
    )

    plt.figure(figsize=(9, 5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    return plt.gcf()


try:
    model, scaler, feature_names, label_encoder = load_artifacts()
    explainer = load_shap_explainer(model)
    recommender = load_recommender(tuple(feature_names))
except Exception as exc:
    st.error("Failed to load project artifacts. Run `python save_model.py`, then restart Streamlit.")
    st.exception(exc)
    st.stop()


st.sidebar.title("PulseWatch")
st.sidebar.caption("Enter an employee profile and review the prediction with suggestions.")

with st.sidebar.form("employee_profile"):
    st.subheader("Personal")
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    exp = st.number_input("Experience years", min_value=0, max_value=40, value=5, step=1)

    st.subheader("Recovery")
    sleep_h = st.slider("Sleep hours/night", min_value=3.0, max_value=12.0, value=7.0, step=0.5)
    exercise = st.slider("Exercise hours/day", min_value=0.0, max_value=5.0, value=0.5, step=0.5)

    st.subheader("Work")
    work_h = st.slider("Daily work hours", min_value=2.0, max_value=16.0, value=8.0, step=0.5)
    screen = st.slider("Screen time hours/day", min_value=2.0, max_value=18.0, value=9.0, step=0.5)
    meetings = st.number_input("Meetings/day", min_value=0, max_value=15, value=2, step=1)

    st.subheader("Developer Signals")
    commits = st.number_input("Code commits/day", min_value=0, max_value=50, value=5, step=1)
    bugs = st.number_input("Bugs fixed/day", min_value=0, max_value=30, value=3, step=1)
    caffeine = st.number_input("Caffeine cups/day", min_value=0, max_value=15, value=2, step=1)

    submitted = st.form_submit_button("Analyze Employee")

top_n = st.sidebar.slider("Number of suggestions", min_value=3, max_value=5, value=3)
show_shap = st.sidebar.checkbox("Show SHAP explanation", value=False)

input_data = build_input_frame(
    age,
    exp,
    sleep_h,
    exercise,
    work_h,
    screen,
    meetings,
    commits,
    bugs,
    caffeine,
    feature_names,
)

input_scaled = scaler.transform(input_data)
prediction_idx = int(model.predict(input_scaled)[0])
probabilities = model.predict_proba(input_scaled)[0]
prediction_label = label_encoder.inverse_transform([prediction_idx])[0]
risk_score = continuous_risk_score(probabilities, label_encoder)
recommendation_report = recommender.recommend_for_profile(
    input_data.iloc[0],
    burnout_risk_score=risk_score,
    top_n=top_n,
)

prob_df = pd.DataFrame(
    {
        "Risk Level": label_encoder.classes_,
        "Probability %": np.round(probabilities * 100, 1),
    }
)
recommendations = recommendation_report["recommendations"][
    ["rank", "recommendation", "category", "confidence", "why"]
]
drivers = recommendation_report["top_drivers"][
    ["driver", "employee_value", "healthy_reference", "gap_strength"]
]

st.markdown('<div class="app-title">PulseWatch</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">A Hybrid Predictive System for Employee Burnout Prediction</div>',
    unsafe_allow_html=True,
)

with st.expander("Model details", expanded=False):
    c1, c2, c3 = st.columns(3)
    c1.metric("Classifier", type(model).__name__)
    c2.metric("Features", len(feature_names))
    c3.metric("Suggestion engine", str(recommender.best_model_info_["algorithm"]))

st.subheader("Prediction")
result_col, cluster_col = st.columns([0.62, 0.38], gap="large")

with result_col:
    metric_cols = st.columns(3)
    metric_cols[0].metric("Burnout level", prediction_label)
    metric_cols[1].metric("Risk score", f"{risk_score:.1f}%")
    metric_cols[2].metric("Confidence", f"{np.max(probabilities) * 100:.1f}%")
    risk_message(prediction_label)

with cluster_col:
    st.markdown(
        f"""
        <div class="cluster-box">
            <div class="cluster-label">Behavior cluster</div>
            <div class="cluster-value">{recommendation_report["cluster_name"]}</div>
            <div class="cluster-label" style="margin-top:0.8rem;">Recovery score</div>
            <div class="cluster-value">{recommendation_report["recovery_score"]}/100</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.subheader("Recommended Actions")
for _, row in recommendations.iterrows():
    st.markdown(
        f"""
        <div class="recommendation-card">
            <div class="recommendation-title">
                {int(row["rank"])}. {row["recommendation"]} - confidence {row["confidence"]}
            </div>
            <div class="recommendation-meta">{row["category"]}: {row["why"]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

tab_summary, tab_explainability, tab_engine = st.tabs(
    ["Summary", "Explainability", "Technical Details"]
)

with tab_summary:
    left, right = st.columns(2, gap="large")
    with left:
        st.write("Class probabilities")
        st.dataframe(prob_df, hide_index=True, width="stretch")
    with right:
        st.write("Current input used by model")
        st.dataframe(input_data, hide_index=True, width="stretch")

with tab_explainability:
    st.write("Main burnout drivers compared with the healthier cluster reference.")
    st.dataframe(drivers, hide_index=True, width="stretch")

    if show_shap:
        st.write("SHAP explanation for the classifier prediction")
        try:
            fig = shap_waterfall_figure(explainer, input_scaled, input_data, prediction_idx)
            st.pyplot(fig, clear_figure=True)
        except Exception as exc:
            st.warning("SHAP explanation could not be rendered for this run.")
            st.exception(exc)
    else:
        st.info("Enable SHAP explanation from the sidebar if you need the model-level waterfall plot.")

with tab_engine:
    st.write("Unsupervised clustering model comparison")
    st.dataframe(recommender.algorithm_summary(), hide_index=True, width="stretch")

st.caption("PulseWatch uses the trained burnout classifier for risk and an unsupervised recommender for suggestions.")
