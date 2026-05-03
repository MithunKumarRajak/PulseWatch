import os
from typing import Optional

import joblib
import numpy as np
import streamlit as st


st.set_page_config(page_title="Burnout Predictor", page_icon="", layout="centered")


def load_model(path: str) -> Optional[object]:
    try:
        if os.path.exists(path):
            return joblib.load(path)
        return None
    except Exception:
        return None


def colored_badge(text: str, color: str) -> str:
    return f"<span style='background:{color};color:#fff;padding:6px 12px;border-radius:12px;font-weight:600;'>{text}</span>"


def map_risk_label(pred) -> str:
    if isinstance(pred, (list, tuple, np.ndarray)):
        val = pred[0]
    else:
        val = pred
    try:
        valf = float(val)
        mapping = {0.0: "Low", 1.0: "Medium", 2.0: "High"}
        return mapping.get(valf, str(val))
    except Exception:
        return str(val).title()


def safe_div(a, b):
    try:
        if b == 0 or b == 0.0:
            return 0.0
        return a / b
    except Exception:
        return 0.0


def load_models_for_developer():
    base = os.path.join("models")
    r_model = load_model(os.path.join(base, "developer_regression_model.pkl"))
    c_model = load_model(os.path.join(base, "developer_classification_model.pkl"))
    r_scaler = load_model(os.path.join(base, "developer_reg_scaler.pkl"))
    c_scaler = load_model(os.path.join(base, "developer_clf_scaler.pkl"))
    return r_model, c_model, r_scaler, c_scaler


def load_models_for_wfh():
    base = os.path.join("models")
    r_model = load_model(os.path.join(base, "wfh_regression_model.pkl"))
    c_model = load_model(os.path.join(base, "wfh_classification_model.pkl"))
    r_scaler = load_model(os.path.join(base, "wfh_reg_scaler.pkl"))
    c_scaler = load_model(os.path.join(base, "wfh_clf_scaler.pkl"))
    return r_model, c_model, r_scaler, c_scaler


def render_developer_tab():
    st.header("Developer Burnout")
    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 20, 44, 30, 1)
        experience_years = st.slider("Experience Years", 0.0, 30.0, 3.0, 0.5)
        daily_work_hours = st.slider("Daily Work Hours", 0.0, 24.0, 8.0, 0.5)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)
        caffeine_intake = st.number_input("Caffeine Intake (cups/day)", 0, 10, 1)

    with col2:
        bugs_per_day = st.number_input("Bugs per day", 0, 30, 1)
        commits_per_day = st.number_input("Commits per day", 0, 50, 3)
        meetings_per_day = st.number_input("Meetings per day", 0, 15, 2)
        screen_time = st.slider("Screen Time (hours)", 0.0, 24.0, 6.0, 0.5)
        exercise_hours = st.slider("Exercise Hours", 0.0, 2.0, 1.0, 0.1)

    # Engineered features
    bugs_per_commit = safe_div(bugs_per_day, commits_per_day)
    work_sleep_ratio = safe_div(daily_work_hours, sleep_hours)
    screen_sleep_ratio = safe_div(screen_time, sleep_hours)
    sleep_deficit = max(0.0, 8.0 - sleep_hours)
    total_digital_load = daily_work_hours + screen_time + meetings_per_day

    with st.expander(" Engineered Features (auto-calculated)"):
        st.write({
            "bugs_per_commit": round(bugs_per_commit, 3),
            "work_sleep_ratio": round(work_sleep_ratio, 3),
            "screen_sleep_ratio": round(screen_sleep_ratio, 3),
            "sleep_deficit": round(sleep_deficit, 3),
            "total_digital_load": round(total_digital_load, 3),
        })

    st.divider()

    r_model, c_model, r_scaler, c_scaler = load_models_for_developer()
    models_present = all([r_model, c_model, r_scaler, c_scaler])
    if not models_present:
        st.warning(" Model file not found. Please train and save the model first.")

    sleep_zero = sleep_hours == 0
    if sleep_zero:
        st.warning(" Sleep hours is zero — ratios will be invalid. Please enter a non-zero value.")

    predict_disabled = (not models_present) or sleep_zero

    if st.button(" Predict Developer Burnout", disabled=predict_disabled):
        # Order MUST match the DataFrame columns used during training:
        # age, experience_years, daily_work_hours, sleep_hours,
        # caffeine_intake, bugs_per_day, commits_per_day,
        # meetings_per_day, screen_time, exercise_hours,
        # bugs_per_commit, work_sleep_ratio, screen_sleep_ratio,
        # sleep_deficit, total_digital_load
        features = [
            age,
            experience_years,
            daily_work_hours,
            sleep_hours,
            caffeine_intake,
            bugs_per_day,
            commits_per_day,
            meetings_per_day,
            screen_time,
            exercise_hours,
            bugs_per_commit,
            work_sleep_ratio,
            screen_sleep_ratio,
            sleep_deficit,
            total_digital_load,
        ]

        x_reg = np.array(features).reshape(1, -1)
        x_clf = np.array(features).reshape(1, -1)

        try:
            x_reg_scaled = r_scaler.transform(x_reg)
            pred_stress = r_model.predict(x_reg_scaled)[0]
        except Exception:
            pred_stress = r_model.predict(x_reg)[0] if r_model is not None else 0.0

        try:
            x_clf_scaled = c_scaler.transform(x_clf)
            pred_burnout_raw = c_model.predict(x_clf_scaled)
        except Exception:
            pred_burnout_raw = c_model.predict(x_clf) if c_model is not None else ["Low"]

        burnout_label = map_risk_label(pred_burnout_raw)

        # Output
        st.metric(label="Stress Level Score", value=round(float(pred_stress), 2))

        color = "green" if burnout_label == "Low" else ("orange" if burnout_label == "Medium" else "red")
        st.markdown(colored_badge(burnout_label, color), unsafe_allow_html=True)

        # Progress bar normalize 0-100 (stress_level range in training data)
        norm = float(pred_stress)
        try:
            norm = max(0.0, min(100.0, norm))
            st.progress(norm / 100.0)
        except Exception:
            pass

        if burnout_label == "Low":
            st.write("You appear to be managing workload well. Keep regular sleep and breaks.")
        elif burnout_label == "Medium":
            st.write("Moderate risk — consider reducing screen time and managing meetings.")
        else:
            st.write("High risk — consider immediate workload adjustments and professional support.")


def render_wfh_tab():
    st.header("WFH Employee Burnout")
    col1, col2 = st.columns(2)

    with col1:
        day_type = st.selectbox("Day Type", options=["Weekday", "Weekend"], index=0)
        work_hours = st.slider("Work Hours", 0.0, 16.0, 8.0, 0.5)
        screen_time_hours = st.slider("Screen Time Hours", 0.0, 16.0, 6.0, 0.5)
        meetings_count = st.number_input("Meetings Count", 0, 20, 3)
        breaks_taken = st.number_input("Breaks Taken", 1, 5, 3)

    with col2:
        after_hours_work = st.selectbox("After Hours Work", options=["Yes", "No"], index=1)
        sleep_hours = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5, key="wfh_sleep")
        task_completion_rate = st.slider("Task Completion Rate (%)", 0.0, 110.0, 70.0, 1.0)

    after_hours_val = 1 if after_hours_work == "Yes" else 0
    # LabelEncoder encodes alphabetically: Weekday=0, Weekend=1
    day_type_val = 0 if day_type == "Weekday" else 1

    productivity_per_hour = safe_div(task_completion_rate, work_hours)
    workload = work_hours + meetings_count + screen_time_hours
    sleep_deficit = max(0.0, 8.0 - sleep_hours)

    with st.expander("Engineered Features (auto-calculated)"):
        st.write({
            "productivity_per_hour": round(productivity_per_hour, 3),
            "workload": round(workload, 3),
            "sleep_deficit": round(sleep_deficit, 3),
        })

    st.divider()

    r_model, c_model, r_scaler, c_scaler = load_models_for_wfh()
    models_present = all([r_model, c_model, r_scaler, c_scaler])
    if not models_present:
        st.warning("Model file not found. Please train and save the model first.")

    sleep_zero = sleep_hours == 0
    if sleep_zero:
        st.warning("Sleep hours is zero — ratios will be invalid. Please enter a non-zero value.")

    predict_disabled = (not models_present) or sleep_zero

    if st.button("Predict WFH Burnout", disabled=predict_disabled):
        # Order MUST match the DataFrame columns used during training:
        # day_type, work_hours, screen_time_hours, meetings_count,
        # breaks_taken, after_hours_work, sleep_hours,
        # task_completion_rate, productivity_per_hour,
        # workload, sleep_deficit
        features = [
            day_type_val,
            work_hours,
            screen_time_hours,
            meetings_count,
            breaks_taken,
            after_hours_val,
            sleep_hours,
            task_completion_rate,
            productivity_per_hour,
            workload,
            sleep_deficit,
        ]

        x_reg = np.array(features).reshape(1, -1)
        x_clf = np.array(features).reshape(1, -1)

        try:
            x_reg_scaled = r_scaler.transform(x_reg)
            pred_score = r_model.predict(x_reg_scaled)[0]
        except Exception:
            pred_score = r_model.predict(x_reg)[0] if r_model is not None else 0.0

        try:
            x_clf_scaled = c_scaler.transform(x_clf)
            pred_risk_raw = c_model.predict(x_clf_scaled)
        except Exception:
            pred_risk_raw = c_model.predict(x_clf) if c_model is not None else ["Low"]

        burnout_label = map_risk_label(pred_risk_raw)

        st.metric(label="Burnout Score", value=round(float(pred_score), 2))

        # progress 0-100
        try:
            score = float(pred_score)
            score_clipped = max(0.0, min(100.0, score))
            st.progress(score_clipped / 100.0)
        except Exception:
            pass

        color = "green" if burnout_label == "Low" else ("orange" if burnout_label == "Medium" else "red")
        st.markdown(colored_badge(burnout_label, color), unsafe_allow_html=True)

        st.write("**Score thresholds used:** score < 45 → Low, 45–95 → Medium, > 95 → High")

        if burnout_label == "Low":
            st.write("Good job — keep consistent sleep and structured work periods.")
        elif burnout_label == "Medium":
            st.write("Consider increasing breaks, reducing after-hours work, and improving task focus.")
        else:
            st.write("High risk — take immediate actions: rest, boundary-setting, or seek support.")


def main():
    st.markdown("# Burnout Prediction Dashboard")
    st.markdown("### Predict burnout risk using Machine Learning")

    tab1, tab2 = st.tabs([" Developer Burnout", " WFH Employee Burnout"])

    with tab1:
        render_developer_tab()

    with tab2:
        render_wfh_tab()

    st.divider()
    st.markdown("Built with Streamlit")


if __name__ == "__main__":
    main()
