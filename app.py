import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="PulseWatch | Burnout Prediction",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium SaaS aesthetics
st.markdown("""
<style>
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global Styling & Typography */
    .main {
        background-color: #f4f7f6;
        font-family: 'Inter', sans-serif;
        padding-bottom: 80px; /* Prevent footer from covering content */
    }
    
    /* Premium Button with Micro-animation */
    .stButton>button {
        width: 100%; 
        font-size: 18px; 
        font-weight: 600; 
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white; 
        border: none;
        border-radius: 12px;
        padding: 12px 0;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(30, 60, 114, 0.3);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(30, 60, 114, 0.4);
        background: linear-gradient(135deg, #2a5298 0%, #1e3c72 100%);
        color: white;
    }
    
    /* Premium Risk Badges with Gradients */
    .risk-high {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        color: white; padding: 15px 20px; border-radius: 12px; font-weight: 700; font-size: 26px; 
        display: inline-block; text-align: center; width: 100%; box-shadow: 0 10px 20px rgba(255, 75, 43, 0.3);
    }
    .risk-medium {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        color: white; padding: 15px 20px; border-radius: 12px; font-weight: 700; font-size: 26px; 
        display: inline-block; text-align: center; width: 100%; box-shadow: 0 10px 20px rgba(247, 151, 30, 0.3);
    }
    .risk-low {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white; padding: 15px 20px; border-radius: 12px; font-weight: 700; font-size: 26px; 
        display: inline-block; text-align: center; width: 100%; box-shadow: 0 10px 20px rgba(56, 239, 125, 0.3);
    }
    
    /* Ultra-Premium Glassmorphic Footer */
    .custom-footer {
        position: fixed; 
        left: 0; 
        bottom: 0; 
        width: 100%; 
        background: rgba(17, 24, 39, 0.85);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        color: #e5e7eb; 
        text-align: center; 
        padding: 16px; 
        font-size: 14px; 
        font-weight: 500;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        z-index: 9999;
        letter-spacing: 0.5px;
    }
    .custom-footer span {
        color: #3b82f6;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('burnout_model.pkl')
    scaler = joblib.load('scaler.pkl')
    feature_names = joblib.load('feature_names.pkl')
    le = joblib.load('label_encoder.pkl')
    return model, scaler, feature_names, le

try:
    model, scaler, feature_names, le = load_artifacts()
except Exception as e:
    st.error("Failed to load model artifacts. Make sure to run `python save_model.py` first.")
    st.stop()

# Build SHAP explainer
@st.cache_resource
def get_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = get_explainer(model)

st.title("🔥 PulseWatch: Employee Burnout Predictor")
st.markdown("""
**Welcome to PulseWatch!** 👋  
This tool helps HR professionals, managers, and individuals predict if an employee is at risk of burnout. 
You don't need to be a data scientist to use it! Just adjust the sliders on the left to match an employee's typical day, and our AI will calculate their burnout risk and explain exactly *why* it made that decision.
""")

st.sidebar.header("📊 Employee Profile")
st.sidebar.markdown("Slide the dials to match the employee:")

# Move the button to the TOP so it's always visible regardless of screen size!
predict_btn = st.sidebar.button("Predict Burnout Risk 🚀")
st.sidebar.markdown("---")

with st.sidebar.expander("👤 Personal & Rest", expanded=True):
    age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
    exp = st.number_input("Experience (years)", min_value=0, max_value=40, value=5, step=1)
    sleep_h = st.slider("Sleep (hours/night)", min_value=3.0, max_value=12.0, value=7.0, step=0.5, help="Average hours of sleep per night")
    exercise = st.slider("Exercise (hours/day)", min_value=0.0, max_value=5.0, value=0.5, step=0.5)

with st.sidebar.expander("💼 Work Habits", expanded=True):
    work_h = st.slider("Daily Work (hours)", min_value=2.0, max_value=16.0, value=8.0, step=0.5, help="Actual hours spent working per day")
    screen = st.slider("Screen Time (hours/day)", min_value=2.0, max_value=18.0, value=9.0, step=0.5)
    meetings = st.number_input("Meetings per day", min_value=0, max_value=15, value=2, step=1)

with st.sidebar.expander("💻 Coding & Stress", expanded=True):
    commits = st.number_input("Code Commits/day", min_value=0, max_value=50, value=5, step=1, help="How much code they deliver daily")
    bugs = st.number_input("Bugs Fixed/day", min_value=0, max_value=30, value=3, step=1, help="Debugging can be a major source of frustration")
    caffeine = st.number_input("Caffeine (Cups/day)", min_value=0, max_value=15, value=2, step=1)

if predict_btn:
    with st.spinner("Analyzing profile..."):
        # 1. Base features matching original order (approx)
        # Note: must match the exact feature generation logic
        
        # 2. Compute Engineered Features
        work_sleep_ratio = work_h / sleep_h if sleep_h > 0 else work_h
        screen_work_ratio = screen / work_h if work_h > 0 else screen
        bug_commit_ratio = bugs / (commits + 1)
        overwork_flag = 1 if work_h > 10 else 0
        low_sleep_flag = 1 if sleep_h < 6 else 0
        meeting_intensity = meetings / work_h if work_h > 0 else 0
        
        # Create Dataframe with exactly same columns as X_dev_fe
        input_data = pd.DataFrame([{
            'age': age,
            'experience_years': exp,
            'daily_work_hours': work_h,
            'sleep_hours': sleep_h,
            'caffeine_intake': caffeine,
            'bugs_per_day': bugs,
            'commits_per_day': commits,
            'meetings_per_day': meetings,
            'screen_time': screen,
            'exercise_hours': exercise,
            'work_sleep_ratio': work_sleep_ratio,
            'screen_work_ratio': screen_work_ratio,
            'bug_commit_ratio': bug_commit_ratio,
            'overwork_flag': overwork_flag,
            'low_sleep_flag': low_sleep_flag,
            'meeting_intensity': meeting_intensity
        }])
        
        # Ensure column order matches exactly
        input_data = input_data[feature_names]
        
        # 3. Scale the input
        input_scaled = scaler.transform(input_data)
        
        # 4. Predict
        prediction_idx = model.predict(input_scaled)[0]
        proba = model.predict_proba(input_scaled)[0]
        prediction_label = le.inverse_transform([prediction_idx])[0]
        
        st.markdown("---")
        
        # 5. Display Result
        colA, colB = st.columns([1, 2])
        
        with colA:
            st.subheader("Prediction Result")
            if prediction_label == "High":
                st.markdown(f"<span class='risk-high'>HIGH BURNOUT RISK</span>", unsafe_allow_html=True)
                st.error("Action Required: This employee shows strong indicators of burnout. Consider workload reduction or mandatory time off.")
            elif prediction_label == "Medium":
                st.markdown(f"<span class='risk-medium'>MEDIUM BURNOUT RISK</span>", unsafe_allow_html=True)
                st.warning("Monitor closely: Employee is at risk. Consider reviewing their meetings and daily workload.")
            else:
                st.markdown(f"<span class='risk-low'>LOW BURNOUT RISK</span>", unsafe_allow_html=True)
                st.success("Healthy baseline: This profile does not indicate significant burnout risk.")
                
            st.markdown("### Confidence Score")
            st.progress(float(np.max(proba)))
            st.caption(f"{np.max(proba)*100:.1f}% confidence")
            
        with colB:
            st.subheader("🧠 Why was this predicted?")
            st.info("""
            **How to read this chart:**
            - Look at the **bottom** to see where the prediction started, and the **top** to see where it ended.
            - 🔴 **Red bars** represent habits/factors that pushed the employee *towards* this risk level.
            - 🔵 **Blue bars** represent healthy habits that pushed them *away* from this risk level.
            """)
            
            # 6. SHAP Explanation
            # Calculate SHAP values using scaled input so it matches the model
            shap_values = explainer(input_scaled) 
            
            # Overwrite the data attribute to display unscaled original values on the plot
            shap_values.data = input_data.values
            shap_values.feature_names = input_data.columns.tolist()
            
            fig, ax = plt.subplots(figsize=(8, 4))
            
            # Since it's a multi-class model, shap_values.values might be 3D. 
            # We slice it to get the 1D explanation for the predicted class.
            if len(shap_values.shape) == 3:
                shap.plots.waterfall(shap_values[0, :, prediction_idx], show=False)
            else:
                shap.plots.waterfall(shap_values[0], show=False)
                
            st.pyplot(fig)

else:
    st.info("👈 Adjust the employee profile on the left and click 'Predict' to see the model in action.")
    
    st.markdown("### 💡 Try these scenarios:")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**The Overworked Coder**")
        st.caption("12h work, 5h sleep, 15 bugs, 4 coffees")
    with col2:
        st.markdown("**The Meeting Fatigue**")
        st.caption("9h work, 8 meetings, 12h screen time")
    with col3:
        st.markdown("**The Healthy Baseline**")
        st.caption("8h work, 8h sleep, 2 meetings, 1h exercise")

st.markdown("""
<div class="custom-footer">
    <span>PulseWatch</span> — A Predictive System for Employee Burnout. Built with Streamlit.
""", unsafe_allow_html=True)
