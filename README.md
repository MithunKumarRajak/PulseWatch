# PulseWatch: Employee Burnout Predictor 🔥

An interactive Machine Learning web application designed to predict employee burnout risk based on work habits, lifestyle metrics, and engineered features.

## Live Demo
*(You can place your Streamlit Cloud link here once deployed!)*

## Overview
PulseWatch uses a **Random Forest Classifier** trained on synthetic minority over-sampling (SMOTE) balanced data to classify burnout risk into three categories: **Low**, **Medium**, and **High**. 

The app features:
- Real-time prediction based on 10 user inputs.
- 6 dynamically calculated engineered features (e.g., Work-Sleep ratio).
- **SHAP (SHapley Additive exPlanations)** Waterfall plots that explain exactly *why* the model made a specific prediction for that exact employee.

## Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/PulseWatch.git
   cd PulseWatch
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit App:
   ```bash
   streamlit run app.py
   ```

## Repository Contents
- `app.py` - The main Streamlit application script.
- `requirements.txt` - Python dependencies for deployment.
- `*.pkl` - Pre-trained model, scaler, feature names, and label encoder artifacts.
- `PulseWatch_Training.py` - The complete training pipeline (Data cleaning, Feature Engineering, GridSearchCV tuning, and Model evaluation).

## Built With
- [Streamlit](https://streamlit.io/)
- [Scikit-Learn](https://scikit-learn.org/)
- [SHAP](https://shap.readthedocs.io/en/latest/)
- [Pandas](https://pandas.pydata.org/)
