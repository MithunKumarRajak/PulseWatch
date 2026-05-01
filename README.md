# PulseWatch

PulseWatch is a machine learning project for predicting employee burnout risk from workload, recovery, and developer activity signals. The repository includes a Streamlit app, training pipeline, notebook workflow, saved model artifacts, and an unsupervised recommendation engine for next-step suggestions.

## What the Project Includes

- Burnout risk prediction into `Low`, `Medium`, and `High`
- Feature engineering for work-recovery imbalance patterns
- SMOTE-based class balancing during training
- Model comparison across baseline, tuned, and ensemble classifiers
- SHAP-compatible deployed model for local explainability
- A suggestion engine that recommends practical actions based on learned behavior clusters

## Repository Structure

- `app.py` - Streamlit application for prediction, probability display, SHAP view, and recommendations
- `burnout_recommender.py` - recommendation engine and shared feature engineering helpers
- `save_model.py` - exports the current deployable model artifacts used by the app
- `PulseWatch_Model_Training.py` - script version of the training workflow
- `PulseWatch.ipynb` - notebook version of the training and EDA workflow
- `generate_pulsewatch_notebook.py` - regenerates the notebook file from a scripted source
- `Dataset/` - source CSV files used for training
- `Report/` - supporting writeups and project notes
- `burnout_model.pkl`, `scaler.pkl`, `feature_names.pkl`, `label_encoder.pkl` - artifacts loaded by the app

## Datasets

This project uses two datasets already included in the repository:

- `Dataset/developer_burnout.csv`
- `Dataset/wfh_burnout.csv`

The deployed app currently uses the developer dataset feature space plus engineered features. The broader training workflow also includes a shared-feature hybrid experiment combining both datasets.

## Local Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/BurnoutPrediction.git
   cd BurnoutPrediction
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Run the App

```bash
streamlit run app.py
```

The app loads the committed `.pkl` artifacts directly. If you retrain the model, regenerate those artifacts before running Streamlit again.

## Retrain or Rebuild Artifacts

Use the lightweight export path:

```bash
python save_model.py
```

Use the full training workflow:

- run `PulseWatch_Model_Training.py`, or
- open and execute `PulseWatch.ipynb`

If you want to regenerate the notebook file itself:

```bash
python generate_pulsewatch_notebook.py
```

## Model Notes

- The deployed classifier is a `RandomForestClassifier` so SHAP `TreeExplainer` works cleanly in the app.
- The training workflow also evaluates logistic regression, KNN, SVM, XGBoost, hybrid voting, and stacking models.
- Leakage-prone label-derived columns such as `stress_level` and `burnout_score` are excluded from training features.

## GitHub Readiness Notes

- Generated plot images are ignored by `.gitignore`.
- Model artifacts are intentionally kept in the repository because the app depends on them.
- The dataset folder is intentionally kept in the repository so the notebook and training script can run after cloning.

## Built With

- Streamlit
- pandas
- NumPy
- scikit-learn
- XGBoost
- imbalanced-learn
- SHAP
- matplotlib
- seaborn
