# PulseWatch — Burnout Prediction Dashboard

A machine-learning project that predicts **employee burnout risk** from workload, recovery, and developer-activity signals.  
The repository ships a ready-to-deploy **Streamlit** web app backed by pre-trained models.

---

## ✨ Features

| Capability | Details |
|---|---|
| **Developer Burnout** | Predicts stress level (regression) and burnout risk (classification) using developer metrics such as work hours, bugs, commits, screen time, etc. |
| **WFH Employee Burnout** | Predicts burnout score and risk for remote employees using work hours, meetings, sleep, task completion rate, etc. |
| **ML Models** | XGBoost regression + GridSearchCV-tuned Ensemble Voting Classifier (LogisticRegression + ExtraTrees + SVC) |
| **Interactive UI** | Two-tab Streamlit dashboard with sliders, auto-calculated engineered features, color-coded risk badges, and progress bars |

---

## 📁 Project Structure

```
BurnoutPrediction/
├── app.py                          # Streamlit dashboard (main entry point)
├── requirements.txt                # Python dependencies
├── models/                         # Serialized model & scaler artifacts (.pkl)
├── Dataset/
│   ├── developer_burnout.csv       # Developer burnout dataset
│   └── wfh_burnout.csv            # WFH burnout dataset
├── Developer_Burnout_ML.ipynb      # Full ML pipeline — Developer
├── WFH_Burnout_ML.ipynb            # Full ML pipeline — WFH
├── Plots/                          # Saved visualizations from notebooks
├── Notes/                          # Project notes
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/MithunKumarRajak/PulseWatch.git
cd PulseWatch
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The dashboard opens at **<http://localhost:8501>**.

---

## 🧠 ML Pipeline (Notebooks)

Each notebook follows a structured pipeline:

| Step | Description |
|---|---|
| **Step 1–2** | Import libraries & load data |
| **Step 3** | Data exploration & outlier analysis |
| **Step 4** | EDA visualizations |
| **Step 5** | Feature engineering (e.g. `sleep_deficit`, `work_sleep_ratio`, `productivity_per_hour`) |
| **Step 6** | Preprocessing — encoding, scaling |
| **Part A** | Regression — Linear, Gradient Boosting, XGBoost |
| **Part B1** | Classification — GBC, XGBoost, CatBoost |
| **Part B2** | SVM + 5-Fold Cross Validation |
| **Part B3** | Hyperparameter Tuning (RandomizedSearchCV) |
| **Part B4** | Ensemble Voting Classifier (GridSearchCV) |
| **Part B5** | Pipeline models — LR, RF, DT, SVM, KNN |

---

## 🛠️ Re-training Models

To re-export model artifacts after modifying notebooks:

1. Open `Developer_Burnout_ML.ipynb` → **Run All** → the final cell saves `.pkl` files to `models/`.
2. Open `WFH_Burnout_ML.ipynb` → **Run All** → the final cell saves `.pkl` files to `models/`.

---

## 📦 Deployment (Streamlit Cloud)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io).
3. Connect your GitHub repo.
4. Set **Main file path** to `app.py`.
5. Deploy — done!

---

## 📜 License

This project is for educational and demonstration purposes.
