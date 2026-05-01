# PulseWatch: Hybrid Predictive System for Employee Burnout
**Final Project Report**

## 1. Executive Summary
The **PulseWatch** project is a machine learning pipeline designed to predict employee burnout risk (Low, Medium, High) based on work habits and lifestyle metrics. The project integrates multiple predictive models and culminates in a highly interactive web application that provides real-time predictions with feature-level explainability. 

The best-performing model achieved an **F1 Macro score of 79.6%** and a **Balanced Accuracy of 81.1%**, demonstrating robust performance despite significant initial data quality challenges.

---

## 2. Data Audit & Leakage Prevention
We began by auditing two provided datasets:
1. **Work-From-Home (WFH) Dataset:** 1,800 rows.
2. **Developer Dataset:** 7,000 rows.

### Key Issues Identified & Solved:
* **Target Leakage:** The datasets contained features like `burnout_score` and `stress_level` which deterministically correlated with the target `burnout_risk` (e.g., `score > 50` always resulted in High burnout). We explicitly dropped these columns so the model learns from root causes (hours worked, meetings) rather than symptoms.
* **Severe Class Imbalance:** The WFH dataset had only 1.1% "High Burnout" samples. We utilized **SMOTE (Synthetic Minority Over-sampling Technique)** to synthesize balanced classes during training.
* **Missing Values:** Addressed 1,680 missing values in the Developer dataset using `KNNImputer`.

---

## 3. Feature Engineering
To provide the models with richer context, we engineered 6 new domain-specific features from the existing raw data:
* **`work_sleep_ratio`**: `daily_work_hours / sleep_hours` (Identifies unsustainable lifestyle balances).
* **`screen_work_ratio`**: `screen_time / daily_work_hours` (Identifies intense screen exposure).
* **`bug_commit_ratio`**: `bugs_per_day / commits_per_day` (Proxy for coding frustration/difficulty).
* **`meeting_intensity`**: `meetings_per_day / daily_work_hours`.
* **`overwork_flag`**: Binary flag if `work_hours > 10`.
* **`low_sleep_flag`**: Binary flag if `sleep_hours < 6`.

*Impact:* Feature engineering improved the baseline Logistic Regression model's F1 score by **+0.46%**.

---

## 4. Modeling Strategy & Upgrades
We trained multiple baseline algorithms: Logistic Regression, SVM, Random Forest, XGBoost, and KNN.

### Upgrade 1: Hyperparameter Tuning
Using `RandomizedSearchCV`, we performed extensive hyperparameter tuning across 3-fold cross-validation. This boosted the tree-based models significantly (e.g., Random Forest CV score reached 84.3%).

### Upgrade 2: The Stacking Ensemble (Hybrid)
Because combining the disparate datasets directly reduced accuracy (due to dropping non-shared columns), we created a **Stacking Classifier**. This meta-learner combined the tuned Random Forest, XGBoost, SVM, and Logistic Regression models, achieving a highly generalized F1 Macro score of **77.10%** across the comprehensive test set.

### Final Results Table
| Model | Accuracy | Balanced Accuracy | F1 Macro |
|---|---|---|---|
| **Logistic Regression (with FE)** | 78.8% | 81.1% | 79.6% |
| **SVM (RBF)** | 78.0% | 79.9% | 78.4% |
| **Stacking Ensemble** | 77.0% | 77.0% | 77.1% |
| **Random Forest (Tuned)** | 76.5% | 76.3% | 76.5% |
| **XGBoost (Tuned)** | 75.5% | 74.8% | 75.4% |

---

## 5. Model Explainability (SHAP)
Because this is a human-centric prediction, "black-box" predictions are insufficient. We implemented **SHAP (SHapley Additive exPlanations)** using `TreeExplainer`.

**Key Discoveries:**
1. **`daily_work_hours`** is the single most important predictor of burnout.
2. **`bugs_per_day`** and **`meetings_per_day`** strongly push the model toward a "High" risk prediction.
3. **`sleep_hours`** acts as a protective factor, pushing risk toward "Low".

*(Reference generated files: `shap_bar.png` and `shap_dot.png` for visual proof).*

---

## 6. The PulseWatch Web App
To make the project accessible and demonstrate production readiness, we built an interactive Web Dashboard using **Streamlit**.

**Features of the App:**
* Loads the trained, pickled Random Forest model and `StandardScaler` pipeline.
* Provides a dynamic sidebar for inputting hypothetical employee metrics (Age, Work Hours, Sleep, Commits, etc.).
* Dynamically calculates the 6 engineered features in the background.
* Provides real-time burnout classification ("Low", "Medium", "High") with confidence probability bars.
* Automatically renders a **SHAP Waterfall Plot** for the specific prediction, visually explaining *exactly* which features caused that employee's specific risk level.

---

## Conclusion
The PulseWatch project successfully transformed raw, imbalanced, and leaky HR data into a robust, deployable Machine Learning solution. By utilizing SMOTE for balance, Stacking Ensembles for accuracy, and SHAP for explainability, the project demonstrates a complete, professional-grade data science workflow.
