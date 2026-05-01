# PulseWatch — Upgrade Plan

## Current Baseline
| Best Model | F1 Macro | Balanced Acc |
|---|---|---|
| Logistic Regression | 79.18% | 81.22% |
| Hybrid Ensemble | 65.66% | 67.30% |

---

## Priority 1: High Impact Upgrades

### 1. Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
**Why:** Current models use default/manual params. Tuning can boost F1 by 3-8%.
```
- RandomForest: n_estimators, max_depth, min_samples_split
- XGBoost: learning_rate, max_depth, subsample, colsample_bytree
- SVM: C, gamma, kernel
- KNN: n_neighbors, weights, metric
```
**Complexity:** Medium | **Expected boost:** +3-8% F1

### 2. Stacking Ensemble (Fix the Hybrid)
**Why:** Current VotingClassifier on 4 shared features loses information. A StackingClassifier with a meta-learner would perform much better.
```
Level 0: LR + SVM + RF + XGBoost (on full developer features)
Level 1: Logistic Regression meta-learner
```
**Complexity:** Low | **Expected boost:** +2-5% over best single model

### 3. SHAP Explainability
**Why:** Shows WHY a prediction was made — very impressive in project reports.
```
- SHAP summary plot (global feature importance)
- SHAP force plot (individual prediction explanation)
- SHAP dependence plots (feature interactions)
```
**Complexity:** Low | **Presentation impact:** Very high

### 4. ROC Curves + Learning Curves
**Why:** Standard ML evaluation visuals expected in any ML project.
```
- ROC-AUC (one-vs-rest) for all models
- Learning curves (train vs validation score over dataset size)
- Precision-Recall curves (better for imbalanced data)
```
**Complexity:** Low | **Presentation impact:** High

---

## Priority 2: Medium Impact

### 5. Feature Engineering
**Why:** Create new features from existing ones to improve model performance.
```
Developer dataset:
- work_sleep_ratio = daily_work_hours / sleep_hours
- screen_work_ratio = screen_time / daily_work_hours
- productivity_index = commits_per_day / bugs_per_day
- overwork_flag = 1 if daily_work_hours > 10

WFH dataset:
- screen_work_ratio = screen_time_hours / work_hours
- meeting_intensity = meetings_count / work_hours
```
**Complexity:** Low | **Expected boost:** +1-3% F1

### 6. Streamlit Web App
**Why:** Interactive demo where you input employee data and get a burnout prediction. Very impressive for presentations.
```
- Input form for employee features
- Real-time prediction with probability chart
- SHAP explanation for each prediction
```
**Complexity:** Medium | **Presentation impact:** Very high

### 7. Detailed GridSearchCV Report
**Why:** Show you tested many hyperparameter combinations systematically.
```
- Best params per model
- CV score distribution boxplot
- Param sensitivity analysis
```
**Complexity:** Medium | **Presentation impact:** Medium

---

## Priority 3: Nice to Have

### 8. Simple Neural Network (MLP)
**Why:** Shows you explored deep learning approaches too.
```
- sklearn MLPClassifier or PyTorch simple model
- Compare with traditional ML models
```
**Complexity:** Medium | **Expected boost:** +0-2% (tabular data rarely benefits)

### 9. PCA / t-SNE Visualization
**Why:** 2D visualization of how classes separate in feature space.
```
- PCA scatter plot colored by burnout level
- t-SNE for non-linear separation visualization
```
**Complexity:** Low | **Presentation impact:** Medium

### 10. Model Saving & Loading (Deployment Ready)
**Why:** Save trained model for reuse without retraining.
```
- joblib.dump() / joblib.load()
- Save scaler + encoder + model as pipeline
```
**Complexity:** Very Low | **Practical value:** High

---

## Recommended Implementation Order

```
Step 1: Hyperparameter Tuning (#1)         → Improve scores
Step 2: Stacking Ensemble (#2)             → Fix hybrid model
Step 3: Feature Engineering (#5)           → Squeeze more performance
Step 4: ROC + Learning Curves (#4)         → Better evaluation visuals
Step 5: SHAP (#3)                          → Explainability
Step 6: Model Saving (#10)                 → Deployment ready
Step 7: Streamlit App (#6)                 → Interactive demo (optional but impressive)
```

> [!TIP]
> Steps 1-5 are enough for an excellent ML project submission. Step 6-7 would make it stand out significantly.
