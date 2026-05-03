# 🏗️ PulseWatch — Best Training Approach

## TL;DR Recommendation

Given your project title **"A Hybrid Predictive System"**, I recommend this phased approach:

```
Phase 1 → Train on Developer dataset (better features, more data)
Phase 2 → Train on WFH dataset (different domain)
Phase 3 → Combine via Hybrid Ensemble (matches your project title)
```

---

## 📋 Available Features After Removing Leakage

### WFH Dataset (after dropping leaky + ID columns)

```
Features (6):  work_hours, screen_time_hours, meetings_count, 
               breaks_taken, after_hours_work, sleep_hours, day_type
Target:        burnout_risk (Low/Medium/High)
Usable rows:   1,800
```

### Developer Dataset (after dropping leaky column)

```
Features (10): age, experience_years, daily_work_hours, sleep_hours,
               caffeine_intake, bugs_per_day, commits_per_day,
               meetings_per_day, screen_time, exercise_hours
Target:        burnout_level (Low/Medium/High)
Usable rows:   ~6,860 (after dropping nulls)
```

> [!IMPORTANT]
> Dropped columns: `burnout_score`, `task_completion_rate` (WFH) and `stress_level` (Developer) — these cause data leakage.

---

## Three Training Approaches (Best → Good)

### Approach 1: Hybrid Ensemble ⭐ RECOMMENDED
>
> *Best accuracy + matches your "Hybrid Predictive System" title*

```
┌──────────────────────┐    ┌──────────────────────┐
│   Model A (Dev)      │    │   Model B (WFH)      │
│   XGBoost/RF on      │    │   XGBoost/RF on      │
│   10 dev features    │    │   7 WFH features     │
│   → predict proba    │    │   → predict proba    │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           └──────────┬────────────────┘
                      ▼
           ┌──────────────────────┐
           │   Meta-Learner       │
           │   (Voting / Stacking)│
           │   → Final prediction │
           └──────────────────────┘
```

**Why this is best:**

- Each model learns domain-specific patterns
- No feature alignment issues — each model uses ALL its dataset's features
- Ensemble reduces overfitting
- Literally a "hybrid" system

---

### Approach 2: Combined Common-Feature Model
>
> *Simpler, larger dataset, fewer features*

Use only the 4 shared features across both datasets:

```
work_hours, sleep_hours, meetings, screen_time → burnout_level
```

**Pros:** 8,660 combined rows, simple pipeline
**Cons:** Loses many useful features (age, caffeine, bugs, exercise, etc.)

---

### Approach 3: Developer-Only Model
>
> *Best single-model performance*

Train only on the developer dataset (10 features, ~6,860 rows).

**Pros:** Strongest feature correlations, most data
**Cons:** Won't generalize to non-developer employees

---

## 🔧 Recommended Models to Try

| Model | Why | Best For |
|---|---|---|
| **Random Forest** | Handles imbalance well, no scaling needed | Baseline |
| **XGBoost** | Best for tabular data, handles NaN | Primary model |
| **LightGBM** | Faster than XGBoost, similar performance | Large data |
| **SVM (RBF)** | Good for small datasets | WFH dataset |
| **Logistic Regression** | Interpretable, good baseline | Comparison |
| **KNN** | Simple, no assumptions | Baseline |

> [!TIP]
> For a college ML project, showing **5-6 models compared** with proper evaluation is ideal. Use Random Forest, XGBoost, SVM, Logistic Regression, KNN, and then the Hybrid Ensemble as your final model.

---

## 📊 Evaluation Strategy

> [!WARNING]
> Do NOT use plain **accuracy** — it will be misleading due to class imbalance (WFH dataset is 85% "Low").

### Use These Metrics Instead

```python
from sklearn.metrics import (
    classification_report,      # Precision, Recall, F1 per class
    f1_score,                   # F1 macro for overall
    confusion_matrix,           # See misclassifications
    ConfusionMatrixDisplay,     # Visualize confusion matrix
    balanced_accuracy_score,    # Adjusted for imbalance
    roc_auc_score               # AUC (one-vs-rest)
)
```

### Comparison Table Format

```
Model              Accuracy   Balanced-Acc   F1-Macro   Precision   Recall
─────────────────────────────────────────────────────────────────────────
Logistic Reg        xx.x%       xx.x%        xx.x%      xx.x%      xx.x%
KNN                 xx.x%       xx.x%        xx.x%      xx.x%      xx.x%
SVM                 xx.x%       xx.x%        xx.x%      xx.x%      xx.x%
Random Forest       xx.x%       xx.x%        xx.x%      xx.x%      xx.x%
XGBoost             xx.x%       xx.x%        xx.x%      xx.x%      xx.x%
Hybrid Ensemble     xx.x%       xx.x%        xx.x%      xx.x%      xx.x%  ← Your model
```

---

## 💻 Complete Implementation Code

### Step 1: Data Preprocessing

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, f1_score, confusion_matrix, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ── Load & Clean WFH Dataset ──
df_wfh = pd.read_csv('./Dataset/wfh_burnout.csv')
df_wfh.drop(columns=['user_id', 'burnout_score', 'task_completion_rate'], inplace=True)
df_wfh['day_type'] = df_wfh['day_type'].map({'Weekday': 1, 'Weekend': 0})
df_wfh.rename(columns={'burnout_risk': 'burnout_level'}, inplace=True)

# ── Load & Clean Developer Dataset ──
df_dev = pd.read_csv('./Dataset/developer_burnout.csv')
df_dev.drop(columns=['stress_level'], inplace=True)
df_dev.dropna(inplace=True)

# ── Encode Target ──
le = LabelEncoder()
le.fit(['Low', 'Medium', 'High'])  # Consistent encoding

df_wfh['target'] = le.transform(df_wfh['burnout_level'])
df_dev['target'] = le.transform(df_dev['burnout_level'])

print(f"WFH:       {df_wfh.shape[0]} rows, {df_wfh.shape[1]-2} features")
print(f"Developer: {df_dev.shape[0]} rows, {df_dev.shape[1]-2} features")
print(f"Classes:   {le.classes_}")
```

### Step 2: Handle Class Imbalance (SMOTE)

```python
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ── WFH split ──
X_wfh = df_wfh.drop(columns=['burnout_level', 'target'])
y_wfh = df_wfh['target']
X_wfh_train, X_wfh_test, y_wfh_train, y_wfh_test = train_test_split(
    X_wfh, y_wfh, test_size=0.2, random_state=42, stratify=y_wfh
)

# ── Developer split ──
X_dev = df_dev.drop(columns=['burnout_level', 'target'])
y_dev = df_dev['target']
X_dev_train, X_dev_test, y_dev_train, y_dev_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
)

# ── Apply SMOTE on training sets ──
smote = SMOTE(random_state=42)

X_wfh_train_sm, y_wfh_train_sm = smote.fit_resample(X_wfh_train, y_wfh_train)
X_dev_train_sm, y_dev_train_sm = smote.fit_resample(X_dev_train, y_dev_train)

print(f"\nWFH After SMOTE:  {pd.Series(y_wfh_train_sm).value_counts().to_dict()}")
print(f"Dev After SMOTE:  {pd.Series(y_dev_train_sm).value_counts().to_dict()}")
```

### Step 3: Train Multiple Models (Developer Dataset)

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# Scale features for SVM, KNN, and LR
scaler_dev = StandardScaler()
X_dev_train_scaled = scaler_dev.fit_transform(X_dev_train_sm)
X_dev_test_scaled = scaler_dev.transform(X_dev_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=7),
    'SVM': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=200, class_weight='balanced', random_state=42
    ),
    'XGBoost': XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    ),
}

results = {}
for name, model in models.items():
    # Use scaled data for LR, KNN, SVM; raw for tree-based
    if name in ['Logistic Regression', 'KNN', 'SVM']:
        model.fit(X_dev_train_scaled, y_dev_train_sm)
        y_pred = model.predict(X_dev_test_scaled)
    else:
        model.fit(X_dev_train_sm, y_dev_train_sm)
        y_pred = model.predict(X_dev_test)
    
    bal_acc = balanced_accuracy_score(y_dev_test, y_pred)
    f1 = f1_score(y_dev_test, y_pred, average='macro')
    results[name] = {'Balanced Accuracy': bal_acc, 'F1 Macro': f1}
    
    print(f"\n{'='*50}")
    print(f"{name}")
    print(f"{'='*50}")
    print(classification_report(y_dev_test, y_pred, target_names=le.classes_))

# ── Results Table ──
results_df = pd.DataFrame(results).T.round(4)
print("\n\n📊 MODEL COMPARISON:")
print(results_df.sort_values('F1 Macro', ascending=False))
```

### Step 4: Build the Hybrid Ensemble 🌟

```python
from sklearn.ensemble import VotingClassifier

# ── Train best model on each dataset ──

# Model A: XGBoost on Developer data
model_dev = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    use_label_encoder=False, eval_metric='mlogloss', random_state=42
)
model_dev.fit(X_dev_train_sm, y_dev_train_sm)

# Model B: Random Forest on WFH data
model_wfh = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', random_state=42
)
model_wfh.fit(X_wfh_train_sm, y_wfh_train_sm)

# ── For Hybrid prediction on shared features ──
shared_features_wfh = ['work_hours', 'sleep_hours', 'meetings_count', 'screen_time_hours']
shared_features_dev = ['daily_work_hours', 'sleep_hours', 'meetings_per_day', 'screen_time']
common_names = ['work_hours', 'sleep_hours', 'meetings', 'screen_time']

# Prepare common-feature datasets
X_wfh_common = df_wfh[shared_features_wfh].copy()
X_wfh_common.columns = common_names
y_wfh_all = df_wfh['target']

X_dev_common = df_dev[shared_features_dev].copy()
X_dev_common.columns = common_names
y_dev_all = df_dev['target']

# Combine
X_combined = pd.concat([X_wfh_common, X_dev_common], ignore_index=True)
y_combined = pd.concat([y_wfh_all, y_dev_all], ignore_index=True)

X_comb_train, X_comb_test, y_comb_train, y_comb_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

X_comb_train_sm, y_comb_train_sm = smote.fit_resample(X_comb_train, y_comb_train)

# ── Hybrid Voting Ensemble ──
hybrid_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ],
    voting='soft'  # Uses probability averaging
)

scaler_comb = StandardScaler()
X_comb_train_scaled = scaler_comb.fit_transform(X_comb_train_sm)
X_comb_test_scaled = scaler_comb.transform(X_comb_test)

hybrid_model.fit(X_comb_train_scaled, y_comb_train_sm)
y_hybrid_pred = hybrid_model.predict(X_comb_test_scaled)

print("\n🌟 HYBRID ENSEMBLE RESULTS (Combined Dataset)")
print("=" * 50)
print(classification_report(y_comb_test, y_hybrid_pred, target_names=le.classes_))
print(f"Balanced Accuracy: {balanced_accuracy_score(y_comb_test, y_hybrid_pred):.4f}")
print(f"F1 Macro:          {f1_score(y_comb_test, y_hybrid_pred, average='macro'):.4f}")
```

### Step 5: Visualizations

```python
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import ConfusionMatrixDisplay

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# ── Confusion Matrix ──
ConfusionMatrixDisplay.from_predictions(
    y_comb_test, y_hybrid_pred,
    display_labels=le.classes_,
    cmap='Blues', ax=axes[0]
)
axes[0].set_title('Hybrid Ensemble — Confusion Matrix')

# ── Model Comparison Bar Chart ──
results['Hybrid Ensemble'] = {
    'Balanced Accuracy': balanced_accuracy_score(y_comb_test, y_hybrid_pred),
    'F1 Macro': f1_score(y_comb_test, y_hybrid_pred, average='macro')
}
results_df = pd.DataFrame(results).T
results_df['F1 Macro'].sort_values().plot(kind='barh', ax=axes[1], color='steelblue')
axes[1].set_title('F1 Macro Score Comparison')
axes[1].set_xlim(0, 1)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
```

---

## 📁 Final Project Structure

```
BurnoutPrediction/
├── Dataset/
│   ├── wfh_burnout.csv
│   └── developer_burnout.csv
├── PulseWatch.ipynb          ← All code goes here
├── model_comparison.png      ← Generated chart
└── Notes/
    ├── Project.txt
    └── Refrence link.txt
```

---

## 🎯 What to Present in Your Report

1. **Problem Statement** — Employee burnout prediction using ML
2. **Dataset Description** — Two datasets, their features, and identified issues
3. **Data Preprocessing** — Leakage removal, null handling, SMOTE, scaling
4. **Model Training** — 5-6 models compared with proper metrics
5. **Hybrid Approach** — Ensemble combining multiple models on combined data
6. **Results** — Comparison table + confusion matrix + visualizations
7. **Conclusion** — Which model performed best and why
