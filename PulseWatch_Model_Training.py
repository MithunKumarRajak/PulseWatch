# =============================================================================
# PulseWatch: A Hybrid Predictive System for Employee Burnout Prediction
# Name: Mithun Kumar Rajak | Roll No.: 2023BTCSE010 | JLU ID: JLU08355
# Subject: Machine Learning
# =============================================================================

# %% [markdown]
# # PulseWatch: A Hybrid Predictive System for Employee Burnout Prediction
# Using two datasets: WFH Employee Burnout + Developer Burnout

# %% Cell 1 - Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, f1_score, confusion_matrix,
                             balanced_accuracy_score, accuracy_score, ConfusionMatrixDisplay)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

print("All libraries imported successfully!")

# %% Cell 2 - Load Datasets
df_wfh = pd.read_csv('./Dataset/wfh_burnout.csv')
df_dev = pd.read_csv('./Dataset/developer_burnout.csv')

print("=== WFH Dataset ===")
print(f"Shape: {df_wfh.shape}")
print(df_wfh.head(3))

print("\n=== Developer Dataset ===")
print(f"Shape: {df_dev.shape}")
print(df_dev.head(3))

# %% Cell 3 - Data Quality Check
print("=== WFH Null Values ===")
print(df_wfh.isnull().sum())
print(f"\n=== Developer Null Values ===")
print(df_dev.isnull().sum())

print(f"\nWFH Duplicates: {df_wfh.duplicated().sum()}")
print(f"Developer Duplicates: {df_dev.duplicated().sum()}")

print(f"\nWFH Target Distribution:")
print(df_wfh['burnout_risk'].value_counts())
print(f"\nDeveloper Target Distribution:")
print(df_dev['burnout_level'].value_counts())

# %% Cell 4 - Identify Data Leakage
print("=== DATA LEAKAGE CHECK ===\n")

print("WFH: burnout_score ranges per burnout_risk:")
for level in ['Low', 'Medium', 'High']:
    subset = df_wfh[df_wfh['burnout_risk'] == level]['burnout_score']
    print(f"  {level}: {subset.min():.2f} - {subset.max():.2f}")

print("\nDeveloper: stress_level ranges per burnout_level:")
for level in ['Low', 'Medium', 'High']:
    subset = df_dev[df_dev['burnout_level'] == level]['stress_level']
    print(f"  {level}: {subset.min():.2f} - {subset.max():.2f}")

print("\n[WARNING] Target is derived from score/stress columns - MUST DROP to avoid leakage!")

# %% Cell 5 - Data Preprocessing
# --- WFH Dataset ---
df_wfh_clean = df_wfh.drop(columns=['user_id', 'burnout_score', 'task_completion_rate'])
df_wfh_clean['day_type'] = df_wfh_clean['day_type'].map({'Weekday': 1, 'Weekend': 0})
df_wfh_clean.rename(columns={'burnout_risk': 'burnout_level'}, inplace=True)

# --- Developer Dataset ---
df_dev_clean = df_dev.drop(columns=['stress_level'])
df_dev_clean.dropna(inplace=True)

print(f"WFH after cleaning: {df_wfh_clean.shape}")
print(f"Developer after cleaning: {df_dev_clean.shape}")

# --- Encode Target ---
le = LabelEncoder()
le.fit(['High', 'Low', 'Medium'])  # alphabetical order

df_wfh_clean['target'] = le.transform(df_wfh_clean['burnout_level'])
df_dev_clean['target'] = le.transform(df_dev_clean['burnout_level'])

print(f"\nLabel Encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# %% Cell 6 - EDA Visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Exploratory Data Analysis', fontsize=16, fontweight='bold')

# Target distribution comparison
for i, (df_plot, name) in enumerate([(df_wfh_clean, 'WFH'), (df_dev_clean, 'Developer')]):
    df_plot['burnout_level'].value_counts().plot(kind='bar', ax=axes[0, i], color=['#2ecc71','#f39c12','#e74c3c'])
    axes[0, i].set_title(f'{name} - Target Distribution')
    axes[0, i].set_ylabel('Count')
    axes[0, i].tick_params(axis='x', rotation=0)

# Correlation heatmap - Developer dataset
numeric_dev = df_dev_clean.select_dtypes(include=[np.number]).drop(columns=['target'])
sns.heatmap(numeric_dev.corr(), annot=True, fmt='.2f', cmap='coolwarm',
            ax=axes[0, 2], vmin=-1, vmax=1, linewidths=0.5, annot_kws={'size': 7})
axes[0, 2].set_title('Developer - Feature Correlations')

# Key feature distributions
sns.boxplot(data=df_dev_clean, x='burnout_level', y='daily_work_hours', ax=axes[1, 0],
            palette=['#2ecc71','#f39c12','#e74c3c'], order=['Low', 'Medium', 'High'])
axes[1, 0].set_title('Work Hours by Burnout Level (Dev)')

sns.boxplot(data=df_dev_clean, x='burnout_level', y='sleep_hours', ax=axes[1, 1],
            palette=['#2ecc71','#f39c12','#e74c3c'], order=['Low', 'Medium', 'High'])
axes[1, 1].set_title('Sleep Hours by Burnout Level (Dev)')

sns.boxplot(data=df_dev_clean, x='burnout_level', y='screen_time', ax=axes[1, 2],
            palette=['#2ecc71','#f39c12','#e74c3c'], order=['Low', 'Medium', 'High'])
axes[1, 2].set_title('Screen Time by Burnout Level (Dev)')

plt.tight_layout()
plt.savefig('eda_plots.png', dpi=150, bbox_inches='tight')
plt.show()

# %% Cell 7 - Prepare Training Data (Developer Dataset)
X_dev = df_dev_clean.drop(columns=['burnout_level', 'target'])
y_dev = df_dev_clean['target']

X_dev_train, X_dev_test, y_dev_train, y_dev_test = train_test_split(
    X_dev, y_dev, test_size=0.2, random_state=42, stratify=y_dev
)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_dev_train_sm, y_dev_train_sm = smote.fit_resample(X_dev_train, y_dev_train)

print(f"Before SMOTE: {pd.Series(y_dev_train).value_counts().to_dict()}")
print(f"After SMOTE:  {pd.Series(y_dev_train_sm).value_counts().to_dict()}")

# Scale features
scaler_dev = StandardScaler()
X_dev_train_scaled = scaler_dev.fit_transform(X_dev_train_sm)
X_dev_test_scaled = scaler_dev.transform(X_dev_test)

# %% Cell 8 - Train & Compare Models (Developer Dataset)
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'SVM (RBF)': SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                             eval_metric='mlogloss', random_state=42),
}

results = {}
trained_models = {}

for name, model in models.items():
    # Tree-based models don't need scaling
    if name in ['Random Forest', 'XGBoost']:
        model.fit(X_dev_train_sm, y_dev_train_sm)
        y_pred = model.predict(X_dev_test)
    else:
        model.fit(X_dev_train_scaled, y_dev_train_sm)
        y_pred = model.predict(X_dev_test_scaled)

    acc = accuracy_score(y_dev_test, y_pred)
    bal_acc = balanced_accuracy_score(y_dev_test, y_pred)
    f1 = f1_score(y_dev_test, y_pred, average='macro')

    results[name] = {'Accuracy': acc, 'Balanced Accuracy': bal_acc, 'F1 Macro': f1}
    trained_models[name] = model

    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(classification_report(y_dev_test, y_pred, target_names=le.classes_))

# Results table
results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('F1 Macro', ascending=False)
print("\n[RESULTS] DEVELOPER DATASET - MODEL COMPARISON")
print(results_df.to_string())

# %% Cell 9 - Prepare Combined Dataset for Hybrid Model
shared_wfh = ['work_hours', 'sleep_hours', 'meetings_count', 'screen_time_hours']
shared_dev = ['daily_work_hours', 'sleep_hours', 'meetings_per_day', 'screen_time']
common_names = ['work_hours', 'sleep_hours', 'meetings', 'screen_time']

X_wfh_common = df_wfh_clean[shared_wfh].copy()
X_wfh_common.columns = common_names
y_wfh_all = df_wfh_clean['target']

X_dev_common = df_dev_clean[shared_dev].copy()
X_dev_common.columns = common_names
y_dev_all = df_dev_clean['target']

# Add source indicator
X_wfh_common['source'] = 0  # WFH
X_dev_common['source'] = 1  # Developer

# Combine
X_combined = pd.concat([X_wfh_common, X_dev_common], ignore_index=True)
y_combined = pd.concat([y_wfh_all, y_dev_all], ignore_index=True)

print(f"Combined dataset: {X_combined.shape}")
print(f"Target distribution:\n{y_combined.value_counts()}")

X_comb_train, X_comb_test, y_comb_train, y_comb_test = train_test_split(
    X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined
)

X_comb_train_sm, y_comb_train_sm = smote.fit_resample(X_comb_train, y_comb_train)

scaler_comb = StandardScaler()
X_comb_train_scaled = scaler_comb.fit_transform(X_comb_train_sm)
X_comb_test_scaled = scaler_comb.transform(X_comb_test)

print(f"\nAfter SMOTE: {pd.Series(y_comb_train_sm).value_counts().to_dict()}")

# %% Cell 10 - Hybrid Ensemble Model
hybrid_model = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               eval_metric='mlogloss', random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)),
        ('svm', SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42)),
    ],
    voting='soft'
)

hybrid_model.fit(X_comb_train_scaled, y_comb_train_sm)
y_hybrid_pred = hybrid_model.predict(X_comb_test_scaled)

hybrid_acc = accuracy_score(y_comb_test, y_hybrid_pred)
hybrid_bal = balanced_accuracy_score(y_comb_test, y_hybrid_pred)
hybrid_f1 = f1_score(y_comb_test, y_hybrid_pred, average='macro')

print("=" * 55)
print("  [STAR] HYBRID ENSEMBLE (Combined Dataset)")
print("=" * 55)
print(classification_report(y_comb_test, y_hybrid_pred, target_names=le.classes_))
print(f"Accuracy:          {hybrid_acc:.4f}")
print(f"Balanced Accuracy: {hybrid_bal:.4f}")
print(f"F1 Macro:          {hybrid_f1:.4f}")

# %% Cell 11 - Cross Validation
print("=== 5-Fold Stratified Cross-Validation (Hybrid Model) ===\n")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(hybrid_model, X_comb_train_scaled, y_comb_train_sm,
                            cv=cv, scoring='f1_macro')
print(f"CV F1 Macro Scores: {cv_scores.round(4)}")
print(f"Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# %% Cell 12 - Final Comparison & Visualizations
results['Hybrid Ensemble'] = {'Accuracy': hybrid_acc, 'Balanced Accuracy': hybrid_bal, 'F1 Macro': hybrid_f1}
final_df = pd.DataFrame(results).T.sort_values('F1 Macro', ascending=False).round(4)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('PulseWatch - Model Performance Summary', fontsize=16, fontweight='bold')

# Bar chart - F1 Macro
colors = ['#e74c3c' if name == 'Hybrid Ensemble' else '#3498db' for name in final_df.index]
final_df['F1 Macro'].plot(kind='barh', ax=axes[0], color=colors)
axes[0].set_title('F1 Macro Score')
axes[0].set_xlim(0, 1)
for i, v in enumerate(final_df['F1 Macro']):
    axes[0].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)

# Confusion Matrix - Hybrid
ConfusionMatrixDisplay.from_predictions(
    y_comb_test, y_hybrid_pred,
    display_labels=le.classes_, cmap='Blues', ax=axes[1]
)
axes[1].set_title('Hybrid Ensemble - Confusion Matrix')

# Grouped bar chart - All metrics
final_df.plot(kind='bar', ax=axes[2], color=['#3498db', '#2ecc71', '#e74c3c'])
axes[2].set_title('All Metrics Comparison')
axes[2].set_ylim(0, 1)
axes[2].legend(loc='lower right')
axes[2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Print final table
print("\n[RESULTS] FINAL MODEL COMPARISON TABLE")
print("=" * 65)
print(final_df.to_string())
print("=" * 65)

# %% Cell 13 - Feature Importance (XGBoost)
best_xgb = trained_models['XGBoost']
feat_imp = pd.Series(best_xgb.feature_importances_, index=X_dev.columns)
feat_imp = feat_imp.sort_values(ascending=True)

fig, ax = plt.subplots(figsize=(10, 6))
feat_imp.plot(kind='barh', color='#3498db', ax=ax)
ax.set_title('Feature Importance (XGBoost - Developer Dataset)', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n--- Base training complete. Starting upgrades... ---")

# =========================================================================
# UPGRADE 1: Feature Engineering
# =========================================================================

# %% Cell 14 - Feature Engineering (Developer Dataset)
print("\n" + "=" * 65)
print("  UPGRADE 1: Feature Engineering")
print("=" * 65)

df_dev_fe = df_dev_clean.copy()

# New engineered features
df_dev_fe['work_sleep_ratio'] = df_dev_fe['daily_work_hours'] / df_dev_fe['sleep_hours']
df_dev_fe['screen_work_ratio'] = df_dev_fe['screen_time'] / df_dev_fe['daily_work_hours']
df_dev_fe['bug_commit_ratio'] = df_dev_fe['bugs_per_day'] / (df_dev_fe['commits_per_day'] + 1)
df_dev_fe['overwork_flag'] = (df_dev_fe['daily_work_hours'] > 10).astype(int)
df_dev_fe['low_sleep_flag'] = (df_dev_fe['sleep_hours'] < 6).astype(int)
df_dev_fe['meeting_intensity'] = df_dev_fe['meetings_per_day'] / df_dev_fe['daily_work_hours']

print("New features added:")
new_feats = ['work_sleep_ratio', 'screen_work_ratio', 'bug_commit_ratio',
             'overwork_flag', 'low_sleep_flag', 'meeting_intensity']
for f in new_feats:
    print(f"  {f}: min={df_dev_fe[f].min():.2f}, max={df_dev_fe[f].max():.2f}")

# Prepare FE data
X_dev_fe = df_dev_fe.drop(columns=['burnout_level', 'target'])
y_dev_fe = df_dev_fe['target']

X_fe_train, X_fe_test, y_fe_train, y_fe_test = train_test_split(
    X_dev_fe, y_dev_fe, test_size=0.2, random_state=42, stratify=y_dev_fe
)

X_fe_train_sm, y_fe_train_sm = smote.fit_resample(X_fe_train, y_fe_train)

scaler_fe = StandardScaler()
X_fe_train_scaled = scaler_fe.fit_transform(X_fe_train_sm)
X_fe_test_scaled = scaler_fe.transform(X_fe_test)

# Quick test with Logistic Regression (previous best)
from sklearn.linear_model import LogisticRegression as LR
lr_fe = LR(max_iter=1000, random_state=42)
lr_fe.fit(X_fe_train_scaled, y_fe_train_sm)
y_fe_pred = lr_fe.predict(X_fe_test_scaled)

fe_f1 = f1_score(y_fe_test, y_fe_pred, average='macro')
fe_bal = balanced_accuracy_score(y_fe_test, y_fe_pred)
print(f"\nLogistic Regression WITH Feature Engineering:")
print(f"  F1 Macro:          {fe_f1:.4f} (was 0.7918)")
print(f"  Balanced Accuracy: {fe_bal:.4f} (was 0.8122)")
print(f"  Improvement:       {(fe_f1 - 0.7918)*100:+.2f}% F1")

# =========================================================================
# UPGRADE 2: Hyperparameter Tuning
# =========================================================================

# %% Cell 15 - Hyperparameter Tuning (RandomizedSearchCV)
from sklearn.model_selection import RandomizedSearchCV

print("\n" + "=" * 65)
print("  UPGRADE 2: Hyperparameter Tuning (RandomizedSearchCV)")
print("=" * 65)

# --- Random Forest Tuning ---
print("\n[1/3] Tuning Random Forest...")
rf_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [5, 10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample'],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    rf_params, n_iter=30, cv=3, scoring='f1_macro',
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_fe_train_sm, y_fe_train_sm)
print(f"  Best RF params: {rf_search.best_params_}")
print(f"  Best RF CV F1:  {rf_search.best_score_:.4f}")

# --- XGBoost Tuning ---
print("\n[2/3] Tuning XGBoost...")
xgb_params = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5],
}

xgb_search = RandomizedSearchCV(
    XGBClassifier(eval_metric='mlogloss', random_state=42),
    xgb_params, n_iter=30, cv=3, scoring='f1_macro',
    random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_fe_train_sm, y_fe_train_sm)
print(f"  Best XGB params: {xgb_search.best_params_}")
print(f"  Best XGB CV F1:  {xgb_search.best_score_:.4f}")

# --- SVM Tuning ---
print("\n[3/3] Tuning SVM...")
svm_params = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': ['rbf'],
    'class_weight': ['balanced'],
}

svm_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    svm_params, n_iter=15, cv=3, scoring='f1_macro',
    random_state=42, n_jobs=-1, verbose=0
)
svm_search.fit(X_fe_train_scaled, y_fe_train_sm)
print(f"  Best SVM params: {svm_search.best_params_}")
print(f"  Best SVM CV F1:  {svm_search.best_score_:.4f}")

# Evaluate tuned models on test set
tuned_results = {}
for name, searcher, X_test_use in [
    ('RF (Tuned)', rf_search, X_fe_test),
    ('XGBoost (Tuned)', xgb_search, X_fe_test),
    ('SVM (Tuned)', svm_search, X_fe_test_scaled),
]:
    y_pred_t = searcher.predict(X_test_use)
    t_f1 = f1_score(y_fe_test, y_pred_t, average='macro')
    t_bal = balanced_accuracy_score(y_fe_test, y_pred_t)
    t_acc = accuracy_score(y_fe_test, y_pred_t)
    tuned_results[name] = {'Accuracy': t_acc, 'Balanced Accuracy': t_bal, 'F1 Macro': t_f1}
    print(f"\n  {name}: Acc={t_acc:.4f}, Bal-Acc={t_bal:.4f}, F1={t_f1:.4f}")

# =========================================================================
# UPGRADE 3: Stacking Ensemble
# =========================================================================

# %% Cell 16 - Stacking Ensemble (Hybrid Fix)
from sklearn.ensemble import StackingClassifier

print("\n" + "=" * 65)
print("  UPGRADE 3: Stacking Ensemble (Improved Hybrid)")
print("=" * 65)

stacking_model = StackingClassifier(
    estimators=[
        ('rf', rf_search.best_estimator_),
        ('xgb', xgb_search.best_estimator_),
        ('svm', svm_search.best_estimator_),
        ('lr', LogisticRegression(max_iter=1000, random_state=42)),
    ],
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5, n_jobs=-1, passthrough=False
)

# Train on FE developer data (full features, not just shared)
stacking_model.fit(X_fe_train_scaled, y_fe_train_sm)
y_stack_pred = stacking_model.predict(X_fe_test_scaled)

stack_acc = accuracy_score(y_fe_test, y_stack_pred)
stack_bal = balanced_accuracy_score(y_fe_test, y_stack_pred)
stack_f1 = f1_score(y_fe_test, y_stack_pred, average='macro')

print(classification_report(y_fe_test, y_stack_pred, target_names=le.classes_))
print(f"Accuracy:          {stack_acc:.4f}")
print(f"Balanced Accuracy: {stack_bal:.4f}")
print(f"F1 Macro:          {stack_f1:.4f}")

tuned_results['Stacking Ensemble'] = {
    'Accuracy': stack_acc, 'Balanced Accuracy': stack_bal, 'F1 Macro': stack_f1
}

# =========================================================================
# FULL COMPARISON: Before vs After Upgrades
# =========================================================================

# %% Cell 17 - Before vs After Comparison
print("\n" + "=" * 65)
print("  BEFORE vs AFTER UPGRADES")
print("=" * 65)

# Merge old results with tuned results
all_results = {**results, **tuned_results}
all_df = pd.DataFrame(all_results).T.sort_values('F1 Macro', ascending=False).round(4)
print(all_df.to_string())

fig, ax = plt.subplots(figsize=(12, 7))
colors = []
for name in all_df.index:
    if 'Tuned' in name or 'Stacking' in name:
        colors.append('#e74c3c')
    elif name == 'Hybrid Ensemble':
        colors.append('#95a5a6')
    else:
        colors.append('#3498db')

all_df['F1 Macro'].plot(kind='barh', ax=ax, color=colors)
ax.set_title('PulseWatch - Before vs After Upgrades (F1 Macro)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1)
for i, v in enumerate(all_df['F1 Macro']):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)
ax.axvline(x=0.7918, color='gray', linestyle='--', alpha=0.5, label='Previous best (LR: 0.792)')
ax.legend()
plt.tight_layout()
plt.savefig('upgrade_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# =========================================================================
# UPGRADE 4: ROC Curves + Learning Curves
# =========================================================================

# %% Cell 18 - ROC Curves & Learning Curves
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve

print("\n" + "=" * 65)
print("  UPGRADE 4: ROC Curves & Learning Curves")
print("=" * 65)

# --- ROC Curves (One-vs-Rest) ---
y_test_bin = label_binarize(y_fe_test, classes=[0, 1, 2])
n_classes = 3

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Get probability predictions from stacking model
y_stack_proba = stacking_model.predict_proba(X_fe_test_scaled)

for i, class_name in enumerate(le.classes_):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_stack_proba[:, i])
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, linewidth=2, label=f'{class_name} (AUC = {roc_auc:.3f})')

axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].set_title('ROC Curves - Stacking Ensemble (One-vs-Rest)', fontsize=12, fontweight='bold')
axes[0].legend(loc='lower right')
axes[0].grid(alpha=0.3)

# --- Learning Curve ---
train_sizes, train_scores, val_scores = learning_curve(
    xgb_search.best_estimator_, X_fe_train_sm, y_fe_train_sm,
    train_sizes=np.linspace(0.1, 1.0, 8), cv=5, scoring='f1_macro',
    n_jobs=-1, random_state=42
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

axes[1].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='#3498db')
axes[1].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='#e74c3c')
axes[1].plot(train_sizes, train_mean, 'o-', color='#3498db', label='Training Score')
axes[1].plot(train_sizes, val_mean, 'o-', color='#e74c3c', label='Validation Score')
axes[1].set_xlabel('Training Set Size')
axes[1].set_ylabel('F1 Macro Score')
axes[1].set_title('Learning Curve - XGBoost (Tuned)', fontsize=12, fontweight='bold')
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig('roc_learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("ROC and Learning curves saved!")

# =========================================================================
# UPGRADE 5: SHAP Explainability
# =========================================================================

# %% Cell 19 - SHAP Explainability
import shap

print("\n" + "=" * 65)
print("  UPGRADE 5: SHAP Explainability")
print("=" * 65)

# Use the tuned Random Forest for SHAP (fully compatible with TreeExplainer)
best_rf_tuned = rf_search.best_estimator_

# Create SHAP explainer
explainer = shap.TreeExplainer(best_rf_tuned)
X_fe_test_df = pd.DataFrame(X_fe_test, columns=X_dev_fe.columns)
shap_values = explainer.shap_values(X_fe_test_df)

# shap_values for RF might be a 3D array of shape (n_samples, n_features, n_classes)
# We convert it to a list of 2D arrays for multi-class summary_plot
if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
    shap_values_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
else:
    shap_values_list = shap_values

print(f"SHAP values computed: {len(shap_values_list)} classes, shape per class: {shap_values_list[0].shape}")

# --- SHAP Summary Plot (Bar - all classes) ---
fig1, ax1 = plt.subplots(figsize=(10, 8))
plt.sca(ax1)
shap.summary_plot(shap_values_list, X_fe_test_df, plot_type='bar',
                  class_names=le.classes_, show=False, max_display=16)
ax1.set_title('SHAP Feature Importance (All Classes)', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_bar.png', dpi=150, bbox_inches='tight')
plt.show()

# --- SHAP Summary Plot (Dot - High burnout class) ---
fig2, ax2 = plt.subplots(figsize=(10, 8))
plt.sca(ax2)
high_idx = list(le.classes_).index('High')
shap.summary_plot(shap_values_list[high_idx], X_fe_test_df, show=False, max_display=16)
ax2.set_title('SHAP Values - High Burnout Class', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_dot.png', dpi=150, bbox_inches='tight')
plt.show()

print("SHAP analysis saved!")

# =========================================================================
# FINAL SUMMARY
# =========================================================================

# %% Cell 20 - Final Summary
print("\n" + "=" * 65)
print("  PULSEWATCH - FINAL RESULTS SUMMARY")
print("=" * 65)
print(all_df.to_string())
print("=" * 65)

best_model_name = all_df['F1 Macro'].idxmax()
best_f1 = all_df.loc[best_model_name, 'F1 Macro']
print(f"\nBest Model: {best_model_name}")
print(f"Best F1 Macro: {best_f1:.4f}")
print(f"\nImprovement over baseline LR (0.7918): {(best_f1 - 0.7918)*100:+.2f}%")
print("\nGenerated files:")
print("  - eda_plots.png")
print("  - model_comparison.png")
print("  - feature_importance.png")
print("  - upgrade_comparison.png")
print("  - roc_learning_curves.png")
print("  - shap_analysis.png")
print("\n[DONE] PulseWatch Training + All Upgrades Complete!")
