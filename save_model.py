import pandas as pd
import numpy as np
import joblib
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score

print("Loading dataset...")
df_dev = pd.read_csv('./Dataset/developer_burnout.csv')
df_dev_clean = df_dev.drop(columns=['stress_level'])
df_dev_clean.dropna(inplace=True)

# Encode Target
le = LabelEncoder()
le.fit(['High', 'Low', 'Medium'])  # Keep consistent encoding
df_dev_clean['target'] = le.transform(df_dev_clean['burnout_level'])

print("Performing Feature Engineering...")
df_dev_fe = df_dev_clean.copy()
df_dev_fe['work_sleep_ratio'] = df_dev_fe['daily_work_hours'] / df_dev_fe['sleep_hours']
df_dev_fe['screen_work_ratio'] = df_dev_fe['screen_time'] / df_dev_fe['daily_work_hours']
df_dev_fe['bug_commit_ratio'] = df_dev_fe['bugs_per_day'] / (df_dev_fe['commits_per_day'] + 1)
df_dev_fe['overwork_flag'] = (df_dev_fe['daily_work_hours'] > 10).astype(int)
df_dev_fe['low_sleep_flag'] = (df_dev_fe['sleep_hours'] < 6).astype(int)
df_dev_fe['meeting_intensity'] = df_dev_fe['meetings_per_day'] / df_dev_fe['daily_work_hours']

X = df_dev_fe.drop(columns=['burnout_level', 'target'])
y = df_dev_fe['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_sm)
X_test_scaled = scaler.transform(X_test)

from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest model (SHAP compatible)...")
# Best params from our tuning for RF
best_model = RandomForestClassifier(
    n_estimators=300, max_depth=20, min_samples_split=2, 
    min_samples_leaf=1, max_features='log2', 
    class_weight='balanced', random_state=42
)

best_model.fit(X_train_scaled, y_train_sm)

y_pred = best_model.predict(X_test_scaled)
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Test F1 Macro: {f1_score(y_test, y_pred, average='macro'):.4f}")

print("Saving model and scaler...")
joblib.dump(best_model, 'burnout_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(list(X.columns), 'feature_names.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("Done! Model artifacts saved for Streamlit.")
