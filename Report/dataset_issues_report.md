# 🔍 Dataset Issues Report — PulseWatch

## Summary of Issues Found

| # | Issue | Dataset | Severity |
|---|---|---|---|
| 1 | **Severe class imbalance** | `wfh_burnout.csv` | 🔴 Critical |
| 2 | **1,680 missing values** | `developer_burnout.csv` | 🟡 Moderate |
| 3 | **`burnout_score` > 100** (37 rows) | `wfh_burnout.csv` | 🟡 Moderate |
| 4 | **`task_completion_rate` > 100%** (10 rows) | `wfh_burnout.csv` | 🟡 Moderate |
| 5 | **Target is directly derivable from score** | Both | 🔴 Critical (Data Leakage) |
| 6 | **Very weak feature correlations** | `wfh_burnout.csv` | 🟠 Important |
| 7 | **Class distribution mismatch** between datasets | Both | 🟠 Important |

---

## 🔴 Issue 1: Severe Class Imbalance (WFH Dataset)

```
burnout_risk distribution:
  Low     → 1527 rows (84.8%)
  Medium  →  253 rows (14.1%)
  High    →   20 rows ( 1.1%)  ← ONLY 20 SAMPLES!
```

> [!CAUTION]
> With only **20 "High" burnout samples** out of 1,800 rows, any model will be heavily biased toward predicting "Low". This is a classic imbalanced classification problem.

**Fix options:**
- Use **SMOTE** or **ADASYN** to oversample the minority classes
- Use **class_weight='balanced'** in sklearn models
- Combine with the developer dataset (which has a better distribution)
- Use evaluation metrics like **F1-score (macro)** or **balanced accuracy** instead of plain accuracy

---

## 🔴 Issue 5: Data Leakage — Target Derived from Score

This is the **most critical** issue. The burnout target is clearly a **threshold-based derivation** of the score column:

### WFH Dataset (`burnout_risk` from `burnout_score`)
```
Low    → burnout_score:   2.50 –  69.92
Medium → burnout_score:  70.01 – 109.99
High   → burnout_score: 110.22 – 143.92
```

### Developer Dataset (`burnout_level` from `stress_level`)
```
Low    → stress_level:   0.00 –  35.00
Medium → stress_level:  35.02 –  69.97
High   → stress_level:  70.01 – 100.00
```

> [!CAUTION]
> The target labels are **deterministically computed** from `burnout_score` / `stress_level`. If you include these columns as features, your model will achieve near-perfect accuracy — but it's **fake performance**. The model is just re-learning the threshold rules.

**Fix:** You have two options:
1. **Drop** `burnout_score` and `stress_level` from features (recommended)
2. **Use** `burnout_score` / `stress_level` as the target instead (regression task)

**Correlation evidence:**
```
wfh:       burnout_score ↔ burnout_risk       = 0.7763
developer: stress_level  ↔ burnout_level      = 0.9123
```

---

## 🟡 Issue 2: Missing Values (Developer Dataset)

```
Every column has exactly 140 null values
Total null cells: 1,680
Rows with ANY null: 1,510
Avg nulls per null-row: 1.1 / 12 columns
```

The nulls are **evenly spread** (140 per column) and scattered across 1,510 different rows (not concentrated in the same rows). This looks like a **random dropout pattern** — possibly synthetic data generation.

**Fix options:**
- **Drop rows** with nulls (loses ~1,510 rows → 5,490 remain, still large enough)
- **Impute** with median (for numeric) — safer for skewed data
- Use **models that handle NaN natively** (XGBoost, LightGBM)

---

## 🟡 Issue 3: `burnout_score` Exceeds 100

```
37 rows have burnout_score > 100 (max = 143.92)
```

If the score is meant to be a 0–100 scale, these are **invalid data points**. However, examining the data shows these correspond to "High" burnout level, so the scale may simply be 0–150.

**Fix:** Clarify the intended scale. If 0–100, clip the values:
```python
df['burnout_score'] = df['burnout_score'].clip(upper=100)
```

---

## 🟡 Issue 4: `task_completion_rate` Exceeds 100%

```
10 rows have task_completion_rate > 100 (max = 107.2)
```

A completion rate above 100% is **logically impossible** (unless it means "exceeded target").

**Fix:**
```python
df['task_completion_rate'] = df['task_completion_rate'].clip(upper=100)
```

---

## 🟠 Issue 6: Weak Feature Correlations (WFH Dataset)

After removing the leaky `burnout_score`, the remaining features have **very weak correlation** with the target:

```
Feature                   Correlation with burnout_risk
─────────────────────────────────────────────────────
task_completion_rate      -0.7072   ← Also likely leaky
burnout_score              0.7763   ← LEAKY - must drop
screen_time_hours          0.0628   ← Very weak
work_hours                 0.0601   ← Very weak
meetings_count             0.0364   ← Very weak
breaks_taken               0.0064   ← Almost zero
sleep_hours                0.0051   ← Almost zero
after_hours_work          -0.0034   ← Almost zero
```

> [!WARNING]
> Once you remove `burnout_score` and `task_completion_rate` (both likely leaky), the remaining WFH features have **correlations below 0.07** with the target. This means the WFH dataset alone may produce a **weak model**.

**In contrast**, the developer dataset has much better feature-target relationships:
```
stress_level              0.9123   ← LEAKY - must drop
daily_work_hours          0.5465   ← Strong
screen_time               0.5002   ← Strong
bugs_per_day              0.4575   ← Moderate
meetings_per_day          0.3202   ← Moderate
sleep_hours              -0.2165   ← Weak-Moderate
caffeine_intake           0.1803   ← Weak
```

---

## 🟠 Issue 7: Class Distribution Mismatch

The two datasets have **very different** class distributions:

```
              Low     Medium    High
─────────────────────────────────────
WFH:         84.8%    14.1%     1.1%
Developer:   23.2%    50.8%    26.0%
```

Directly merging them will create a biased dataset where the "source" of the data strongly predicts the label.

**Fix:** Use stratified sampling or add a `source` feature column.

---

## ✅ Recommended Preprocessing Pipeline

```python
# 1. Drop leaky columns
df1.drop(columns=['burnout_score', 'task_completion_rate', 'user_id'], inplace=True)
df2.drop(columns=['stress_level'], inplace=True)

# 2. Handle missing values in developer dataset
df2.dropna(inplace=True)  # or use imputation

# 3. Clip anomalous values (if using WFH dataset standalone)
# df1['task_completion_rate'] = df1['task_completion_rate'].clip(upper=100)

# 4. Handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# 5. Use proper evaluation metrics
from sklearn.metrics import classification_report, f1_score
# Use f1_score(y_test, y_pred, average='macro') instead of accuracy
```

> [!IMPORTANT]
> The **data leakage** (Issue 5) and **class imbalance** (Issue 1) are the two most critical problems. If you don't address them, your results will be misleading — either artificially inflated accuracy or a model that only predicts "Low".
