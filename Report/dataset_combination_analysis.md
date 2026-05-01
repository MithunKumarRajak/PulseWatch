# Can We Use Both Datasets for Burnout Prediction?

## ✅ Short Answer: **Yes, but with careful feature alignment**

Both datasets target the **same prediction task** (burnout level: Low / Medium / High) and share several overlapping features. However, their schemas are quite different, so a direct `concat()` won't work — you'll need a feature-mapping strategy.

---

## 📊 Dataset Comparison

| Aspect | `wfh_burnout.csv` | `developer_burnout.csv` |
|---|---|---|
| **Rows** | 1,800 | 7,000 |
| **Columns** | 11 | 12 |
| **Null values** | 0 | 140 per column |
| **Target column** | `burnout_risk` (Low/Medium/High) | `burnout_level` (Low/Medium/High) |
| **Domain** | WFH employees (general) | Software developers |

### Columns Side-by-Side

| Feature Area | `wfh_burnout.csv` | `developer_burnout.csv` | Mappable? |
|---|---|---|---|
| Work hours | `work_hours` | `daily_work_hours` | ✅ Direct |
| Sleep | `sleep_hours` | `sleep_hours` | ✅ Identical |
| Meetings | `meetings_count` | `meetings_per_day` | ✅ Direct |
| Screen time | `screen_time_hours` | `screen_time` | ✅ Direct |
| Burnout target | `burnout_risk` | `burnout_level` | ✅ Same classes |
| Burnout score | `burnout_score` | `stress_level` | ⚠️ Similar concept, different scale |
| Breaks | `breaks_taken` | — | ❌ Only in WFH |
| After-hours | `after_hours_work` | — | ❌ Only in WFH |
| Day type | `day_type` | — | ❌ Only in WFH |
| Task completion | `task_completion_rate` | — | ❌ Only in WFH |
| User ID | `user_id` | — | ❌ Only in WFH |
| Age | — | `age` | ❌ Only in Dev |
| Experience | — | `experience_years` | ❌ Only in Dev |
| Caffeine | — | `caffeine_intake` | ❌ Only in Dev |
| Bugs/day | — | `bugs_per_day` | ❌ Only in Dev |
| Commits/day | — | `commits_per_day` | ❌ Only in Dev |
| Exercise | — | `exercise_hours` | ❌ Only in Dev |

---

## 🔗 Common Features (5 usable + 1 target)

These columns can be directly mapped between datasets:

```
work_hours        ↔  daily_work_hours
sleep_hours       ↔  sleep_hours
meetings_count    ↔  meetings_per_day
screen_time_hours ↔  screen_time
burnout_risk      ↔  burnout_level      (TARGET)
```

---

## 🛠️ Three Strategies to Combine

### Strategy 1: **Common-Feature Merge** (Recommended for simplicity)
Keep only the 4 shared features + target, rename columns, and concatenate.

```python
# Align WFH dataset
df1 = pd.read_csv('./Dataset/wfh_burnout.csv')
df1_aligned = df1[['work_hours', 'sleep_hours', 'meetings_count', 'screen_time_hours']].copy()
df1_aligned.columns = ['work_hours', 'sleep_hours', 'meetings', 'screen_time']
df1_aligned['burnout_level'] = df1['burnout_risk']
df1_aligned['source'] = 'wfh'

# Align Developer dataset
df2 = pd.read_csv('./Dataset/developer_burnout.csv')
df2_aligned = df2[['daily_work_hours', 'sleep_hours', 'meetings_per_day', 'screen_time']].copy()
df2_aligned.columns = ['work_hours', 'sleep_hours', 'meetings', 'screen_time']
df2_aligned['burnout_level'] = df2['burnout_level']
df2_aligned['source'] = 'developer'

# Combine
combined = pd.concat([df1_aligned, df2_aligned], ignore_index=True)
combined.dropna(inplace=True)
```

> **Result**: ~8,660 rows × 5 features + target

### Strategy 2: **Full-Feature Merge with NaN Padding**
Keep ALL columns from both datasets. Missing columns are filled with NaN and handled via imputation or model-native missing support (e.g., XGBoost, LightGBM).

```python
df1['burnout_level'] = df1['burnout_risk']
df2_renamed = df2.rename(columns={
    'daily_work_hours': 'work_hours',
    'meetings_per_day': 'meetings_count',
})
combined = pd.concat([df1, df2_renamed], ignore_index=True)
# Many columns will have NaN — handle with imputation or tree-based models
```

> [!WARNING]
> This introduces **~50% missing values** in dataset-exclusive columns, which can hurt model performance unless you use models like XGBoost that handle NaN natively.

### Strategy 3: **Separate Models + Ensemble** (Best accuracy, more complex)
Train one model per dataset using ALL their features, then combine predictions via a meta-learner or voting.

```
Model_WFH    → trained on all 9 WFH features
Model_Dev    → trained on all 11 Dev features
Meta-Learner → combines both predictions
```

> [!TIP]
> This is the "hybrid" approach that best fits your project title **"PulseWatch: A Hybrid Predictive System"** — it literally uses a hybrid of two models!

---

## 📋 Recommendation

| If you want... | Use Strategy |
|---|---|
| Simple, quick results | **1** (Common Features) |
| Maximum data, accept some noise | **2** (NaN Padding) |
| Best accuracy + matches "Hybrid" title | **3** (Ensemble) |

> [!IMPORTANT]
> **Strategy 1** is the safest starting point. You get a larger combined dataset (8,660 rows vs. 1,800 or 7,000 alone) while keeping the model clean. You can always add Strategy 3 later as the "hybrid" component.

---

## ⚠️ Things to Watch Out For

1. **Scale differences** — `screen_time` ranges may differ between datasets. Always **standardize** (StandardScaler / MinMaxScaler) after merging.
2. **Class imbalance** — Check if Low/Medium/High distributions are similar across both datasets. If not, use stratified sampling or SMOTE.
3. **Domain shift** — WFH employees ≠ developers. Adding a `source` column as a feature lets the model learn domain-specific patterns.
4. **Null handling** — `developer_burnout.csv` has 140 nulls per column — clean before merging.
