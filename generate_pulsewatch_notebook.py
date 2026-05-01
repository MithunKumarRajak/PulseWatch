import json
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "PulseWatch.ipynb"


def to_source(text: str) -> list[str]:
    text = dedent(text).strip("\n")
    if not text:
        return []
    return [line + "\n" for line in text.splitlines()]


def markdown_cell(text: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": to_source(text),
    }


def code_cell(text: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": to_source(text),
    }


cells = [
    markdown_cell(
        """
        # PulseWatch

        **Project:** A Hybrid Predictive System for Employee Burnout Prediction

        This notebook is the notebook version of `PulseWatch_Model_Training.py`, expanded into a cleaner ML workflow with:

        - dataset loading and auditing
        - EDA for both burnout datasets
        - leakage checks
        - baseline model comparison
        - shared-feature hybrid training
        - feature engineering aligned with the Streamlit app
        - hyperparameter tuning and stacking
        - SHAP-friendly deployment artifact export
        """
    ),
    markdown_cell(
        """
        ## Notebook Notes

        The repository currently uses the developer dataset and engineered features for the deployed Streamlit classifier.
        This notebook keeps the broader hybrid analysis from the training script, but it also includes a final export cell that saves artifacts in the same format expected by `app.py`.
        """
    ),
    code_cell(
        """
        import warnings
        from pathlib import Path

        import joblib
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns

        from imblearn.over_sampling import SMOTE
        from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            accuracy_score,
            balanced_accuracy_score,
            classification_report,
            f1_score,
        )
        from sklearn.model_selection import (
            RandomizedSearchCV,
            StratifiedKFold,
            cross_val_score,
            train_test_split,
        )
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
        from sklearn.svm import SVC
        from xgboost import XGBClassifier

        from burnout_recommender import MODEL_FEATURES, add_engineered_features

        warnings.filterwarnings("ignore")
        sns.set_theme(style="whitegrid", context="notebook")

        RANDOM_STATE = 42
        PLOTS_DIR = Path("Plots")
        PLOTS_DIR.mkdir(exist_ok=True)

        try:
            import shap

            SHAP_AVAILABLE = True
        except Exception:
            shap = None
            SHAP_AVAILABLE = False

        print("Setup complete")
        print(f"SHAP available: {SHAP_AVAILABLE}")
        """
    ),
    code_cell(
        """
        df_wfh = pd.read_csv("Dataset/wfh_burnout.csv")
        df_dev = pd.read_csv("Dataset/developer_burnout.csv")

        print("WFH shape:", df_wfh.shape)
        display(df_wfh.head())
        print("Developer shape:", df_dev.shape)
        display(df_dev.head())
        """
    ),
    code_cell(
        """
        def audit_frame(df: pd.DataFrame, name: str, target_col: str) -> pd.DataFrame:
            summary = pd.DataFrame(
                {
                    "dtype": df.dtypes.astype(str),
                    "missing": df.isna().sum(),
                    "missing_pct": (df.isna().mean() * 100).round(2),
                    "n_unique": df.nunique(),
                }
            )
            print(f"=== {name} dataset audit ===")
            print("Shape:", df.shape)
            print("Duplicates:", df.duplicated().sum())
            print("Target distribution:")
            print(df[target_col].value_counts(dropna=False))
            return summary


        wfh_audit = audit_frame(df_wfh, "WFH", "burnout_risk")
        dev_audit = audit_frame(df_dev, "Developer", "burnout_level")

        display(wfh_audit)
        display(dev_audit)
        """
    ),
    markdown_cell(
        """
        ## Leakage Check

        Both datasets include a score-like signal that is directly tied to the label:

        - `burnout_score` in the WFH dataset
        - `stress_level` in the developer dataset

        Those fields should not be used as predictive features because they make the task unrealistically easy and would inflate reported performance.
        """
    ),
    code_cell(
        """
        print("WFH burnout_score ranges by burnout_risk")
        for level in ["Low", "Medium", "High"]:
            subset = df_wfh.loc[df_wfh["burnout_risk"] == level, "burnout_score"]
            print(f"  {level:6s}: min={subset.min():.2f}, max={subset.max():.2f}")

        print("\\nDeveloper stress_level ranges by burnout_level")
        for level in ["Low", "Medium", "High"]:
            subset = df_dev.loc[df_dev["burnout_level"] == level, "stress_level"]
            print(f"  {level:6s}: min={subset.min():.2f}, max={subset.max():.2f}")
        """
    ),
    code_cell(
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle("PulseWatch EDA", fontsize=16, fontweight="bold")

        df_wfh["burnout_risk"].value_counts().reindex(["Low", "Medium", "High"]).plot(
            kind="bar", color=["#2e8b57", "#f0ad4e", "#c0392b"], ax=axes[0, 0], title="WFH target distribution"
        )
        df_dev["burnout_level"].value_counts().reindex(["Low", "Medium", "High"]).plot(
            kind="bar", color=["#2e8b57", "#f0ad4e", "#c0392b"], ax=axes[0, 1], title="Developer target distribution"
        )

        sns.heatmap(
            df_dev.select_dtypes(include=[np.number]).corr(),
            cmap="coolwarm",
            center=0,
            annot=True,
            fmt=".2f",
            ax=axes[0, 2],
        )
        axes[0, 2].set_title("Developer correlation heatmap")

        sns.boxplot(
            data=df_dev,
            x="burnout_level",
            y="daily_work_hours",
            order=["Low", "Medium", "High"],
            palette=["#2e8b57", "#f0ad4e", "#c0392b"],
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Daily work hours vs burnout")

        sns.boxplot(
            data=df_dev,
            x="burnout_level",
            y="sleep_hours",
            order=["Low", "Medium", "High"],
            palette=["#2e8b57", "#f0ad4e", "#c0392b"],
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Sleep hours vs burnout")

        sns.boxplot(
            data=df_dev,
            x="burnout_level",
            y="screen_time",
            order=["Low", "Medium", "High"],
            palette=["#2e8b57", "#f0ad4e", "#c0392b"],
            ax=axes[1, 2],
        )
        axes[1, 2].set_title("Screen time vs burnout")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "pulsewatch_eda.png", dpi=160, bbox_inches="tight")
        plt.show()
        """
    ),
    code_cell(
        """
        def prepare_wfh_dataset(frame: pd.DataFrame, label_encoder: LabelEncoder) -> pd.DataFrame:
            out = frame.copy()
            out = out.drop(columns=["user_id", "burnout_score", "task_completion_rate"])
            out["day_type"] = out["day_type"].map({"Weekday": 1, "Weekend": 0})
            out = out.rename(columns={"burnout_risk": "burnout_level"})
            out["target"] = label_encoder.transform(out["burnout_level"])
            return out


        def prepare_dev_dataset(frame: pd.DataFrame, label_encoder: LabelEncoder) -> pd.DataFrame:
            out = frame.copy()
            out = out.drop(columns=["stress_level"]).dropna().reset_index(drop=True)
            out["target"] = label_encoder.transform(out["burnout_level"])
            return out


        label_encoder = LabelEncoder()
        label_encoder.fit(["High", "Low", "Medium"])

        df_wfh_clean = prepare_wfh_dataset(df_wfh, label_encoder)
        df_dev_clean = prepare_dev_dataset(df_dev, label_encoder)

        print("Target encoding:", dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))
        print("WFH clean shape:", df_wfh_clean.shape)
        print("Developer clean shape:", df_dev_clean.shape)
        """
    ),
    code_cell(
        """
        def evaluate_model_suite(
            X: pd.DataFrame,
            y: pd.Series,
            models: dict,
            scaled_model_names: set[str],
            title: str,
        ):
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.2,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_sm)
            X_test_scaled = scaler.transform(X_test)

            rows = []
            fitted = {}

            print(title)
            print("Train class counts before SMOTE:", y_train.value_counts().to_dict())
            print("Train class counts after SMOTE: ", pd.Series(y_train_sm).value_counts().to_dict())

            for name, model in models.items():
                if name in scaled_model_names:
                    model.fit(X_train_scaled, y_train_sm)
                    preds = model.predict(X_test_scaled)
                else:
                    model.fit(X_train_sm, y_train_sm)
                    preds = model.predict(X_test)

                row = {
                    "model": name,
                    "accuracy": accuracy_score(y_test, preds),
                    "balanced_accuracy": balanced_accuracy_score(y_test, preds),
                    "f1_macro": f1_score(y_test, preds, average="macro"),
                }
                rows.append(row)
                fitted[name] = model

                print(f"\\n{name}")
                print(classification_report(y_test, preds, target_names=label_encoder.classes_))

            results = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
            return {
                "results": results,
                "models": fitted,
                "X_train": X_train,
                "X_test": X_test,
                "y_train": y_train,
                "y_test": y_test,
                "X_train_sm": X_train_sm,
                "y_train_sm": y_train_sm,
                "X_train_scaled": X_train_scaled,
                "X_test_scaled": X_test_scaled,
                "scaler": scaler,
            }
        """
    ),
    code_cell(
        """
        baseline_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "KNN": KNeighborsClassifier(n_neighbors=7),
            "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE),
            "Random Forest": RandomForestClassifier(
                n_estimators=250,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "XGBoost": XGBClassifier(
                n_estimators=250,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
            ),
        }

        scaled_models = {"Logistic Regression", "KNN", "SVM"}

        X_dev_base = df_dev_clean.drop(columns=["burnout_level", "target"])
        y_dev_base = df_dev_clean["target"]

        dev_base_run = evaluate_model_suite(
            X_dev_base,
            y_dev_base,
            baseline_models,
            scaled_models,
            title="Developer dataset baseline comparison",
        )

        display(dev_base_run["results"].style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))
        """
    ),
    code_cell(
        """
        shared_wfh = ["work_hours", "sleep_hours", "meetings_count", "screen_time_hours"]
        shared_dev = ["daily_work_hours", "sleep_hours", "meetings_per_day", "screen_time"]
        aligned_names = ["work_hours", "sleep_hours", "meetings", "screen_time"]

        X_wfh_shared = df_wfh_clean[shared_wfh].copy()
        X_wfh_shared.columns = aligned_names
        X_wfh_shared["source"] = 0

        X_dev_shared = df_dev_clean[shared_dev].copy()
        X_dev_shared.columns = aligned_names
        X_dev_shared["source"] = 1

        y_wfh_shared = df_wfh_clean["target"]
        y_dev_shared = df_dev_clean["target"]

        X_combined = pd.concat([X_wfh_shared, X_dev_shared], ignore_index=True)
        y_combined = pd.concat([y_wfh_shared, y_dev_shared], ignore_index=True)

        Xc_train, Xc_test, yc_train, yc_test = train_test_split(
            X_combined,
            y_combined,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=y_combined,
        )

        smote = SMOTE(random_state=RANDOM_STATE)
        Xc_train_sm, yc_train_sm = smote.fit_resample(Xc_train, yc_train)

        hybrid_scaler = StandardScaler()
        Xc_train_scaled = hybrid_scaler.fit_transform(Xc_train_sm)
        Xc_test_scaled = hybrid_scaler.transform(Xc_test)

        hybrid_model = VotingClassifier(
            estimators=[
                (
                    "rf",
                    RandomForestClassifier(
                        n_estimators=250,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
                (
                    "xgb",
                    XGBClassifier(
                        n_estimators=220,
                        max_depth=5,
                        learning_rate=0.08,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="mlogloss",
                        random_state=RANDOM_STATE,
                    ),
                ),
                ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)),
            ],
            voting="soft",
        )

        hybrid_model.fit(Xc_train_scaled, yc_train_sm)
        hybrid_preds = hybrid_model.predict(Xc_test_scaled)

        hybrid_summary = pd.DataFrame(
            [
                {
                    "model": "Hybrid Voting Ensemble",
                    "accuracy": accuracy_score(yc_test, hybrid_preds),
                    "balanced_accuracy": balanced_accuracy_score(yc_test, hybrid_preds),
                    "f1_macro": f1_score(yc_test, hybrid_preds, average="macro"),
                }
            ]
        )

        print(classification_report(yc_test, hybrid_preds, target_names=label_encoder.classes_))
        display(hybrid_summary.style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        hybrid_cv_scores = cross_val_score(
            hybrid_model,
            Xc_train_scaled,
            yc_train_sm,
            cv=cv,
            scoring="f1_macro",
        )
        print("Hybrid 5-fold CV F1 macro:", np.round(hybrid_cv_scores, 4))
        print("Hybrid CV mean:", round(hybrid_cv_scores.mean(), 4))
        """
    ),
    code_cell(
        """
        df_dev_fe = add_engineered_features(df_dev_clean.drop(columns=["target"])).copy()
        df_dev_fe["target"] = df_dev_clean["target"].values

        print("Engineered features used by the app:")
        print([col for col in MODEL_FEATURES if col in df_dev_fe.columns])

        X_dev_fe = df_dev_fe[MODEL_FEATURES].copy()
        y_dev_fe = df_dev_fe["target"]

        fe_models = {
            "Logistic Regression + FE": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            "Random Forest + FE": RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "XGBoost + FE": XGBClassifier(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.08,
                subsample=0.9,
                colsample_bytree=0.9,
                eval_metric="mlogloss",
                random_state=RANDOM_STATE,
            ),
        }

        dev_fe_run = evaluate_model_suite(
            X_dev_fe,
            y_dev_fe,
            fe_models,
            {"Logistic Regression + FE"},
            title="Developer dataset with engineered features",
        )

        fe_compare = dev_fe_run["results"].copy()
        display(fe_compare.style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))
        """
    ),
    code_cell(
        """
        X_train_fe = dev_fe_run["X_train"]
        X_test_fe = dev_fe_run["X_test"]
        y_train_fe = dev_fe_run["y_train"]
        y_test_fe = dev_fe_run["y_test"]

        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_fe_sm, y_train_fe_sm = smote.fit_resample(X_train_fe, y_train_fe)

        fe_scaler = StandardScaler()
        X_train_fe_scaled = fe_scaler.fit_transform(X_train_fe_sm)
        X_test_fe_scaled = fe_scaler.transform(X_test_fe)

        rf_search = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=RANDOM_STATE),
            param_distributions={
                "n_estimators": [200, 300, 400],
                "max_depth": [10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2"],
                "class_weight": ["balanced", "balanced_subsample"],
            },
            n_iter=12,
            scoring="f1_macro",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        rf_search.fit(X_train_fe_sm, y_train_fe_sm)

        xgb_search = RandomizedSearchCV(
            estimator=XGBClassifier(eval_metric="mlogloss", random_state=RANDOM_STATE),
            param_distributions={
                "n_estimators": [180, 250, 320],
                "max_depth": [4, 5, 6, 7],
                "learning_rate": [0.03, 0.05, 0.08, 0.1],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "min_child_weight": [1, 3, 5],
            },
            n_iter=12,
            scoring="f1_macro",
            cv=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        xgb_search.fit(X_train_fe_sm, y_train_fe_sm)

        tuned_rows = []
        for name, estimator, X_eval in [
            ("Random Forest Tuned", rf_search.best_estimator_, X_test_fe),
            ("XGBoost Tuned", xgb_search.best_estimator_, X_test_fe),
        ]:
            preds = estimator.predict(X_eval)
            tuned_rows.append(
                {
                    "model": name,
                    "accuracy": accuracy_score(y_test_fe, preds),
                    "balanced_accuracy": balanced_accuracy_score(y_test_fe, preds),
                    "f1_macro": f1_score(y_test_fe, preds, average="macro"),
                }
            )

        tuned_results = pd.DataFrame(tuned_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)

        print("Best RF params:", rf_search.best_params_)
        print("Best RF CV F1:", round(rf_search.best_score_, 4))
        print("Best XGB params:", xgb_search.best_params_)
        print("Best XGB CV F1:", round(xgb_search.best_score_, 4))
        display(tuned_results.style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))
        """
    ),
    code_cell(
        """
        svm_for_stack = SVC(
            kernel="rbf",
            C=10,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )
        svm_for_stack.fit(X_train_fe_scaled, y_train_fe_sm)

        stacking_model = StackingClassifier(
            estimators=[
                ("rf", rf_search.best_estimator_),
                ("xgb", xgb_search.best_estimator_),
                ("svm", svm_for_stack),
            ],
            final_estimator=LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            cv=5,
            n_jobs=-1,
        )

        stacking_model.fit(X_train_fe_scaled, y_train_fe_sm)
        stacking_preds = stacking_model.predict(X_test_fe_scaled)

        stack_summary = pd.DataFrame(
            [
                {
                    "model": "Stacking Ensemble",
                    "accuracy": accuracy_score(y_test_fe, stacking_preds),
                    "balanced_accuracy": balanced_accuracy_score(y_test_fe, stacking_preds),
                    "f1_macro": f1_score(y_test_fe, stacking_preds, average="macro"),
                }
            ]
        )

        print(classification_report(y_test_fe, stacking_preds, target_names=label_encoder.classes_))
        display(stack_summary.style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))
        """
    ),
    code_cell(
        """
        leaderboard = pd.concat(
            [
                dev_base_run["results"],
                hybrid_summary,
                fe_compare,
                tuned_results,
                stack_summary,
            ],
            ignore_index=True,
        ).sort_values("f1_macro", ascending=False).reset_index(drop=True)

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ["#c0392b" if "Tuned" in m or "Stacking" in m else "#2c7fb8" for m in leaderboard["model"]]
        ax.barh(leaderboard["model"], leaderboard["f1_macro"], color=colors)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_title("PulseWatch model leaderboard by macro F1")
        ax.set_xlabel("Macro F1")
        for idx, value in enumerate(leaderboard["f1_macro"]):
            ax.text(value + 0.01, idx, f"{value:.3f}", va="center")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "pulsewatch_model_leaderboard.png", dpi=160, bbox_inches="tight")
        plt.show()

        display(leaderboard.style.format({"accuracy": "{:.4f}", "balanced_accuracy": "{:.4f}", "f1_macro": "{:.4f}"}))
        """
    ),
    code_cell(
        """
        deploy_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features="log2",
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )

        deploy_scaler = StandardScaler()
        X_full_fe = df_dev_fe[MODEL_FEATURES].copy()
        y_full_fe = df_dev_fe["target"].copy()

        smote = SMOTE(random_state=RANDOM_STATE)
        X_full_fe_sm, y_full_fe_sm = smote.fit_resample(X_full_fe, y_full_fe)

        X_full_fe_scaled = deploy_scaler.fit_transform(X_full_fe_sm)
        deploy_model.fit(X_full_fe_scaled, y_full_fe_sm)

        joblib.dump(deploy_model, "burnout_model.pkl")
        joblib.dump(deploy_scaler, "scaler.pkl")
        joblib.dump(list(MODEL_FEATURES), "feature_names.pkl")
        joblib.dump(label_encoder, "label_encoder.pkl")

        print("Saved app-compatible artifacts:")
        print("- burnout_model.pkl")
        print("- scaler.pkl")
        print("- feature_names.pkl")
        print("- label_encoder.pkl")
        print("\\nDeployment model kept as RandomForest for SHAP TreeExplainer compatibility in app.py")
        """
    ),
    code_cell(
        """
        if SHAP_AVAILABLE:
            explainer = shap.TreeExplainer(deploy_model)
            sample_frame = pd.DataFrame(X_test_fe, columns=MODEL_FEATURES).iloc[:200].copy()
            sample_scaled = deploy_scaler.transform(sample_frame)
            shap_values = explainer.shap_values(sample_scaled)

            if isinstance(shap_values, list):
                shap_target = shap_values[0]
            elif np.ndim(shap_values) == 3:
                shap_target = shap_values[:, :, 0]
            else:
                shap_target = shap_values

            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_target,
                pd.DataFrame(sample_scaled, columns=MODEL_FEATURES),
                show=False,
                max_display=12,
            )
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / "pulsewatch_shap_summary.png", dpi=160, bbox_inches="tight")
            plt.show()
        else:
            print("SHAP is not available in this environment. Skip this cell or install shap to generate explainability plots.")
        """
    ),
    markdown_cell(
        """
        ## Summary

        This notebook now covers the full PulseWatch training story:

        - compares the original developer-only baseline models
        - keeps the shared-feature hybrid experiment across both datasets
        - evaluates the app-aligned engineered feature set
        - adds tuned models and stacking
        - exports deployable artifacts for `app.py`

        If you want to make the report stronger, the next useful improvements are:

        1. add calibration analysis for predicted probabilities
        2. test class weighting vs SMOTE for deployment stability
        3. evaluate grouped cross-validation if future data has team-level leakage risk
        """
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


TARGET.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
print(f"Wrote notebook to {TARGET}")
