from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


BASE_FEATURES = [
    "age",
    "experience_years",
    "daily_work_hours",
    "sleep_hours",
    "caffeine_intake",
    "bugs_per_day",
    "commits_per_day",
    "meetings_per_day",
    "screen_time",
    "exercise_hours",
]

ENGINEERED_FEATURES = [
    "work_sleep_ratio",
    "screen_work_ratio",
    "bug_commit_ratio",
    "overwork_flag",
    "low_sleep_flag",
    "meeting_intensity",
]

MODEL_FEATURES = BASE_FEATURES + ENGINEERED_FEATURES

RECOMMENDATION_FEATURES = [
    "experience_years",
    "daily_work_hours",
    "sleep_hours",
    "caffeine_intake",
    "bugs_per_day",
    "commits_per_day",
    "meetings_per_day",
    "screen_time",
    "exercise_hours",
    "work_sleep_ratio",
    "screen_work_ratio",
    "bug_commit_ratio",
    "overwork_flag",
    "low_sleep_flag",
    "meeting_intensity",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the same engineered columns used by the saved prediction model."""
    out = df.copy()
    out["work_sleep_ratio"] = out["daily_work_hours"] / out["sleep_hours"].replace(0, np.nan)
    out["screen_work_ratio"] = out["screen_time"] / out["daily_work_hours"].replace(0, np.nan)
    out["bug_commit_ratio"] = out["bugs_per_day"] / (out["commits_per_day"] + 1)
    out["overwork_flag"] = (out["daily_work_hours"] > 10).astype(int)
    out["low_sleep_flag"] = (out["sleep_hours"] < 6).astype(int)
    out["meeting_intensity"] = out["meetings_per_day"] / out["daily_work_hours"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def load_developer_recommender_data(path: str = "Dataset/developer_burnout.csv") -> pd.DataFrame:
    """Load historical developer data without using burnout labels as recommendation labels."""
    df = pd.read_csv(path)
    df = df.copy()
    for col in BASE_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = add_engineered_features(df)
    df = df[MODEL_FEATURES].copy()
    df.insert(0, "employee_id", np.arange(1001, 1001 + len(df)))
    return df


@dataclass(frozen=True)
class Action:
    action_id: str
    suggestion: str
    category: str
    target_feature: str
    mode: str
    impacts: Dict[str, float]


class BurnoutSuggestionRecommender:
    """Unsupervised clustering plus ranking engine for burnout suggestions.

    No Suggested_Action labels are used. Suggestions are ranked by matching a
    person's burnout-gap vector to an action-effect vector.
    """

    PRESSURE_DIRECTIONS = {
        "experience_years": -1,
        "daily_work_hours": 1,
        "sleep_hours": -1,
        "caffeine_intake": 1,
        "bugs_per_day": 1,
        "commits_per_day": -1,
        "meetings_per_day": 1,
        "screen_time": 1,
        "exercise_hours": -1,
        "work_sleep_ratio": 1,
        "screen_work_ratio": 1,
        "bug_commit_ratio": 1,
        "overwork_flag": 1,
        "low_sleep_flag": 1,
        "meeting_intensity": 1,
    }

    DRIVER_LABELS = {
        "experience_years": "Early Tenure Support",
        "daily_work_hours": "Overwork",
        "sleep_hours": "Low Sleep Recovery",
        "caffeine_intake": "High Caffeine Load",
        "bugs_per_day": "Debugging Pressure",
        "commits_per_day": "Productivity Decline",
        "meetings_per_day": "Meeting Overload",
        "screen_time": "Screen Fatigue",
        "exercise_hours": "Low Physical Recovery",
        "work_sleep_ratio": "Work-Sleep Imbalance",
        "screen_work_ratio": "Screen Intensity",
        "bug_commit_ratio": "High Bug-to-Commit Ratio",
        "overwork_flag": "Overwork Pattern",
        "low_sleep_flag": "Low Sleep Pattern",
        "meeting_intensity": "Meeting Intensity",
    }

    ACTIONS = [
        Action(
            "recovery_break",
            "Take short recovery leave",
            "Recovery",
            "sleep_hours",
            "increase",
            {
                "sleep_hours": 1.0,
                "low_sleep_flag": 0.85,
                "work_sleep_ratio": 0.75,
                "daily_work_hours": 0.35,
                "caffeine_intake": 0.25,
            },
        ),
        Action(
            "reduce_workload",
            "Reduce workload",
            "Workload",
            "daily_work_hours",
            "decrease_percent",
            {
                "daily_work_hours": 1.0,
                "overwork_flag": 0.95,
                "work_sleep_ratio": 0.85,
                "screen_time": 0.45,
                "sleep_hours": 0.35,
            },
        ),
        Action(
            "reduce_meetings",
            "Reduce meeting load",
            "Focus Time",
            "meetings_per_day",
            "decrease_percent",
            {
                "meetings_per_day": 1.0,
                "meeting_intensity": 0.95,
                "screen_time": 0.35,
                "screen_work_ratio": 0.30,
            },
        ),
        Action(
            "flexible_hours",
            "Flexible working hours",
            "Work Arrangement",
            "daily_work_hours",
            "stabilize",
            {
                "daily_work_hours": 0.45,
                "sleep_hours": 0.65,
                "work_sleep_ratio": 0.70,
                "low_sleep_flag": 0.45,
                "exercise_hours": 0.25,
            },
        ),
        Action(
            "manager_checkin",
            "Weekly manager 1:1 support",
            "Manager Support",
            "bugs_per_day",
            "support",
            {
                "bugs_per_day": 0.55,
                "bug_commit_ratio": 0.55,
                "commits_per_day": 0.40,
                "experience_years": 0.45,
                "daily_work_hours": 0.25,
            },
        ),
        Action(
            "wellness_counseling",
            "Wellness counseling",
            "Wellbeing",
            "caffeine_intake",
            "decrease_percent",
            {
                "caffeine_intake": 1.0,
                "sleep_hours": 0.70,
                "low_sleep_flag": 0.65,
                "exercise_hours": 0.50,
                "work_sleep_ratio": 0.35,
            },
        ),
        Action(
            "team_redistribution",
            "Team redistribution",
            "Resourcing",
            "daily_work_hours",
            "decrease_percent",
            {
                "daily_work_hours": 0.85,
                "bugs_per_day": 0.65,
                "screen_time": 0.45,
                "overwork_flag": 0.75,
                "work_sleep_ratio": 0.55,
            },
        ),
        Action(
            "skill_mentoring",
            "Skill support / mentoring",
            "Enablement",
            "bug_commit_ratio",
            "decrease_percent",
            {
                "bug_commit_ratio": 1.0,
                "bugs_per_day": 0.80,
                "commits_per_day": 0.65,
                "experience_years": 0.45,
            },
        ),
        Action(
            "focus_blocks",
            "Protected deep-work blocks",
            "Focus Time",
            "screen_work_ratio",
            "decrease_percent",
            {
                "screen_work_ratio": 0.75,
                "screen_time": 0.55,
                "meetings_per_day": 0.55,
                "meeting_intensity": 0.50,
                "commits_per_day": 0.35,
            },
        ),
    ]

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        id_col: str = "employee_id",
        random_state: int = RANDOM_STATE,
    ) -> None:
        self.feature_columns = feature_columns or MODEL_FEATURES
        self.recommendation_features = [c for c in RECOMMENDATION_FEATURES if c in self.feature_columns]
        self.id_col = id_col
        self.random_state = random_state

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _scale_0_1(values: pd.Series) -> pd.Series:
        values = values.astype(float)
        if values.max() == values.min():
            return pd.Series(np.full(len(values), 0.5), index=values.index)
        return (values - values.min()) / (values.max() - values.min())

    def _cluster_metrics(self, x: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        labels = np.asarray(labels)
        mask = labels != -1
        x_valid = x[mask]
        labels_valid = labels[mask]
        unique = np.unique(labels_valid)
        if len(unique) < 2 or len(unique) >= len(labels_valid):
            return {
                "valid": False,
                "n_clusters": len(unique),
                "noise_ratio": float(np.mean(labels == -1)),
                "silhouette": np.nan,
                "davies_bouldin": np.nan,
                "calinski_harabasz": np.nan,
                "intra_cluster_variance": np.nan,
            }

        centroids = np.vstack([x_valid[labels_valid == c].mean(axis=0) for c in unique])
        variances = []
        for idx, cluster in enumerate(unique):
            points = x_valid[labels_valid == cluster]
            variances.append(np.mean(np.sum((points - centroids[idx]) ** 2, axis=1)))

        return {
            "valid": True,
            "n_clusters": len(unique),
            "noise_ratio": float(np.mean(labels == -1)),
            "silhouette": float(silhouette_score(x_valid, labels_valid)),
            "davies_bouldin": float(davies_bouldin_score(x_valid, labels_valid)),
            "calinski_harabasz": float(calinski_harabasz_score(x_valid, labels_valid)),
            "intra_cluster_variance": float(np.mean(variances)),
        }

    def _evaluate_cluster_models(self, x: np.ndarray, k_range: Iterable[int]) -> pd.DataFrame:
        rows = []
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=20)
            labels = kmeans.fit_predict(x)
            row = {"algorithm": "KMeans", "k": k, "eps": np.nan, "model": kmeans}
            row.update(self._cluster_metrics(x, labels))
            rows.append(row)

            gmm = GaussianMixture(n_components=k, covariance_type="full", random_state=self.random_state)
            labels = gmm.fit_predict(x)
            row = {"algorithm": "GaussianMixture", "k": k, "eps": np.nan, "model": gmm}
            row.update(self._cluster_metrics(x, labels))
            rows.append(row)

            agg = AgglomerativeClustering(n_clusters=k)
            labels = agg.fit_predict(x)
            row = {"algorithm": "Agglomerative", "k": k, "eps": np.nan, "model": agg}
            row.update(self._cluster_metrics(x, labels))
            rows.append(row)

        if len(x) > 20:
            nn = NearestNeighbors(n_neighbors=min(8, len(x) - 1)).fit(x)
            distances, _ = nn.kneighbors(x)
            kth_dist = distances[:, -1]
            for eps in np.quantile(kth_dist, [0.60, 0.75, 0.90]):
                dbscan = DBSCAN(eps=float(eps), min_samples=max(5, int(math.sqrt(len(x)) / 2)))
                labels = dbscan.fit_predict(x)
                row = {"algorithm": "DBSCAN", "k": np.nan, "eps": float(eps), "model": dbscan}
                row.update(self._cluster_metrics(x, labels))
                rows.append(row)

        metrics = pd.DataFrame(rows)
        valid = metrics["valid"] == True
        metrics["silhouette_scaled"] = 0.0
        metrics["dbi_scaled"] = 0.0
        metrics["ch_scaled"] = 0.0
        metrics.loc[valid, "silhouette_scaled"] = (metrics.loc[valid, "silhouette"] + 1) / 2
        metrics.loc[valid, "dbi_scaled"] = 1 / (1 + metrics.loc[valid, "davies_bouldin"])
        metrics.loc[valid, "ch_scaled"] = self._scale_0_1(np.log1p(metrics.loc[valid, "calinski_harabasz"]))
        noise_penalty = 1 - metrics["noise_ratio"].fillna(0).clip(0, 0.8)
        metrics["selection_score"] = (
            0.45 * metrics["silhouette_scaled"]
            + 0.35 * metrics["dbi_scaled"]
            + 0.20 * metrics["ch_scaled"]
        ) * noise_penalty
        metrics.loc[~valid, "selection_score"] = -np.inf
        return metrics.sort_values("selection_score", ascending=False).reset_index(drop=True)

    def fit(self, df: pd.DataFrame, k_range: Iterable[int] = range(2, 8)) -> "BurnoutSuggestionRecommender":
        frame = df.copy()
        if self.id_col not in frame.columns:
            frame[self.id_col] = np.arange(1, len(frame) + 1)

        for col in self.feature_columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

        self.df_ = frame[[self.id_col] + self.feature_columns].copy()
        self.pipeline_ = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        self.x_ = self.pipeline_.fit_transform(self.df_[self.feature_columns])

        self.cluster_metrics_ = self._evaluate_cluster_models(self.x_, k_range)
        self.best_model_info_ = self.cluster_metrics_.iloc[0].copy()
        self.cluster_model_ = self.best_model_info_["model"]
        self.labels_ = self._fit_predict_best_model()

        self.centroids_ = self._compute_centroids(self.x_, self.labels_)
        self.feature_z_ = pd.DataFrame(
            self.pipeline_.transform(self.df_[self.feature_columns]),
            columns=self.feature_columns,
            index=self.df_.index,
        )

        self.pressure_features_ = [c for c in self.recommendation_features if c in self.feature_columns]
        self.feature_directions_ = pd.Series(
            {c: self.PRESSURE_DIRECTIONS.get(c, 1) for c in self.pressure_features_},
            dtype=float,
        )

        cluster_z = self.feature_z_.assign(cluster=self.labels_).query("cluster != -1")
        self.cluster_feature_z_ = cluster_z.groupby("cluster")[self.pressure_features_].mean()
        pressure = self.cluster_feature_z_.mul(self.feature_directions_, axis=1).clip(lower=0).mean(axis=1)
        self.cluster_pressure_ = pressure
        self.healthy_cluster_ = int(pressure.idxmin())
        self.healthy_centroid_z_ = self.cluster_feature_z_.loc[self.healthy_cluster_].values

        self.cluster_names_ = self._name_clusters()
        self.action_matrix_ = self._build_action_matrix()
        self.feature_weights_ = pd.Series(1.0 / len(self.pressure_features_), index=self.pressure_features_)
        return self

    def _fit_predict_best_model(self) -> np.ndarray:
        algo = self.best_model_info_["algorithm"]
        if algo == "KMeans":
            self.cluster_model_ = KMeans(
                n_clusters=int(self.best_model_info_["k"]),
                random_state=self.random_state,
                n_init=20,
            )
            return self.cluster_model_.fit_predict(self.x_)
        if algo == "GaussianMixture":
            self.cluster_model_ = GaussianMixture(
                n_components=int(self.best_model_info_["k"]),
                covariance_type="full",
                random_state=self.random_state,
            )
            return self.cluster_model_.fit_predict(self.x_)
        if algo == "Agglomerative":
            self.cluster_model_ = AgglomerativeClustering(n_clusters=int(self.best_model_info_["k"]))
            return self.cluster_model_.fit_predict(self.x_)
        self.cluster_model_ = DBSCAN(
            eps=float(self.best_model_info_["eps"]),
            min_samples=max(5, int(math.sqrt(len(self.x_)) / 2)),
        )
        return self.cluster_model_.fit_predict(self.x_)

    @staticmethod
    def _compute_centroids(x: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        clusters = sorted([int(c) for c in np.unique(labels) if c != -1])
        values = [x[labels == c].mean(axis=0) for c in clusters]
        return pd.DataFrame(values, index=clusters)

    def _name_clusters(self) -> Dict[int, str]:
        names = {}
        for cluster in self.cluster_feature_z_.index:
            if int(cluster) == self.healthy_cluster_:
                names[int(cluster)] = "Healthy Reference Group"
                continue
            gap = self._cluster_gap_vector(int(cluster))
            top_features = (
                pd.Series(gap, index=self.pressure_features_)
                .sort_values(ascending=False)
                .loc[lambda s: s > 0.10]
                .head(2)
                .index.tolist()
            )
            if top_features:
                names[int(cluster)] = " + ".join(self.DRIVER_LABELS.get(f, f) for f in top_features)
            else:
                names[int(cluster)] = "Balanced Burnout Pattern"
        return names

    def _cluster_gap_vector(self, cluster: int) -> np.ndarray:
        centroid = self.cluster_feature_z_.loc[cluster].values
        gap = self.feature_directions_.values * (centroid - self.healthy_centroid_z_)
        return np.clip(gap, 0, None)

    def _build_action_matrix(self) -> pd.DataFrame:
        rows = []
        for action in self.ACTIONS:
            row = {"action_id": action.action_id}
            for feature in self.pressure_features_:
                row[feature] = float(action.impacts.get(feature, 0.0))
            rows.append(row)
        matrix = pd.DataFrame(rows).set_index("action_id")
        return matrix[matrix.sum(axis=1) > 0]

    def _profile_to_frame(self, profile: pd.Series | Dict[str, float] | pd.DataFrame) -> pd.DataFrame:
        if isinstance(profile, pd.DataFrame):
            frame = profile.copy()
        elif isinstance(profile, pd.Series):
            frame = profile.to_frame().T
        else:
            frame = pd.DataFrame([profile])
        for col in self.feature_columns:
            if col not in frame.columns:
                frame[col] = np.nan
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        return frame[self.feature_columns]

    def _profile_z(self, profile: pd.Series | Dict[str, float] | pd.DataFrame) -> np.ndarray:
        frame = self._profile_to_frame(profile)
        transformed = self.pipeline_.transform(frame)
        z_full = pd.DataFrame(transformed, columns=self.feature_columns)
        return z_full[self.pressure_features_].iloc[0].values

    def assign_cluster(self, profile: pd.Series | Dict[str, float] | pd.DataFrame) -> Tuple[int, float]:
        frame = self._profile_to_frame(profile)
        x = self.pipeline_.transform(frame)

        if isinstance(self.cluster_model_, GaussianMixture):
            probabilities = self.cluster_model_.predict_proba(x)[0]
            cluster = int(np.argmax(probabilities))
            return cluster, float(np.max(probabilities))

        distances = np.linalg.norm(x[:, None, :] - self.centroids_.values[None, :, :], axis=2)[0]
        nearest_order = np.argsort(distances)
        cluster = int(self.centroids_.index[nearest_order[0]])
        if len(nearest_order) == 1:
            confidence = 0.5
        else:
            confidence = 1 - distances[nearest_order[0]] / (distances[nearest_order[1]] + 1e-9)
        return cluster, float(np.clip(confidence, 0, 1))

    def driver_table(self, profile: pd.Series | Dict[str, float] | pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        frame = self._profile_to_frame(profile)
        z = self._profile_z(frame)
        gap = self.feature_directions_.values * (z - self.healthy_centroid_z_)
        gap = np.clip(gap, 0, None)

        scaler = self.pipeline_.named_steps["scaler"]
        healthy_reference = {}
        for pos, feature in enumerate(self.pressure_features_):
            feature_idx = self.feature_columns.index(feature)
            healthy_reference[feature] = (
                self.healthy_centroid_z_[pos] * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]
            )

        out = pd.DataFrame(
            {
                "feature": self.pressure_features_,
                "driver": [self.DRIVER_LABELS.get(f, f.replace("_", " ").title()) for f in self.pressure_features_],
                "employee_value": [float(frame.iloc[0][f]) if pd.notna(frame.iloc[0][f]) else np.nan for f in self.pressure_features_],
                "healthy_reference": [float(healthy_reference[f]) for f in self.pressure_features_],
                "gap_strength": gap,
            }
        )
        return out.sort_values("gap_strength", ascending=False).head(top_k).reset_index(drop=True)

    def _personalize_text(self, action: Action, profile: pd.Series) -> str:
        target = action.target_feature
        if target not in profile.index or pd.isna(profile[target]):
            return action.suggestion

        value = float(profile[target])
        if action.mode == "decrease_percent" and value > 0:
            if target in ["daily_work_hours", "screen_time"]:
                pct = 15 if value < 10 else 25
            elif target in ["meetings_per_day", "meeting_intensity"]:
                pct = 25 if value < 6 else 35
            else:
                pct = 20
            return f"{action.suggestion} by {pct}%"

        if action.mode == "increase":
            if target == "sleep_hours":
                return "Take 1-2 days recovery leave and restore sleep routine"
            if target == "exercise_hours":
                return "Add short daily recovery activity"

        return action.suggestion

    def recommend_for_profile(
        self,
        profile: pd.Series | Dict[str, float] | pd.DataFrame,
        burnout_risk_score: Optional[float] = None,
        top_n: int = 5,
    ) -> Dict[str, object]:
        frame = self._profile_to_frame(profile)
        row = frame.iloc[0]
        cluster, cluster_confidence = self.assign_cluster(frame)
        z = self._profile_z(frame)
        gap = self.feature_directions_.values * (z - self.healthy_centroid_z_)
        gap = np.clip(gap, 0, None)
        weighted_gap = gap * self.feature_weights_.values
        cluster_gap = self._cluster_gap_vector(cluster)

        risk_factor = 0.5 if burnout_risk_score is None else float(np.clip(burnout_risk_score / 100, 0, 1))
        distance_to_healthy = float(np.linalg.norm(weighted_gap))
        recovery_score = float(np.clip(100 * (1 - distance_to_healthy / (distance_to_healthy + 0.35)), 0, 100))

        rows = []
        for action in self.ACTIONS:
            if action.action_id not in self.action_matrix_.index:
                continue
            action_vec = self.action_matrix_.loc[action.action_id].values.astype(float)
            similarity = self._cosine(gap, action_vec)
            coverage = float(np.dot(weighted_gap, action_vec) / (weighted_gap.sum() + 1e-9))
            cluster_alignment = self._cosine(cluster_gap, action_vec)
            confidence = float(
                np.clip(
                    0.42 * similarity
                    + 0.30 * coverage
                    + 0.18 * cluster_alignment
                    + 0.10 * risk_factor,
                    0,
                    1,
                )
            )

            impacted = [
                feature
                for idx, feature in enumerate(self.pressure_features_)
                if action_vec[idx] > 0 and gap[idx] > 0
            ]
            impacted = sorted(
                impacted,
                key=lambda feature: gap[self.pressure_features_.index(feature)],
                reverse=True,
            )[:3]
            if impacted:
                reason = "Targets " + ", ".join(self.DRIVER_LABELS.get(f, f) for f in impacted)
            else:
                reason = "Matches the learned cluster profile"

            rows.append(
                {
                    "recommendation": self._personalize_text(action, row),
                    "category": action.category,
                    "confidence": round(confidence, 2),
                    "why": reason,
                    "action_id": action.action_id,
                    "similarity": round(similarity, 3),
                    "coverage": round(coverage, 3),
                }
            )

        recommendations = (
            pd.DataFrame(rows)
            .sort_values("confidence", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )
        recommendations.insert(0, "rank", np.arange(1, len(recommendations) + 1))

        drivers = self.driver_table(frame, top_k=5)
        return {
            "cluster": cluster,
            "cluster_name": self.cluster_names_.get(cluster, "Outlier / Unmapped Pattern"),
            "cluster_confidence": round(cluster_confidence, 2),
            "burnout_risk_score": None if burnout_risk_score is None else round(float(burnout_risk_score), 1),
            "recovery_score": round(recovery_score, 1),
            "top_drivers": drivers,
            "recommendations": recommendations,
        }

    def algorithm_summary(self) -> pd.DataFrame:
        columns = [
            "algorithm",
            "k",
            "n_clusters",
            "noise_ratio",
            "silhouette",
            "davies_bouldin",
            "calinski_harabasz",
            "intra_cluster_variance",
            "selection_score",
        ]
        return self.cluster_metrics_[[c for c in columns if c in self.cluster_metrics_.columns]].head(10)
