"""Aggregate NYISO/BPA data, select one shared window, and train local RF models.

The shared stages deliberately stop before modelling geographic labels.  Zone IDs
are local to each transmission system, so the final WHEN and WHERE models are
trained and evaluated independently for NYISO and BPA.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import deque
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


RANDOM_STATE = 42
N_ESTIMATORS = 100
MIN_SAMPLES_LEAF = 5
RANDOM_BASELINE_REPEATS = 200
REFERENCE_WINDOW_HOURS = 1.0
EVENT_WINDOW_HOURS_VALUES = [0.25, 0.5, *map(float, range(1, 13))]

FEATURE_COLUMNS = [
    "num_events",
    "num_unique_ptids",
    "num_69kv_lines",
    "num_115kv_lines",
    "num_132kv_lines",
    "num_220kv_lines",
    "num_345kv_lines",
    "num_500kv_lines",
    "num_735kv_lines",
    "num_planned_outages",
    "num_auto_outages",
    "num_unique_buses",
    "num_events_last_15_min",
    "num_events_last_30_min",
    "num_events_last_60_min",
    "num_events_last_120_min",
    "node_degree_mean",
    "node_degree_std",
    "node_degree_min",
    "node_degree_max",
    "last_planned_voltage_69kv",
    "last_planned_voltage_115kv",
    "last_planned_voltage_132kv",
    "last_planned_voltage_220kv",
    "last_planned_voltage_345kv",
    "last_planned_voltage_500kv",
    "last_planned_voltage_735kv",
    "last_planned_node_degree_mean",
]
LABEL_COLUMNS = [
    "label_is_auto",
    "label_time_to_event_seconds",
    "label_from_zone",
    "label_to_zone",
]
REQUIRED_WHEN_COLUMNS = ["fold_id", *FEATURE_COLUMNS, *LABEL_COLUMNS, "is_bg"]
SOURCE_ORDER = ("nyiso", "bpa")


def window_label(hours: float) -> str:
    """Return the filename-compatible representation used by source notebooks."""
    return f"{hours:g}"


def source_paths(root: Path) -> dict[str, dict[str, Path]]:
    return {
        "nyiso": {
            "base": root / "src",
            "data": root / "src" / "output",
            "edge_zones": root / "src" / "output" / "edge_zones.csv",
        },
        "bpa": {
            "base": root / "src-bpa",
            "data": root / "src-bpa" / "output",
            "edge_zones": root / "src-bpa" / "output" / "edge_zones.csv",
        },
    }


def when_path(data_dir: Path, hours: float) -> Path:
    return data_dir / f"dataset_winsize{window_label(hours)}h_when.csv"


def where_path(data_dir: Path, hours: float) -> Path:
    return data_dir / f"dataset_winsize{window_label(hours)}h_where.csv"


def read_source_pair(data_dir: Path, hours: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    when_csv = when_path(data_dir, hours)
    where_csv = where_path(data_dir, hours)
    if not when_csv.exists() or not where_csv.exists():
        missing = [str(path) for path in (when_csv, where_csv) if not path.exists()]
        raise FileNotFoundError(f"Missing source datasets: {missing}")

    when_df = pd.read_csv(when_csv)
    where_df = pd.read_csv(where_csv)
    missing_columns = set(REQUIRED_WHEN_COLUMNS) - set(when_df.columns)
    if missing_columns:
        raise ValueError(f"{when_csv} is missing {sorted(missing_columns)}")
    if not {"fold_id", "window", "label", "is_bg"}.issubset(where_df.columns):
        raise ValueError(f"{where_csv} does not have the finalized WHERE schema")
    if len(when_df) != len(where_df):
        raise ValueError(f"WHEN/WHERE row counts differ for {hours:g}h in {data_dir}")
    if not np.array_equal(when_df["fold_id"], where_df["fold_id"]):
        raise ValueError(f"WHEN/WHERE folds differ for {hours:g}h in {data_dir}")
    return when_df, where_df


def normalize_bus_name(value: object) -> str | None:
    if pd.isna(value):
        return None
    name = re.sub(r"\s+", " ", str(value).strip()).upper()
    return None if not name or name == "---" else name


def coalesce(frame: pd.DataFrame, primary: str, fallback: str) -> pd.Series:
    return frame[primary].combine_first(frame[fallback])


def normalize_nyiso_outages(root: Path) -> pd.DataFrame:
    input_path = root / "src" / "processed-actual-outages.csv"
    raw = pd.read_csv(input_path, low_memory=False)
    event_time = pd.to_datetime(raw["MinTimeStamp"], errors="coerce")
    out_time = pd.to_datetime(raw["OutDatetime"], errors="coerce")
    end_time = pd.to_datetime(raw["MaxTimeStamp"], errors="coerce")

    frame = pd.DataFrame(
        {
            "operator": "nyiso",
            "source_row_id": np.arange(len(raw), dtype=int),
            "event_id": raw["PTID"].astype("string")
            + "@"
            + raw["OutDatetime"].astype("string"),
            "line_id": "nyiso::" + raw["PTID"].astype("string"),
            "line_name": raw["Name"].astype("string"),
            "event_datetime": event_time,
            "out_datetime": out_time,
            "end_datetime": end_time,
            "voltage_kv": pd.to_numeric(raw["Voltage"], errors="coerce"),
            "first_bus": "nyiso::" + raw["FirstBus"].astype("string"),
            "second_bus": "nyiso::" + raw["SecondBus"].astype("string"),
            "outage_type": raw["OutageType"].astype("string"),
        }
    )
    valid = (
        frame["event_datetime"].notna()
        & frame["out_datetime"].notna()
        & frame["end_datetime"].notna()
        & frame["voltage_kv"].notna()
        & frame["outage_type"].isin(["Planned", "Auto"])
        & (frame["event_datetime"].dt.year >= 2008)
        & (frame["out_datetime"].dt.year >= 2008)
    )
    return frame.loc[valid].reset_index(drop=True)


def normalize_bpa_outages(root: Path) -> pd.DataFrame:
    input_path = root / "src-bpa" / "bpa_processed.csv"
    raw = pd.read_csv(input_path, low_memory=False)
    out_time = pd.to_datetime(
        coalesce(raw, "Out Datetime", "Out Datetime (PPT)"),
        format="mixed",
        errors="coerce",
    )
    end_time = pd.to_datetime(
        coalesce(raw, "In Datetime", "In Datetime (PPT)"),
        format="mixed",
        errors="coerce",
    )
    voltage = pd.to_numeric(
        coalesce(raw, "Voltage (kV)", "Kilovolt"), errors="coerce"
    )
    first_bus = raw["First Bus"].map(normalize_bus_name)
    second_bus = raw["Second Bus"].map(normalize_bus_name)
    outage_type = raw["Outage Type"].replace({"Plan": "Planned"}).astype("string")

    source_event_id = raw["Outage ID"].astype("string")
    source_event_id = source_event_id.mask(
        source_event_id.isna(), raw["OMS Outage ID"].astype("string")
    )
    source_event_id = source_event_id.mask(
        source_event_id.isna(), raw["OARS Outage ID"].astype("string")
    )
    source_event_id = source_event_id.fillna(pd.Series(raw.index.astype(str)))

    frame = pd.DataFrame(
        {
            "operator": "bpa",
            "source_row_id": np.arange(len(raw), dtype=int),
            "event_id": source_event_id,
            "line_id": "bpa::" + raw["Name"].astype("string"),
            "line_name": raw["Name"].astype("string"),
            "event_datetime": out_time,
            "out_datetime": out_time,
            "end_datetime": end_time,
            "voltage_kv": voltage,
            "first_bus": first_bus.map(
                lambda value: None if value is None else f"bpa::{value}"
            ),
            "second_bus": second_bus.map(
                lambda value: None if value is None else f"bpa::{value}"
            ),
            "outage_type": outage_type,
        }
    )
    valid = (
        frame["event_datetime"].notna()
        & frame["voltage_kv"].notna()
        & frame["line_name"].notna()
        & frame["first_bus"].notna()
        & frame["second_bus"].notna()
        & frame["outage_type"].isin(["Planned", "Auto"])
        & (frame["event_datetime"].dt.year >= 2008)
    )
    return frame.loc[valid].reset_index(drop=True)


def write_aggregated_outages(root: Path, output_dir: Path) -> dict[str, Any]:
    """Normalize source labels and concatenate the event-level outage tables."""
    frames = [normalize_nyiso_outages(root), normalize_bpa_outages(root)]
    aggregate = pd.concat(frames, ignore_index=True, sort=False)
    aggregate.sort_values(["operator", "event_datetime", "source_row_id"], inplace=True)
    output_path = output_dir / "aggregated_outages.csv"
    aggregate.to_csv(output_path, index=False)

    type_counts = (
        aggregate.groupby(["operator", "outage_type"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    return {
        "path": str(output_path),
        "rows": int(len(aggregate)),
        "counts": {
            operator: {
                outage_type: int(type_counts.loc[operator, outage_type])
                for outage_type in type_counts.columns
            }
            for operator in type_counts.index
        },
    }


def truncated_exponential_pdf(
    values: np.ndarray, horizon: float, rate: float
) -> np.ndarray:
    return rate * np.exp(-rate * values) / (-np.expm1(-rate * horizon))


def truncated_exponential_cdf(
    values: np.ndarray, horizon: float, rate: float
) -> np.ndarray:
    numerator = -np.expm1(-rate * np.clip(values, 0, horizon))
    return numerator / (-np.expm1(-rate * horizon))


def mixture_negative_log_likelihood(
    parameters: np.ndarray, values: np.ndarray, horizon: float
) -> float:
    rate, foreground_probability = parameters
    density = foreground_probability * truncated_exponential_pdf(
        values, horizon, rate
    ) + (1 - foreground_probability) / horizon
    return float(-np.log(np.clip(density, 1e-30, None)).sum())


def mixture_ks_distance(
    values: np.ndarray,
    horizon: float,
    rate: float,
    foreground_probability: float,
) -> float:
    sorted_values = np.sort(values)
    empirical_cdf = np.arange(1, len(sorted_values) + 1) / len(sorted_values)
    model_cdf = foreground_probability * truncated_exponential_cdf(
        sorted_values, horizon, rate
    ) + (1 - foreground_probability) * np.clip(sorted_values / horizon, 0, 1)
    return float(np.max(np.abs(empirical_cdf - model_cdf)))


def fit_global_background_model(time_to_event: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame]:
    """Fit the same truncated-exponential/uniform mixture as the source notebooks."""
    finite = time_to_event[np.isfinite(time_to_event)]
    if len(finite) < 2 or float(finite.max()) <= 1800:
        raise ValueError("Insufficient finite time-to-event values for mixture fitting")

    fits: list[dict[str, Any]] = []
    for horizon in range(1800, int(finite.max()), 900):
        values = finite[(finite >= 0) & (finite <= horizon)]
        if len(values) < 2:
            continue
        result = minimize(
            mixture_negative_log_likelihood,
            x0=[1 / max(float(np.median(values)), 1.0), 0.95],
            args=(values, horizon),
            bounds=[(1e-8, 1e-2), (0.0, 1.0)],
            method="L-BFGS-B",
        )
        rate, foreground_probability = map(float, result.x)
        fits.append(
            {
                "horizon_seconds": int(horizon),
                "horizon_hours": horizon / 3600,
                "rate_per_second": rate,
                "mean_foreground_hours": (1 / rate) / 3600,
                "foreground_probability": foreground_probability,
                "ks_distance": mixture_ks_distance(
                    values, horizon, rate, foreground_probability
                ),
                "samples_within_horizon": int(len(values)),
                "optimizer_success": bool(result.success),
            }
        )

    if not fits:
        raise ValueError("No valid aggregate mixture fit was produced")
    fits_df = pd.DataFrame(fits).sort_values("horizon_seconds").reset_index(drop=True)
    best_index = fits_df["ks_distance"].idxmin()
    return fits_df.loc[best_index].to_dict(), fits_df


def classify_background(time_to_event: pd.Series, fit: dict[str, Any]) -> np.ndarray:
    values = time_to_event.to_numpy(dtype=float)
    horizon = float(fit["horizon_seconds"])
    rate = float(fit["rate_per_second"])
    foreground_probability = float(fit["foreground_probability"])
    foreground_density = foreground_probability * truncated_exponential_pdf(
        values, horizon, rate
    )
    background_density = np.full_like(
        values, (1 - foreground_probability) / horizon, dtype=float
    )
    return background_density > foreground_density


def fit_shared_outage_regime(
    paths: dict[str, dict[str, Path]], output_dir: Path
) -> tuple[dict[str, Any], pd.DataFrame]:
    reference_frames = []
    for operator in SOURCE_ORDER:
        when_df, _ = read_source_pair(
            paths[operator]["data"], REFERENCE_WINDOW_HOURS
        )
        reference_frames.append(when_df)
    reference = pd.concat(reference_frames, ignore_index=True)
    fit, fits_df = fit_global_background_model(
        reference["label_time_to_event_seconds"].to_numpy(dtype=float)
    )
    fits_df.to_csv(output_dir / "aggregate_mixture_fits.csv", index=False)
    return fit, fits_df


def aggregate_all_windows(
    paths: dict[str, dict[str, Path]],
    output_dir: Path,
    global_fit: dict[str, Any],
) -> tuple[dict[float, pd.DataFrame], list[dict[str, Any]]]:
    """Concatenate aligned WHEN/WHERE pairs and overwrite is_bg from one shared fit."""
    aggregate_when: dict[float, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []

    for hours in EVENT_WINDOW_HOURS_VALUES:
        when_frames = []
        where_frames = []
        for operator in SOURCE_ORDER:
            when_df, where_df = read_source_pair(paths[operator]["data"], hours)
            is_bg = classify_background(
                when_df["label_time_to_event_seconds"], global_fit
            )

            when_df = when_df.copy()
            where_df = where_df.copy()
            when_df["is_bg"] = is_bg
            where_df["is_bg"] = is_bg
            when_df.insert(0, "source_row_id", np.arange(len(when_df), dtype=int))
            when_df.insert(0, "operator", operator)
            where_df.insert(0, "source_row_id", np.arange(len(where_df), dtype=int))
            where_df.insert(0, "operator", operator)
            when_frames.append(when_df)
            where_frames.append(where_df)

        combined_when = pd.concat(when_frames, ignore_index=True, sort=False)
        combined_where = pd.concat(where_frames, ignore_index=True, sort=False)
        if len(combined_when) != len(combined_where):
            raise AssertionError("Aggregate WHEN/WHERE row counts differ")
        key_columns = ["operator", "source_row_id", "fold_id", "is_bg"]
        if not combined_when[key_columns].equals(combined_where[key_columns]):
            raise AssertionError("Aggregate WHEN/WHERE identity columns differ")

        label = window_label(hours)
        combined_when.to_csv(
            output_dir / f"dataset_winsize{label}h_when.csv", index=False
        )
        combined_where.to_csv(
            output_dir / f"dataset_winsize{label}h_where.csv", index=False
        )
        aggregate_when[hours] = combined_when

        summary: dict[str, Any] = {
            "event_window_hours": hours,
            "rows": int(len(combined_when)),
            "foreground_rows": int((~combined_when["is_bg"].astype(bool)).sum()),
            "background_rows": int(combined_when["is_bg"].astype(bool).sum()),
        }
        for operator in SOURCE_ORDER:
            operator_rows = combined_when["operator"] == operator
            summary[f"{operator}_rows"] = int(operator_rows.sum())
            summary[f"{operator}_foreground_rows"] = int(
                (operator_rows & ~combined_when["is_bg"].astype(bool)).sum()
            )
        summaries.append(summary)

    pd.DataFrame(summaries).to_csv(
        output_dir / "aggregate_dataset_summary.csv", index=False
    )
    return aggregate_when, summaries


def make_aggregate_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = frame[FEATURE_COLUMNS].copy()
    # The source indicator is used only while choosing a shared window.  Final
    # source-specific models do not need or receive it.
    features["operator_is_bpa"] = (frame["operator"] == "bpa").astype(int)
    if features.isna().any().any():
        missing = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Missing aggregate model features: {missing}")
    return features


def select_shared_event_window(
    datasets: dict[float, pd.DataFrame], model_output_dir: Path
) -> tuple[float, pd.DataFrame]:
    """Choose one lookback by aggregate five-fold RF validation MAE."""
    fold_rows: list[dict[str, Any]] = []
    for hours in EVENT_WINDOW_HOURS_VALUES:
        frame = datasets[hours]
        foreground = ~frame["is_bg"].astype(bool)
        X = make_aggregate_features(frame)
        y = frame["label_time_to_event_seconds"].astype(float)

        for fold_id in range(5):
            train_mask = foreground & (frame["fold_id"] >= 0) & (
                frame["fold_id"] != fold_id
            )
            validation_mask = foreground & (frame["fold_id"] == fold_id)
            if not train_mask.any() or not validation_mask.any():
                raise ValueError(f"Empty aggregate fold {fold_id} for {hours:g}h")

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                min_samples_leaf=MIN_SAMPLES_LEAF,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X.loc[train_mask], y.loc[train_mask])
            predictions = pd.Series(
                model.predict(X.loc[validation_mask]),
                index=frame.index[validation_mask],
            )
            row: dict[str, Any] = {
                "event_window_hours": hours,
                "fold_id": fold_id,
                "n_train": int(train_mask.sum()),
                "n_validation": int(validation_mask.sum()),
                "validation_mae_seconds": mean_absolute_error(
                    y.loc[validation_mask], predictions
                ),
            }
            for operator in SOURCE_ORDER:
                operator_mask = validation_mask & (frame["operator"] == operator)
                row[f"{operator}_n_validation"] = int(operator_mask.sum())
                row[f"{operator}_validation_mae_seconds"] = mean_absolute_error(
                    y.loc[operator_mask], predictions.loc[operator_mask]
                )
            fold_rows.append(row)

    fold_metrics = pd.DataFrame(fold_rows)
    fold_metrics.to_csv(model_output_dir / "window_cv_fold_metrics.csv", index=False)
    summary = (
        fold_metrics.groupby("event_window_hours", as_index=False)
        .agg(
            mean_validation_mae_seconds=("validation_mae_seconds", "mean"),
            std_validation_mae_seconds=("validation_mae_seconds", "std"),
            nyiso_mean_validation_mae_seconds=(
                "nyiso_validation_mae_seconds",
                "mean",
            ),
            bpa_mean_validation_mae_seconds=("bpa_validation_mae_seconds", "mean"),
        )
        .sort_values("event_window_hours")
    )
    summary.to_csv(model_output_dir / "window_cv_summary.csv", index=False)
    best_row = summary.sort_values(
        ["mean_validation_mae_seconds", "event_window_hours"]
    ).iloc[0]
    return float(best_row["event_window_hours"]), summary


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    error = np.asarray(y_pred, dtype=float) - y_true.to_numpy(dtype=float)
    absolute_error = np.abs(error)
    asymmetric_cost = np.where(error > 0, 5.0, 1.0) * absolute_error
    return {
        "mae_seconds": float(mean_absolute_error(y_true, y_pred)),
        "mae_minutes": float(mean_absolute_error(y_true, y_pred) / 60),
        "median_absolute_error_seconds": float(
            median_absolute_error(y_true, y_pred)
        ),
        "rmse_seconds": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_signed_error_seconds": float(error.mean()),
        "late_prediction_rate": float((error > 0).mean()),
        "asymmetric_cost_mae_seconds": float(asymmetric_cost.mean()),
    }


def zone_distance_lookup(edge_zone_path: Path) -> dict[int, dict[int, int]]:
    edges = pd.read_csv(edge_zone_path)
    required = {"from_zone", "to_zone"}
    if not required.issubset(edges.columns):
        raise ValueError(f"{edge_zone_path} is missing {sorted(required - set(edges))}")
    zones = set(edges["from_zone"].astype(int)) | set(edges["to_zone"].astype(int))
    adjacency = {zone: {zone} for zone in zones}
    for source, target in zip(
        edges["from_zone"].astype(int), edges["to_zone"].astype(int)
    ):
        adjacency[source].add(target)
        adjacency[target].add(source)

    lookup: dict[int, dict[int, int]] = {}
    for source in adjacency:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency[current]:
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        lookup[source] = distances
    return lookup


def distance(
    true_zone: int, predicted_zone: int, lookup: dict[int, dict[int, int]]
) -> float:
    return float(lookup.get(int(true_zone), {}).get(int(predicted_zone), np.nan))


def unordered_pair_metrics(
    true_from: pd.Series,
    true_to: pd.Series,
    predicted_from: np.ndarray,
    predicted_to: np.ndarray,
    lookup: dict[int, dict[int, int]],
) -> tuple[dict[str, Any], np.ndarray]:
    true_from_values = true_from.to_numpy(dtype=int)
    true_to_values = true_to.to_numpy(dtype=int)
    predicted_from = np.asarray(predicted_from, dtype=int)
    predicted_to = np.asarray(predicted_to, dtype=int)
    pair_distances = []
    for tf, tt, pf, pt in zip(
        true_from_values, true_to_values, predicted_from, predicted_to
    ):
        direct_parts = [distance(tf, pf, lookup), distance(tt, pt, lookup)]
        swapped_parts = [distance(tf, pt, lookup), distance(tt, pf, lookup)]
        candidates = []
        if np.isfinite(direct_parts).all():
            candidates.append(float(np.mean(direct_parts)))
        if np.isfinite(swapped_parts).all():
            candidates.append(float(np.mean(swapped_parts)))
        pair_distances.append(min(candidates) if candidates else np.nan)

    distances_array = np.asarray(pair_distances, dtype=float)
    finite = distances_array[np.isfinite(distances_array)]
    metrics: dict[str, Any] = {
        "mean_pair_graph_distance": float(finite.mean()) if len(finite) else None,
        "median_pair_graph_distance": float(np.median(finite)) if len(finite) else None,
        "p90_pair_graph_distance": (
            float(np.percentile(finite, 90)) if len(finite) else None
        ),
        "adjacent_or_same_pair_rate": (
            float((finite <= 1).mean()) if len(finite) else None
        ),
        "unreachable_pair_count": int(np.isnan(distances_array).sum()),
    }
    return metrics, distances_array


def summarize_random_runs(
    runs: list[dict[str, float | int | None]],
) -> dict[str, dict[str, float | None]]:
    """Return mean/std for every numeric metric across random baseline draws."""
    summary: dict[str, dict[str, float | None]] = {}
    for metric in runs[0]:
        values = np.asarray(
            [run[metric] for run in runs if run[metric] is not None], dtype=float
        )
        summary[metric] = {
            "mean": float(values.mean()) if len(values) else None,
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        }
    return summary


def save_feature_importance(
    model: RandomForestRegressor | RandomForestClassifier,
    output_path: Path,
) -> list[dict[str, Any]]:
    importance = pd.DataFrame(
        {"feature": FEATURE_COLUMNS, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)
    importance.to_csv(output_path, index=False)
    return importance.head(10).to_dict(orient="records")


def train_operator_models(
    operator: str,
    selected_frame: pd.DataFrame,
    edge_zone_path: Path,
    model_output_dir: Path,
    selected_window_hours: float,
) -> dict[str, Any]:
    operator_frame = selected_frame[selected_frame["operator"] == operator].copy()
    foreground = ~operator_frame["is_bg"].astype(bool)
    train_mask = foreground & (operator_frame["fold_id"] >= 0)
    test_mask = foreground & (operator_frame["fold_id"] == -1)
    if not train_mask.any() or not test_mask.any():
        raise ValueError(f"{operator} has an empty foreground train or test split")

    output_dir = model_output_dir / operator
    output_dir.mkdir(parents=True, exist_ok=True)
    X_train = operator_frame.loc[train_mask, FEATURE_COLUMNS]
    X_test = operator_frame.loc[test_mask, FEATURE_COLUMNS]

    when_target = "label_time_to_event_seconds"
    y_when_train = operator_frame.loc[train_mask, when_target].astype(float)
    y_when_test = operator_frame.loc[test_mask, when_target].astype(float)
    when_model = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    when_model.fit(X_train, y_when_train)
    when_predictions = when_model.predict(X_test)
    joblib.dump(when_model, output_dir / "random_forest_when.joblib")
    when_importance = save_feature_importance(
        when_model, output_dir / "when_feature_importance.csv"
    )

    rng = np.random.default_rng(RANDOM_STATE)
    random_when_runs = []
    first_random_when_predictions = None
    train_when_values = y_when_train.to_numpy(dtype=float)
    for repeat in range(RANDOM_BASELINE_REPEATS):
        random_predictions = rng.choice(
            train_when_values, size=len(y_when_test), replace=True
        )
        if repeat == 0:
            first_random_when_predictions = random_predictions.copy()
        random_when_runs.append(regression_metrics(y_when_test, random_predictions))

    zone_predictions: dict[str, np.ndarray] = {}
    zone_results: dict[str, Any] = {}
    for endpoint in ("from", "to"):
        target = f"label_{endpoint}_zone"
        y_train = operator_frame.loc[train_mask, target].astype(int)
        y_test = operator_frame.loc[test_mask, target].astype(int)
        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test).astype(int)
        joblib.dump(model, output_dir / f"random_forest_where_{endpoint}.joblib")
        top_features = save_feature_importance(
            model, output_dir / f"where_{endpoint}_feature_importance.csv"
        )
        zone_predictions[endpoint] = predictions
        zone_results[endpoint] = {
            "n_classes_train": int(y_train.nunique()),
            "n_classes_test": int(y_test.nunique()),
            "top_features": top_features,
        }

    true_from = operator_frame.loc[test_mask, "label_from_zone"].astype(int)
    true_to = operator_frame.loc[test_mask, "label_to_zone"].astype(int)
    lookup = zone_distance_lookup(edge_zone_path)
    pair_metrics, pair_distances = unordered_pair_metrics(
        true_from,
        true_to,
        zone_predictions["from"],
        zone_predictions["to"],
        lookup,
    )

    train_from = operator_frame.loc[train_mask, "label_from_zone"].to_numpy(dtype=int)
    train_to = operator_frame.loc[train_mask, "label_to_zone"].to_numpy(dtype=int)
    random_where_runs = []
    first_random_from = None
    first_random_to = None
    first_random_pair_distances = None
    for repeat in range(RANDOM_BASELINE_REPEATS):
        sampled_indices = rng.integers(0, len(train_from), size=len(true_from))
        random_from = train_from[sampled_indices]
        random_to = train_to[sampled_indices]
        random_metrics, random_distances = unordered_pair_metrics(
            true_from,
            true_to,
            random_from,
            random_to,
            lookup,
        )
        if repeat == 0:
            first_random_from = random_from.copy()
            first_random_to = random_to.copy()
            first_random_pair_distances = random_distances.copy()
        random_where_runs.append(random_metrics)

    if (
        first_random_when_predictions is None
        or first_random_from is None
        or first_random_to is None
        or first_random_pair_distances is None
    ):
        raise AssertionError("Random baseline generation produced no predictions")

    prediction_frame = pd.DataFrame(
        {
            "source_row_id": operator_frame.loc[test_mask, "source_row_id"].to_numpy(),
            "when_true_seconds": y_when_test.to_numpy(),
            "when_predicted_seconds": when_predictions,
            "when_error_seconds": when_predictions - y_when_test.to_numpy(),
            "when_random_predicted_seconds": first_random_when_predictions,
            "when_random_error_seconds": (
                first_random_when_predictions - y_when_test.to_numpy()
            ),
            "from_zone_true": true_from.to_numpy(),
            "from_zone_predicted": zone_predictions["from"],
            "to_zone_true": true_to.to_numpy(),
            "to_zone_predicted": zone_predictions["to"],
            "unordered_pair_graph_distance": pair_distances,
            "random_from_zone_predicted": first_random_from,
            "random_to_zone_predicted": first_random_to,
            "random_unordered_pair_graph_distance": first_random_pair_distances,
        }
    )
    prediction_frame.to_csv(output_dir / "held_out_predictions.csv", index=False)

    return {
        "operator": operator,
        "event_window_hours": selected_window_hours,
        "rows_total": int(len(operator_frame)),
        "background_rows": int(operator_frame["is_bg"].astype(bool).sum()),
        "foreground_rows": int(foreground.sum()),
        "n_train": int(train_mask.sum()),
        "n_test": int(test_mask.sum()),
        "when": {
            **regression_metrics(y_when_test, when_predictions),
            "top_features": when_importance,
        },
        "random_baseline": {
            "description": (
                f"{RANDOM_BASELINE_REPEATS} seeded empirical draws from the operator's "
                "foreground training targets; WHERE draws preserve complete training "
                "zone pairs"
            ),
            "when": summarize_random_runs(random_when_runs),
            "where": summarize_random_runs(random_where_runs),
        },
        "where": {
            "from_endpoint": zone_results["from"],
            "to_endpoint": zone_results["to"],
            **pair_metrics,
        },
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_safe(value), indent=2) + "\n")


def fmt(value: float | None, digits: int = 3) -> str:
    return "NA" if value is None else f"{value:.{digits}f}"


def fmt_pm(mean: float | None, std: float | None, digits: int = 3) -> str:
    if mean is None:
        return "NA"
    return f"{mean:.{digits}f} ± {float(std or 0):.{digits}f}"


def write_report(
    report_path: Path,
    outage_summary: dict[str, Any],
    global_fit: dict[str, Any],
    selected_window_hours: float,
    cv_summary: pd.DataFrame,
    operator_results: dict[str, dict[str, Any]],
) -> None:
    selected_cv = cv_summary[
        cv_summary["event_window_hours"] == selected_window_hours
    ].iloc[0]
    lines = [
        "# Aggregate NYISO/BPA random-forest results",
        "",
        "The outage tables and candidate WHEN/WHERE datasets were concatenated with an "
        "explicit `operator` column. Outage labels were normalized to `Planned`/`Auto`, "
        "and one aggregate truncated-exponential/uniform mixture supplied the foreground/"
        "background labels. Geographic zone IDs were never pooled for modelling.",
        "",
        "## Shared aggregate decisions",
        "",
        f"- Aggregated outage rows: {outage_summary['rows']:,}",
        f"- Mixture horizon: {float(global_fit['horizon_hours']):g} h",
        f"- Mixture KS distance: {float(global_fit['ks_distance']):.6f}",
        f"- Mixture foreground probability: {float(global_fit['foreground_probability']):.6f}",
        f"- Aggregate model samples: {sum(result['foreground_rows'] for result in operator_results.values()):,} foreground / "
        f"{sum(result['background_rows'] for result in operator_results.values()):,} background",
        f"- Selected event lookback: **{selected_window_hours:g} h**",
        "- Selection rule: lowest mean five-fold aggregate validation MAE; held-out "
        "`fold_id == -1` rows were not used.",
        f"- Selected-window validation MAE: {float(selected_cv['mean_validation_mae_seconds']):.3f} s "
        f"(SD {float(selected_cv['std_validation_mae_seconds']):.3f} s)",
        "",
        f"The {float(global_fit['horizon_hours']):g} h mixture horizon is a "
        "distributional cutoff used to label background events. It is distinct from "
        f"the {selected_window_hours:g} h feature lookback selected by model validation.",
        "",
        "Normalized event-level outage counts:",
        "",
        "| Operator | Planned | Auto |",
        "|---|---:|---:|",
    ]
    for operator in SOURCE_ORDER:
        counts = outage_summary["counts"][operator]
        lines.append(
            f"| {operator.upper()} | {counts.get('Planned', 0):,} | "
            f"{counts.get('Auto', 0):,} |"
        )
    lines.extend(
        [
            "",
            "## WHEN: separate held-out comparison",
            "",
            "The random baseline draws time-to-event values from the same operator's "
            "foreground training distribution. Lower MAE/RMSE is better.",
            "",
            "| Operator | Predictor | MAE (min) | RMSE (min) | R² |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for operator in SOURCE_ORDER:
        result = operator_results[operator]
        when = result["when"]
        random_when = result["random_baseline"]["when"]
        lines.append(
            f"| {operator.upper()} | Random empirical baseline | "
            f"{fmt_pm(random_when['mae_minutes']['mean'], random_when['mae_minutes']['std'])} | "
            f"{fmt_pm(random_when['rmse_seconds']['mean'] / 60, random_when['rmse_seconds']['std'] / 60)} | "
            f"{fmt_pm(random_when['r2']['mean'], random_when['r2']['std'])} |"
        )
        lines.append(
            f"| {operator.upper()} | Random forest (ours) | {fmt(when['mae_minutes'])} | "
            f"{fmt(when['rmse_seconds'] / 60)} | {fmt(when['r2'])} |"
        )
    lines.extend(
        [
            "",
            "## WHERE: separate held-out graph-distance comparison",
            "",
            "The random baseline draws complete source-local zone pairs from the training "
            "distribution. Distances use each operator's previously saved "
            "`output/edge_zones.csv` graph topology; lower is better.",
            "",
            "| Operator | Predictor | Mean pair distance | Median pair distance | Within one hop |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for operator in SOURCE_ORDER:
        result = operator_results[operator]
        where = result["where"]
        random_where = result["random_baseline"]["where"]
        lines.append(
            f"| {operator.upper()} | Random pair baseline | "
            f"{fmt_pm(random_where['mean_pair_graph_distance']['mean'], random_where['mean_pair_graph_distance']['std'])} | "
            f"{fmt_pm(random_where['median_pair_graph_distance']['mean'], random_where['median_pair_graph_distance']['std'])} | "
            f"{fmt_pm(100 * random_where['adjacent_or_same_pair_rate']['mean'], 100 * random_where['adjacent_or_same_pair_rate']['std'], 1)}% |"
        )
        lines.append(
            f"| {operator.upper()} | Random forest (ours) | "
            f"{fmt(where['mean_pair_graph_distance'])} | "
            f"{fmt(where['median_pair_graph_distance'])} | "
            f"{fmt(100 * where['adjacent_or_same_pair_rate'], 1)}% |"
        )
    lines.extend(
        [
            "",
            "All reported scores use only each operator's original held-out fold and only "
            "events labelled foreground by the shared aggregate mixture. Random forests "
            f"use {N_ESTIMATORS} trees and `min_samples_leaf={MIN_SAMPLES_LEAF}`. Random "
            f"baselines report mean ± SD over {RANDOM_BASELINE_REPEATS} seeded draws.",
            "",
            "Detailed metrics, predictions, feature importances, and serialized models are "
            "under `model_outputs/`.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n")


def run_pipeline(root: Path) -> dict[str, Any]:
    base_dir = root / "src-agg"
    output_dir = base_dir / "output"
    model_output_dir = base_dir / "model_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    paths = source_paths(root)

    outage_summary = write_aggregated_outages(root, output_dir)
    global_fit, _ = fit_shared_outage_regime(paths, output_dir)
    datasets, dataset_summaries = aggregate_all_windows(
        paths, output_dir, global_fit
    )
    selected_window, cv_summary = select_shared_event_window(
        datasets, model_output_dir
    )

    selection = {
        "reference_window_hours_for_mixture": REFERENCE_WINDOW_HOURS,
        "mixture_fit": global_fit,
        "selected_event_window_hours": selected_window,
        "selection_metric": "mean_five_fold_aggregate_validation_mae_seconds",
        "selection_scope": "foreground rows from fold_id 0..4; held-out fold -1 excluded",
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": MIN_SAMPLES_LEAF,
            "random_state": RANDOM_STATE,
        },
        "candidate_results": cv_summary.to_dict(orient="records"),
    }
    write_json(model_output_dir / "aggregate_selection.json", selection)

    operator_results = {}
    selected_frame = datasets[selected_window]
    for operator in SOURCE_ORDER:
        operator_results[operator] = train_operator_models(
            operator,
            selected_frame,
            paths[operator]["edge_zones"],
            model_output_dir,
            selected_window,
        )

    results = {
        "aggregated_outages": outage_summary,
        "aggregate_dataset_summaries": dataset_summaries,
        "selection": selection,
        "operator_results": operator_results,
    }
    write_json(model_output_dir / "results.json", results)
    write_report(
        base_dir / "RESULTS.md",
        outage_summary,
        global_fit,
        selected_window,
        cv_summary,
        operator_results,
    )
    return results


def parse_args() -> argparse.Namespace:
    default_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help="Repository root (defaults to the parent of src-agg)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline_results = run_pipeline(args.root.resolve())
    chosen = pipeline_results["selection"]["selected_event_window_hours"]
    print(f"Selected aggregate event window: {chosen:g}h")
    for source_name in SOURCE_ORDER:
        source_result = pipeline_results["operator_results"][source_name]
        print(
            f"{source_name.upper()}: WHEN MAE "
            f"{source_result['when']['mae_minutes']:.3f} min; "
            f"WHERE mean pair graph distance "
            f"{source_result['where']['mean_pair_graph_distance']:.3f}"
        )
