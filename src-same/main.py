#!/usr/bin/env python3
"""Predict a repeat automatic line outage within 30 days.

The script runs the same leakage-aware random-forest experiment on the
processed NYISO and BPA outage files.  It compares line-history lookback
windows of 0.5, 1, 2, 3, and 4 hours using a chronological validation set,
then evaluates the selected window once on a chronological test set.

Every input record is retained while features are built.  A record can only
be used for supervised learning when it has a timestamp, a line identifier,
and a complete 30-day follow-up period in the source file.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


SEED = 20260922
FOLLOW_UP_DAYS = 30
WINDOW_HOURS = (0.5, 1.0, 2.0, 3.0, 4.0)
TRAIN_FRACTION = 0.60
VALIDATION_FRACTION = 0.20


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    time_columns: tuple[str, ...]
    line_column: str
    outage_type_column: str
    voltage_column: str
    automatic_value: str = "Auto"


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "nyiso": SourceSpec(
        name="nyiso",
        path=ROOT / "src" / "processed-actual-outages.csv",
        time_columns=("OutDatetime",),
        line_column="Name",
        outage_type_column="OutageType",
        voltage_column="Voltage",
    ),
    "bpa": SourceSpec(
        name="bpa",
        path=ROOT / "src-bpa" / "bpa_processed.csv",
        # BPA changed column names across vintages. Exactly one of these is
        # populated per row in the processed file, so coalescing retains both.
        time_columns=("Out Datetime", "Out Datetime (PPT)"),
        line_column="Name",
        outage_type_column="Outage Type",
        voltage_column="Voltage (kV)",
    ),
}

FEATURE_COLUMNS = [
    "current_is_automatic",
    "current_voltage_kv",
    "history_record_count",
    "history_automatic_count",
    "history_planned_count",
    "history_automatic_share",
    "history_distinct_timestamp_count",
    "has_prior_line_record_in_window",
    "minutes_since_prior_line_record",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--systems",
        nargs="+",
        choices=tuple(SOURCES),
        default=list(SOURCES),
        help="Processed systems to run (default: nyiso bpa).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs",
        help="Directory for metrics, predictions, metadata, and fitted models.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=300,
        help="Trees in each random forest (default: 300).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel random-forest jobs (default: all available cores).",
    )
    return parser.parse_args()


def read_source(spec: SourceSpec) -> pd.DataFrame:
    if not spec.path.exists():
        raise FileNotFoundError(f"Missing processed {spec.name.upper()} file: {spec.path}")

    frame = pd.read_csv(spec.path, low_memory=False)
    required = {
        *spec.time_columns,
        spec.line_column,
        spec.outage_type_column,
        spec.voltage_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{spec.path} is missing required columns: {missing}")

    timestamp = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns]")
    for column in spec.time_columns:
        timestamp = timestamp.fillna(pd.to_datetime(frame[column], errors="coerce"))

    line_id = frame[spec.line_column].astype("string").str.strip().str.casefold()
    outage_type = frame[spec.outage_type_column].astype("string").str.strip()

    prepared = pd.DataFrame(
        {
            "source_row": np.arange(len(frame), dtype=np.int64),
            "timestamp": timestamp,
            "line_id": line_id,
            "is_automatic": outage_type.eq(spec.automatic_value).astype(np.int8),
            "voltage_kv": pd.to_numeric(frame[spec.voltage_column], errors="coerce"),
        }
    )
    return prepared


def add_repeat_outage_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Label strictly later automatic records on the same line within 30 days."""
    result = frame.copy()
    result["target_repeat_auto_30d"] = pd.Series(pd.NA, index=result.index, dtype="Int8")

    valid = result["timestamp"].notna() & result["line_id"].notna()
    if not valid.any():
        return result

    reporting_end = result.loc[valid, "timestamp"].max()
    follow_up_end = reporting_end - pd.Timedelta(days=FOLLOW_UP_DAYS)

    for _, group in result.loc[valid].groupby("line_id", sort=False):
        timestamps = group["timestamp"].to_numpy(dtype="datetime64[ns]")
        automatic_times = np.sort(
            group.loc[group["is_automatic"].eq(1), "timestamp"].to_numpy(
                dtype="datetime64[ns]"
            )
        )
        labels = np.zeros(len(group), dtype=np.int8)
        if automatic_times.size:
            next_positions = np.searchsorted(automatic_times, timestamps, side="right")
            has_next = next_positions < automatic_times.size
            next_times = np.full(len(group), np.datetime64("NaT"), dtype="datetime64[ns]")
            next_times[has_next] = automatic_times[next_positions[has_next]]
            labels = (
                has_next
                & (next_times <= timestamps + np.timedelta64(FOLLOW_UP_DAYS, "D"))
            ).astype(np.int8)

        observable = group["timestamp"].le(follow_up_end).to_numpy()
        result.loc[group.index[observable], "target_repeat_auto_30d"] = labels[observable]

    return result


def build_features(frame: pd.DataFrame, window_hours: float) -> pd.DataFrame:
    """Build features from strictly earlier same-line records in the lookback."""
    features = pd.DataFrame(index=frame.index)
    features["current_is_automatic"] = frame["is_automatic"].astype(float)
    features["current_voltage_kv"] = frame["voltage_kv"].astype(float)

    for column in (
        "history_record_count",
        "history_automatic_count",
        "history_planned_count",
        "history_distinct_timestamp_count",
        "has_prior_line_record_in_window",
        "minutes_since_prior_line_record",
    ):
        features[column] = 0.0

    valid = frame["timestamp"].notna() & frame["line_id"].notna()
    window_ns = int(window_hours * 60 * 60 * 1_000_000_000)

    for _, unsorted_group in frame.loc[valid].groupby("line_id", sort=False):
        group = unsorted_group.sort_values(["timestamp", "source_row"], kind="stable")
        indices = group.index.to_numpy()
        times = group["timestamp"].astype("int64").to_numpy()
        automatic = group["is_automatic"].to_numpy(dtype=np.int64)

        starts = np.searchsorted(times, times - window_ns, side="left")
        # side='left' excludes the anchor and every record at the same timestamp.
        ends = np.searchsorted(times, times, side="left")
        counts = ends - starts

        automatic_prefix = np.r_[0, np.cumsum(automatic)]
        automatic_counts = automatic_prefix[ends] - automatic_prefix[starts]
        planned_counts = counts - automatic_counts

        distinct_counts = np.zeros(len(group), dtype=np.int64)
        minutes_since = np.full(len(group), window_hours * 60.0, dtype=float)
        for position, (start, end) in enumerate(zip(starts, ends)):
            if end > start:
                prior_times = times[start:end]
                distinct_counts[position] = np.unique(prior_times).size
                minutes_since[position] = (times[position] - prior_times[-1]) / 60e9

        features.loc[indices, "history_record_count"] = counts
        features.loc[indices, "history_automatic_count"] = automatic_counts
        features.loc[indices, "history_planned_count"] = planned_counts
        features.loc[indices, "history_distinct_timestamp_count"] = distinct_counts
        features.loc[indices, "has_prior_line_record_in_window"] = (counts > 0).astype(int)
        features.loc[indices, "minutes_since_prior_line_record"] = minutes_since

    counts = features["history_record_count"].to_numpy()
    auto_counts = features["history_automatic_count"].to_numpy()
    features["history_automatic_share"] = np.divide(
        auto_counts,
        counts,
        out=np.zeros(len(features), dtype=float),
        where=counts > 0,
    )

    timestamp = frame["timestamp"]
    hour = timestamp.dt.hour + timestamp.dt.minute / 60.0
    weekday = timestamp.dt.dayofweek
    month = timestamp.dt.month - 1
    features["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    features["weekday_sin"] = np.sin(2 * np.pi * weekday / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * weekday / 7)
    features["month_sin"] = np.sin(2 * np.pi * month / 12)
    features["month_cos"] = np.cos(2 * np.pi * month / 12)
    return features[FEATURE_COLUMNS]


def chronological_split(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    labeled = frame["target_repeat_auto_30d"].notna()
    unique_times = np.sort(frame.loc[labeled, "timestamp"].unique())
    if unique_times.size < 3:
        raise ValueError("At least three distinct labeled timestamps are required.")

    train_position = max(0, min(unique_times.size - 3, int(unique_times.size * TRAIN_FRACTION) - 1))
    validation_position = max(
        train_position + 1,
        min(unique_times.size - 2, int(unique_times.size * (TRAIN_FRACTION + VALIDATION_FRACTION)) - 1),
    )
    train_end = pd.Timestamp(unique_times[train_position])
    validation_end = pd.Timestamp(unique_times[validation_position])

    train = labeled & frame["timestamp"].le(train_end)
    validation = labeled & frame["timestamp"].gt(train_end) & frame["timestamp"].le(validation_end)
    test = labeled & frame["timestamp"].gt(validation_end)
    if min(train.sum(), validation.sum(), test.sum()) == 0:
        raise ValueError("Chronological split produced an empty partition.")

    metadata = {
        "train_end": train_end.isoformat(),
        "validation_end": validation_end.isoformat(),
        "reporting_end": frame["timestamp"].max().isoformat(),
        "train_rows": int(train.sum()),
        "validation_rows": int(validation.sum()),
        "test_rows": int(test.sum()),
    }
    return train, validation, test, metadata


def make_model(n_estimators: int, n_jobs: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "random_forest",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    min_samples_leaf=3,
                    class_weight="balanced_subsample",
                    random_state=SEED,
                    n_jobs=n_jobs,
                ),
            ),
        ]
    )


def safe_roc_auc(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(roc_auc_score(y_true, probability)) if np.unique(y_true).size == 2 else float("nan")


def best_f1_threshold(y_true: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    if thresholds.size == 0:
        return 0.5
    denominator = precision[:-1] + recall[:-1]
    scores = np.divide(
        2 * precision[:-1] * recall[:-1],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    return float(thresholds[int(np.argmax(scores))])


def score_predictions(
    y_true: np.ndarray, probability: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        "rows": int(y_true.size),
        "positives": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "average_precision": float(average_precision_score(y_true, probability)),
        "roc_auc": safe_roc_auc(y_true, probability),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_roc_plot(
    y_true: np.ndarray,
    probability: np.ndarray,
    system_name: str,
    window_hours: float,
    output_path: Path,
) -> None:
    """Save the final held-out ROC curve as a vector PDF."""
    if np.unique(y_true).size != 2:
        raise ValueError(f"Cannot plot ROC curve for {system_name}: test set has one class.")

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probability)
    auc = roc_auc_score(y_true, probability)
    figure, axis = plt.subplots(figsize=(6.4, 5.2))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        linewidth=2,
        label=f"Random forest (AUC = {auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="0.5", label="Chance")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"{system_name.upper()} repeat automatic outage ROC ({window_hours:g}h window)",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_path, format="pdf", bbox_inches="tight")
    plt.close(figure)


def run_system(
    spec: SourceSpec, output_root: Path, n_estimators: int, n_jobs: int
) -> dict:
    print(f"\n[{spec.name.upper()}] Reading {spec.path}")
    frame = add_repeat_outage_label(read_source(spec))
    train, validation, test, split_metadata = chronological_split(frame)
    y = frame["target_repeat_auto_30d"].fillna(0).to_numpy(dtype=np.int8)

    system_output = output_root / spec.name
    system_output.mkdir(parents=True, exist_ok=True)
    window_rows: list[dict] = []
    validation_thresholds: dict[float, float] = {}

    for window_hours in WINDOW_HOURS:
        print(f"[{spec.name.upper()}] Fitting {window_hours:g}h history window")
        features = build_features(frame, window_hours)
        model = make_model(n_estimators, n_jobs)
        model.fit(features.loc[train], y[train])
        probability = model.predict_proba(features.loc[validation])[:, 1]
        threshold = best_f1_threshold(y[validation], probability)
        validation_thresholds[window_hours] = threshold
        metrics = score_predictions(y[validation], probability, threshold)
        window_rows.append({"window_hours": window_hours, **metrics})

    window_metrics = pd.DataFrame(window_rows).sort_values("window_hours")
    # Average precision is the primary selection metric because repeat outages
    # are imbalanced. The smaller window wins an exact tie.
    best_row = window_metrics.sort_values(
        ["average_precision", "window_hours"], ascending=[False, True]
    ).iloc[0]
    best_window = float(best_row["window_hours"])
    threshold = validation_thresholds[best_window]

    print(f"[{spec.name.upper()}] Selected {best_window:g}h; refitting on train + validation")
    best_features = build_features(frame, best_window)
    final_model = make_model(n_estimators, n_jobs)
    train_validation = train | validation
    final_model.fit(best_features.loc[train_validation], y[train_validation])
    test_probability = final_model.predict_proba(best_features.loc[test])[:, 1]
    test_metrics = score_predictions(y[test], test_probability, threshold)
    save_roc_plot(
        y[test],
        test_probability,
        spec.name,
        best_window,
        system_output / "test_roc_curve.pdf",
    )

    window_metrics.to_csv(system_output / "window_validation_metrics.csv", index=False)
    joblib.dump(final_model, system_output / "random_forest.joblib")

    predictions = frame.loc[test, ["source_row", "timestamp", "line_id"]].copy()
    predictions["target_repeat_auto_30d"] = y[test]
    predictions["probability"] = test_probability
    predictions["prediction"] = (test_probability >= threshold).astype(np.int8)
    predictions.to_csv(system_output / "test_predictions.csv", index=False)

    transformed_names = final_model.named_steps["imputer"].get_feature_names_out(FEATURE_COLUMNS)
    importance = pd.DataFrame(
        {
            "feature": transformed_names,
            "importance": final_model.named_steps["random_forest"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importance.to_csv(system_output / "feature_importance.csv", index=False)

    has_valid_identity = frame["timestamp"].notna() & frame["line_id"].notna()
    audit = {
        "input_rows": int(len(frame)),
        "rows_with_valid_timestamp_and_line": int(has_valid_identity.sum()),
        "rows_without_valid_timestamp_or_line": int((~has_valid_identity).sum()),
        "valid_rows_without_complete_30_day_follow_up": int(
            (has_valid_identity & frame["target_repeat_auto_30d"].isna()).sum()
        ),
        "labeled_rows": int(frame["target_repeat_auto_30d"].notna().sum()),
        "duplicate_records_removed": 0,
    }
    metadata = {
        "system": spec.name,
        "input_file": str(spec.path),
        "seed": SEED,
        "follow_up_days": FOLLOW_UP_DAYS,
        "candidate_history_windows_hours": list(WINDOW_HOURS),
        "selection_metric": "validation_average_precision",
        "selected_window_hours": best_window,
        "validation_threshold_selected_for_f1": threshold,
        "split": split_metadata,
        "audit": audit,
        "test_metrics": test_metrics,
        "features": FEATURE_COLUMNS,
    }
    with (system_output / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)

    print(
        f"[{spec.name.upper()}] Test AP={test_metrics['average_precision']:.4f}, "
        f"ROC-AUC={test_metrics['roc_auc']:.4f}, F1={test_metrics['f1']:.4f}"
    )
    return metadata


def write_combined_summary(results: Iterable[dict], output_root: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "system": result["system"],
                "selected_window_hours": result["selected_window_hours"],
                **result["test_metrics"],
            }
        )
    pd.DataFrame(rows).to_csv(output_root / "test_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    if args.n_estimators < 1:
        raise ValueError("--n-estimators must be at least 1")

    random.seed(SEED)
    np.random.seed(SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_system(SOURCES[name], args.output_dir, args.n_estimators, args.n_jobs)
        for name in args.systems
    ]
    write_combined_summary(results, args.output_dir)
    print(f"\nSaved results under {args.output_dir}")


if __name__ == "__main__":
    main()
