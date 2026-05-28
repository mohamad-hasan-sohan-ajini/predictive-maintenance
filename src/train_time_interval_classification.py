from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "output"
OUTPUT_DIR = BASE_DIR / "model_outputs" / "time_interval_rf"

CSV_PATTERN = "dataset_winsize*.csv"
RANDOM_STATE = 42
FOLD_COLUMN = "fold_id"
HELD_OUT_FOLD_ID = -1

TIME_LABEL = "label_time_to_event_seconds"
INTERVAL_TARGET = "label_time_to_event_interval"

CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_LABELS = ["[0,1)", "[1,2)", "[2,4)", "[4,8)", "[8,inf)"]
CLASS_NAME_BY_ID = dict(zip(CLASS_IDS, CLASS_LABELS))

DROP_COLUMNS = [
    FOLD_COLUMN,
    "label_is_auto",
    TIME_LABEL,
    INTERVAL_TARGET,
    "label_from_zone",
    "label_to_zone",
]
NUM_ESTIMATORS = 100  # Reduced from 300 for faster training during development


def quantize_time_to_event(seconds):
    """Map seconds to hour intervals: [0,1), [1,2), [2,4), [4,8), [8,inf)."""
    hours = seconds / 3600.0
    return pd.cut(
        hours,
        bins=[0, 1, 2, 4, 8, np.inf],
        labels=CLASS_IDS,
        right=False,
        include_lowest=True,
    ).astype("Int64")


def prepare_features(df):
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    return X.select_dtypes(include=[np.number, "bool"]).copy()


def interval_score(y_true, y_pred):
    """Score ordered-class predictions by normalized class distance."""
    distance = np.abs(np.asarray(y_true, dtype=int) - np.asarray(y_pred, dtype=int))
    return 1.0 - distance / (len(CLASS_IDS) - 1)


def balanced_interval_accuracy(y_true, y_pred):
    scores = interval_score(y_true, y_pred)
    by_class = []

    for class_id in CLASS_IDS:
        class_mask = np.asarray(y_true, dtype=int) == class_id
        if class_mask.any():
            by_class.append(scores[class_mask].mean())

    return float(np.mean(by_class)) if by_class else np.nan


def per_class_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    row_totals = cm.sum(axis=1)

    accuracies = {}
    for class_id, label, total, correct in zip(
        CLASS_IDS,
        CLASS_LABELS,
        row_totals,
        np.diag(cm),
    ):
        accuracies[f"accuracy_{label}"] = correct / total if total else np.nan

    return accuracies


def balanced_class_accuracy(y_true, y_pred):
    class_accuracies = per_class_accuracy(y_true, y_pred).values()
    present_class_accuracies = [
        class_accuracy
        for class_accuracy in class_accuracies
        if not np.isnan(class_accuracy)
    ]
    return float(np.mean(present_class_accuracies))


def class_counts(series):
    counts = series.value_counts().reindex(CLASS_IDS, fill_value=0)
    return {
        f"n_{CLASS_NAME_BY_ID[class_id]}": int(counts.loc[class_id])
        for class_id in CLASS_IDS
    }


def make_model():
    return RandomForestClassifier(
        n_estimators=NUM_ESTIMATORS,
        max_depth=None,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def prediction_frame(y_true, y_pred, row_index=None):
    data = {
        "y_true": np.asarray(y_true, dtype=int),
        "y_true_interval": [CLASS_NAME_BY_ID[i] for i in y_true],
        "y_pred": np.asarray(y_pred, dtype=int),
        "y_pred_interval": [CLASS_NAME_BY_ID[i] for i in y_pred],
        "interval_score": interval_score(y_true, y_pred),
    }

    if row_index is not None:
        data = {"row_index": row_index, **data}

    return pd.DataFrame(data)


def save_prediction_outputs(out_dir, y_true, y_pred, prediction_filename):
    out_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=CLASS_IDS,
        target_names=CLASS_LABELS,
        zero_division=0,
    )
    (out_dir / "rf_classification_report.txt").write_text(report)

    cm = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=CLASS_IDS),
        index=CLASS_LABELS,
        columns=CLASS_LABELS,
    )
    cm.to_csv(out_dir / "rf_confusion_matrix.csv")

    prediction_frame(y_true, y_pred).to_csv(out_dir / prediction_filename, index=False)


def build_metrics(dataset_name, y_true, y_pred, n_train):
    metrics = {
        "dataset": dataset_name,
        "n_train": n_train,
    }

    metrics.update(
        {
            "n_test": len(y_true),
            "accuracy": accuracy_score(y_true, y_pred),
            "balanced_accuracy": balanced_class_accuracy(y_true, y_pred),
            "interval_weighted_accuracy": interval_score(y_true, y_pred).mean(),
            "balanced_interval_weighted_accuracy": balanced_interval_accuracy(
                y_true,
                y_pred,
            ),
        }
    )
    metrics.update(class_counts(pd.Series(y_true)))
    metrics.update(per_class_accuracy(y_true, y_pred))

    return metrics


def train_dataset(csv_path):
    dataset_name = csv_path.stem
    df = pd.read_csv(csv_path)
    df[INTERVAL_TARGET] = quantize_time_to_event(df[TIME_LABEL])
    df = df[df[INTERVAL_TARGET].notna()].copy()

    if FOLD_COLUMN not in df.columns:
        raise ValueError(f"{csv_path} does not contain required column {FOLD_COLUMN!r}")

    train_df = df[df[FOLD_COLUMN] != HELD_OUT_FOLD_ID].copy()
    test_df = df[df[FOLD_COLUMN] == HELD_OUT_FOLD_ID].copy()

    if len(train_df) == 0:
        raise ValueError(
            f"{csv_path} has no training rows outside fold {HELD_OUT_FOLD_ID}"
        )
    if len(test_df) == 0:
        raise ValueError(
            f"{csv_path} has no held-out rows with fold {HELD_OUT_FOLD_ID}"
        )

    X_train = prepare_features(train_df)
    X_test = prepare_features(test_df)
    y_train = train_df[INTERVAL_TARGET].astype(int)
    y_test = test_df[INTERVAL_TARGET].astype(int)

    model = make_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    dataset_out_dir = OUTPUT_DIR / dataset_name
    dataset_out_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, dataset_out_dir / "rf.joblib")
    save_prediction_outputs(dataset_out_dir, y_test, y_pred, "rf_test_predictions.csv")

    return build_metrics(dataset_name, y_test, y_pred, n_train=len(y_train))


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(DATA_DIR.glob(CSV_PATTERN))
    if not csv_files:
        raise FileNotFoundError(f"No files found matching {CSV_PATTERN} in {DATA_DIR}")

    metrics = [train_dataset(csv_path) for csv_path in csv_files]
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(OUTPUT_DIR / "all_interval_metrics.csv", index=False)

    print(metrics_df.to_string(index=False))
    print(f"\nSaved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
