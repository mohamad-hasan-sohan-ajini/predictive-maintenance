from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_fscore_support,
    r2_score,
)
from sklearn.tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    export_text,
    plot_tree,
)


def get_X_y(df, target_col, drop_columns=):
    """
    Prepare features and target.
    Drops all label columns and fold_id from X.
    """
    X = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])
    y = df[target_col]

    # Basic safety: keep only numeric columns
    X = X.select_dtypes(include=[np.number, "bool"]).copy()

    return X, y


def split_by_fold_id(df):
    """
    Uses fold_id = -1 as held-out test set.
    Uses fold_id in {0,1,2,3,4} as training set.
    """
    if "fold_id" not in df.columns:
        raise ValueError("The dataset must contain a 'fold_id' column.")

    train_df = df[df["fold_id"] != -1].copy()
    test_df = df[df["fold_id"] == -1].copy()

    if len(test_df) == 0:
        raise ValueError(
            "No held-out test rows found. Expected fold_id == -1 for test set."
        )

    return train_df, test_df


def classification_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    bal_acc = balanced_accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    precision_weighted, recall_weighted, f1_weighted, _ = (
        precision_recall_fscore_support(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )
    )

    return {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
    }


def regression_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "median_absolute_error": median_absolute_error(y_true, y_pred),
        "mse": mse,
        "rmse": rmse,
        "r2": r2_score(y_true, y_pred),
    }


def save_tree_plot(model, feature_names, out_path, max_depth=4):
    """
    Visualises a tree.
    For a full deep tree, the plot may become unreadable, so max_depth is limited.
    """
    plt.figure(figsize=(24, 12))
    plot_tree(
        model,
        feature_names=feature_names,
        filled=True,
        rounded=True,
        max_depth=max_depth,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=250)
    plt.close()


def save_rf_example_trees(rf_model, feature_names, out_dir, prefix, max_trees=3):
    """
    Visualises a few individual trees from a random forest.
    """
    n = min(max_trees, len(rf_model.estimators_))

    for i in range(n):
        out_path = out_dir / f"{prefix}_rf_tree_{i}.png"
        save_tree_plot(
            rf_model.estimators_[i],
            feature_names=feature_names,
            out_path=out_path,
            max_depth=4,
        )


def save_builtin_feature_importance(model, feature_names, out_path):
    """
    Saves Gini/MSE-based feature importance from tree/RF.
    """
    if not hasattr(model, "feature_importances_"):
        return None

    importance = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance.to_csv(out_path, index=False)
    return importance


def save_permutation_importance(model, X_test, y_test, task_type, out_path):
    """
    Permutation importance is usually more reliable than built-in tree importance.
    """
    if task_type == "classification":
        scoring = "f1_weighted"
    else:
        scoring = "neg_root_mean_squared_error"

    result = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_test.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)

    importance.to_csv(out_path, index=False)
    return importance


def save_shap_importance(model, X_sample, out_path_csv, out_path_png, task_type):
    """
    Computes SHAP values for tree-based models.

    For multiclass classifiers, SHAP returns one explanation per class.
    Here we average absolute SHAP values over samples and classes.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        # Multiclass classification: list[class] of arrays
        mean_abs_shap = np.mean(
            [np.abs(class_shap).mean(axis=0) for class_shap in shap_values],
            axis=0,
        )
    else:
        # Regression or binary classification
        if shap_values.ndim == 3:
            # Some SHAP versions return shape: samples x features x classes
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2))
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "mean_abs_shap": mean_abs_shap,
        }
    ).sort_values("mean_abs_shap", ascending=False)

    shap_importance.to_csv(out_path_csv, index=False)

    # SHAP bar plot
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_sample,
        plot_type="bar",
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=250, bbox_inches="tight")
    plt.close()

    return shap_importance


def train_models_for_target(df, dataset_name, target_col, task_type):
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"Target: {target_col}")
    print(f"Task: {task_type}")
    print(f"{'=' * 80}")

    train_df, test_df = split_by_fold_id(df)

    X_train, y_train = get_X_y(train_df, target_col)
    X_test, y_test = get_X_y(test_df, target_col)

    # Remove rows with missing target
    train_mask = y_train.notna()
    test_mask = y_test.notna()

    X_train = X_train.loc[train_mask]
    y_train = y_train.loc[train_mask]

    X_test = X_test.loc[test_mask]
    y_test = y_test.loc[test_mask]

    target_out_dir = OUTPUT_DIR / dataset_name / target_col
    target_out_dir.mkdir(parents=True, exist_ok=True)

    feature_names = X_train.columns.tolist()

    if task_type == "classification":
        models = {
            "tree": DecisionTreeClassifier(
                max_depth=6,
                min_samples_leaf=10,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            ),
            "rf": RandomForestClassifier(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

    elif task_type == "regression":
        models = {
            "tree": DecisionTreeRegressor(
                max_depth=6,
                min_samples_leaf=10,
                random_state=RANDOM_STATE,
            ),
            "rf": RandomForestRegressor(
                n_estimators=300,
                max_depth=None,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

    else:
        raise ValueError(f"Unknown task type: {task_type}")

    all_metrics = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Save model
        joblib.dump(model, target_out_dir / f"{model_name}.joblib")

        # Metrics
        if task_type == "classification":
            metrics = classification_metrics(y_test, y_pred)

            report = classification_report(
                y_test,
                y_pred,
                zero_division=0,
            )

            with open(
                target_out_dir / f"{model_name}_classification_report.txt", "w"
            ) as f:
                f.write(report)

            cm = pd.DataFrame(confusion_matrix(y_test, y_pred))
            cm.to_csv(
                target_out_dir / f"{model_name}_confusion_matrix.csv", index=False
            )

            print(report)

        else:
            metrics = regression_metrics(y_test, y_pred)
            print(metrics)

        metrics["dataset"] = dataset_name
        metrics["target"] = target_col
        metrics["task"] = task_type
        metrics["model"] = model_name

        all_metrics.append(metrics)

        # Tree visualisation
        if model_name == "tree":
            save_tree_plot(
                model,
                feature_names,
                target_out_dir / f"{model_name}_visualisation.png",
                max_depth=4,
            )

            tree_text = export_text(model, feature_names=feature_names)
            with open(target_out_dir / f"{model_name}_rules.txt", "w") as f:
                f.write(tree_text)

        # RF tree visualisation
        if model_name == "rf":
            save_rf_example_trees(
                model,
                feature_names,
                target_out_dir,
                prefix=model_name,
                max_trees=3,
            )

        # Built-in feature importance
        builtin_importance = save_builtin_feature_importance(
            model,
            feature_names,
            target_out_dir / f"{model_name}_builtin_feature_importance.csv",
        )

        if builtin_importance is not None:
            print(f"\nTop built-in feature importances for {model_name}:")
            print(builtin_importance.head(10))

        # Permutation importance
        permutation_df = save_permutation_importance(
            model,
            X_test,
            y_test,
            task_type,
            target_out_dir / f"{model_name}_permutation_importance.csv",
        )

        print(f"\nTop permutation importances for {model_name}:")
        print(permutation_df.head(10))

        # SHAP importance
        # Use a sample for speed, especially if the dataset is large.
        shap_sample_size = min(1000, len(X_test))
        X_shap = X_test.sample(
            shap_sample_size,
            random_state=RANDOM_STATE,
        )

        shap_df = save_shap_importance(
            model,
            X_shap,
            target_out_dir / f"{model_name}_shap_importance.csv",
            target_out_dir / f"{model_name}_shap_summary.png",
            task_type,
        )

        print(f"\nTop SHAP importances for {model_name}:")
        print(shap_df.head(10))

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(target_out_dir / "metrics.csv", index=False)

    return metrics_df


# -----------------------------
# Main loop over datasets
# -----------------------------

all_results = []

csv_files = sorted(DATA_DIR.glob(CSV_PATTERN))

if len(csv_files) == 0:
    raise FileNotFoundError(
        f"No files found matching {CSV_PATTERN} in {DATA_DIR.resolve()}"
    )

for csv_path in csv_files:
    dataset_name = csv_path.stem
    print(f"\nReading {csv_path}")

    df = pd.read_csv(csv_path)

    for target_col, task_type in TARGETS.items():
        if target_col not in df.columns:
            print(f"Skipping missing target: {target_col}")
            continue

        metrics_df = train_models_for_target(
            df=df,
            dataset_name=dataset_name,
            target_col=target_col,
            task_type=task_type,
        )

        all_results.append(metrics_df)

final_results = pd.concat(all_results, ignore_index=True)
final_results.to_csv(OUTPUT_DIR / "all_metrics_summary.csv", index=False)

print("\nFinal summary:")
print(final_results)
