"""Training module for model selection and hyperparameter tuning."""

from pathlib import Path
from typing import Dict, List, Optional, Union
import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
import joblib

from src.data import load_data, get_feature_target
from src.preprocessing import build_preprocessing_pipeline
from src.evaluate import (
    compute_metrics,
    compute_confusion_matrix,
    threshold_analysis,
    find_best_threshold_by_f1,
    find_threshold_for_recall,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve,
    plot_threshold_analysis,
    save_metrics,
    save_model_comparison,
)


RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SCORING = "recall"


MODEL_PARAM_GRIDS = {
    "Logistic Regression": {
        "model": LogisticRegression(random_state=RANDOM_SEED, max_iter=1000, class_weight="balanced"),
        "params": {
            "clf__C": [0.01, 0.1, 1, 10, 100],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs"],
        },
    },
    "Decision Tree": {
        "model": DecisionTreeClassifier(random_state=RANDOM_SEED, class_weight="balanced"),
        "params": {
            "clf__max_depth": [3, 5, 7, 10, None],
            "clf__min_samples_split": [2, 5, 10],
            "clf__min_samples_leaf": [1, 2, 4],
        },
    },
    "Random Forest": {
        "model": RandomForestClassifier(random_state=RANDOM_SEED, class_weight="balanced", n_jobs=-1),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__max_depth": [5, 10, None],
            "clf__min_samples_split": [2, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__max_features": ["sqrt", "log2"],
        },
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=RANDOM_SEED),
        "params": {
            "clf__n_estimators": [100, 200],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth": [3, 5],
            "clf__min_samples_leaf": [1, 2],
            "clf__subsample": [0.8, 1.0],
        },
    },
}


def train_models(
    data_path: Optional[Union[Path, str]] = None,
    random_seed: int = RANDOM_SEED,
    test_size: float = TEST_SIZE,
    cv_folds: int = CV_FOLDS,
    scoring: str = SCORING,
) -> Dict:
    """Run complete training pipeline: split, tune, evaluate, save.

    Args:
        data_path: Path to data file. Defaults to data/ai4i2020.csv.
        random_seed: Random seed for reproducibility.
        test_size: Fraction of data for test set.
        cv_folds: Number of CV folds.
        scoring: Scoring metric for hyperparameter tuning.

    Returns:
        Dictionary with training results.
    """
    warnings.filterwarnings("ignore", category=UserWarning)

    # Load and prepare data
    df = load_data(data_path)
    X, y = get_feature_target(df, exclude_failure_modes=True, exclude_identifiers=True)

    print(f"Dataset shape: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    print(f"Failure rate: {y.mean():.4f}")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_seed, stratify=y
    )

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Train failure rate: {y_train.mean():.4f}, Test failure rate: {y_test.mean():.4f}")

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline()

    # Cross-validation setup
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)

    results = {}
    best_estimators = {}
    cv_results = {}

    # Train and tune each model
    for name, config in MODEL_PARAM_GRIDS.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")

        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", config["model"]),
            ]
        )

        grid = GridSearchCV(
            pipe,
            config["params"],
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=0,
        )

        grid.fit(X_train, y_train)

        best_estimators[name] = grid.best_estimator_
        cv_results[name] = {
            "best_params": grid.best_params_,
            "best_cv_score": float(grid.best_score_),
            "cv_results": grid.cv_results_,
        }

        print(f"Best params: {grid.best_params_}")
        print(f"Best CV {scoring}: {grid.best_score_:.4f}")

    # Evaluate on test set
    print(f"\n{'='*50}")
    print("Evaluating on test set...")
    print(f"{'='*50}")

    test_results = {}
    test_predictions = {}

    for name, estimator in best_estimators.items():
        y_prob = estimator.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = compute_metrics(y_test.values, y_pred, y_prob)
        cm = compute_confusion_matrix(y_test.values, y_pred)

        # Threshold analysis
        df_thresh = threshold_analysis(y_test.values, y_prob)
        best_thresh_f1 = find_best_threshold_by_f1(y_test.values, y_prob)
        best_thresh_recall = find_threshold_for_recall(y_test.values, y_prob, target_recall=0.8)

        test_results[name] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "best_threshold_f1": best_thresh_f1,
            "best_threshold_recall_80": best_thresh_recall,
            "threshold_analysis": df_thresh.to_dict(orient="records"),
        }

        test_predictions[name] = {
            "y_true": y_test.values,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }

        print(f"{name}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        print(f"  Best threshold (F1): {best_thresh_f1:.3f}")
        print(f"  Best threshold (recall>=0.8): {best_thresh_recall:.3f}")

    # Model comparison table
    comparison_rows = []
    for name, res in test_results.items():
        m = res["metrics"]
        row = {
            "Model": name,
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        }
        comparison_rows.append(row)

    comparison_df = pd.DataFrame(comparison_rows)
    print(f"\n{'='*50}")
    print("Model Comparison (Test Set)")
    print(f"{'='*50}")
    print(comparison_df.to_string(index=False))

    # Select best model based on recall (primary) then F1
    best_model_name = comparison_df.sort_values(["Recall", "F1"], ascending=False).iloc[0]["Model"]
    print(f"\nSelected best model: {best_model_name}")

    best_estimator = best_estimators[best_model_name]
    best_test_result = test_results[best_model_name]
    best_predictions = test_predictions[best_model_name]

    # Save results
    results_dir = Path(__file__).parents[1] / "results"
    results_dir.mkdir(exist_ok=True)

    models_dir = Path(__file__).parents[1] / "models"
    models_dir.mkdir(exist_ok=True)

    # Save model comparison
    save_model_comparison(comparison_df, results_dir / "model_comparison.csv")

    # Save test metrics for best model
    save_metrics(best_test_result["metrics"], results_dir / "test_metrics.json")

    # Save threshold analysis for best model
    pd.DataFrame(best_test_result["threshold_analysis"]).to_csv(
        results_dir / "threshold_analysis.csv", index=False
    )

    # Save best model
    joblib.dump(best_estimator, models_dir / "final_model.joblib")

    # Save feature names
    feature_names = best_estimator.named_steps["preprocessor"].get_feature_names_out().tolist()
    with open(results_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

    # Generate plots
    y_true = best_predictions["y_true"]
    y_prob = best_predictions["y_prob"]
    y_pred = best_predictions["y_pred"]

    plot_confusion_matrix(
        np.array(best_test_result["confusion_matrix"]),
        save_path=results_dir / "confusion_matrix.png",
    )

    plot_roc_curve(y_true, y_prob, save_path=results_dir / "roc_curve.png")
    plot_precision_recall_curve(y_true, y_prob, save_path=results_dir / "precision_recall_curve.png")

    df_thresh = pd.DataFrame(best_test_result["threshold_analysis"])
    plot_threshold_analysis(df_thresh, save_path=results_dir / "threshold_analysis.png")

    # Feature importance for tree-based models
    if hasattr(best_estimator.named_steps["clf"], "feature_importances_"):
        importances = best_estimator.named_steps["clf"].feature_importances_
        fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
        fi_df = fi_df.sort_values("importance", ascending=False)
        fi_df.to_csv(results_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(8, 6))
        top_n = min(15, len(fi_df))
        sns.barplot(data=fi_df.head(top_n), x="importance", y="feature")
        plt.title(f"Top {top_n} Feature Importances - {best_model_name}")
        plt.tight_layout()
        plt.savefig(results_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    elif hasattr(best_estimator.named_steps["clf"], "coef_"):
        coef = best_estimator.named_steps["clf"].coef_[0]
        fi_df = pd.DataFrame({"feature": feature_names, "coefficient": coef})
        fi_df["abs_coef"] = fi_df["coefficient"].abs()
        fi_df = fi_df.sort_values("abs_coef", ascending=False)
        fi_df.to_csv(results_dir / "feature_importance.csv", index=False)

        plt.figure(figsize=(8, 6))
        top_n = min(15, len(fi_df))
        colors = ["red" if c < 0 else "blue" for c in fi_df.head(top_n)["coefficient"]]
        sns.barplot(data=fi_df.head(top_n), x="coefficient", y="feature", palette=colors)
        plt.title(f"Top {top_n} Coefficients - {best_model_name}")
        plt.tight_layout()
        plt.savefig(results_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
        plt.close()

    # Save training summary
    m = best_test_result["metrics"]
    summary = {
        "dataset_shape": list(X.shape),
        "target_distribution": y.value_counts().to_dict(),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_seed": random_seed,
        "cv_folds": cv_folds,
        "scoring": scoring,
        "models_tested": list(MODEL_PARAM_GRIDS.keys()),
        "best_model": best_model_name,
        "best_model_cv_params": cv_results[best_model_name]["best_params"],
        "best_model_cv_score": cv_results[best_model_name]["best_cv_score"],
        "test_metrics": {
            "Accuracy": m["accuracy"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "ROC-AUC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
        },
        "selected_threshold_f1": best_test_result["best_threshold_f1"],
        "selected_threshold_recall_80": best_test_result["best_threshold_recall_80"],
    }

    with open(results_dir / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*50}")
    print("Training complete!")
    print(f"Results saved to {results_dir}")
    print(f"Model saved to {models_dir}/final_model.joblib")
    print(f"{'='*50}")

    return {
        "best_model_name": best_model_name,
        "best_estimator": best_estimator,
        "comparison_df": comparison_df,
        "test_results": test_results,
        "cv_results": cv_results,
        "summary": summary,
    }


if __name__ == "__main__":
    train_models()