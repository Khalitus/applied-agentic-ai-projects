"""Select an XGBoost configuration on past-only validation data and audit its errors."""

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

from src.config import (
    FEATURES,
    OPTIMIZED_MODEL_PATH,
    OPTIMIZED_PREDICTIONS_PATH,
    OPTIMIZATION_REPORT_PATH,
    REPORT_PATH,
    TARGET,
    VALIDATION_PREDICTIONS_PATH,
)
from src.data import load_hourly
from src.features import training_features
from src.train import baseline_predictions, chronological_split, regression_metrics


CANDIDATES = {
    "task6_baseline": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "shallow_regularized": {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 3,
        "reg_lambda": 3,
    },
    "deeper_regularized": {
        "n_estimators": 250,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.9,
        "min_child_weight": 2,
        "reg_lambda": 2,
    },
}

MIN_RELATIVE_IMPROVEMENT = 0.005


def make_model(parameters: dict) -> XGBRegressor:
    return XGBRegressor(
        **parameters,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=2,
    )


def predict_nonnegative(model: XGBRegressor, features: pd.DataFrame) -> np.ndarray:
    return np.maximum(model.predict(features), 0.0)


def choose_candidate(
    results: list[dict],
    minimum_relative_improvement: float = MIN_RELATIVE_IMPROVEMENT,
) -> str:
    """Keep Task 6 settings unless another candidate clears the prespecified gain."""
    if not 0 <= minimum_relative_improvement < 1:
        raise ValueError("minimum_relative_improvement must be in [0, 1)")

    scores = {result["name"]: float(result["mae"]) for result in results}
    if "task6_baseline" not in scores:
        raise ValueError("Missing Task 6 reference candidate")
    if any(not np.isfinite(value) or value < 0 for value in scores.values()):
        raise ValueError("Candidate MAE must be finite and nonnegative")

    best_name = min(scores, key=scores.get)
    baseline_mae = scores["task6_baseline"]
    if (
        best_name != "task6_baseline"
        and scores[best_name] < baseline_mae * (1 - minimum_relative_improvement)
    ):
        return best_name
    return "task6_baseline"


def prediction_frame(data: pd.DataFrame, predictions: np.ndarray) -> pd.DataFrame:
    """Create audit rows without using target values as prediction inputs."""
    if len(data) != len(predictions):
        raise ValueError("Prediction and observation lengths differ")

    columns = ["instant", "dteday", "hr", "workingday", "weathersit", TARGET]
    result = data.loc[:, columns].copy().reset_index(drop=True)
    result = result.rename(columns={TARGET: "actual_rentals"})
    result["predicted_rentals"] = np.asarray(predictions, dtype=float)
    result["signed_error"] = (
        result["predicted_rentals"] - result["actual_rentals"]
    )
    result["absolute_error"] = result["signed_error"].abs()
    return result


def error_profile(rows: pd.DataFrame, group: str) -> list[dict]:
    """Report error and signed bias by a known, fixed grouping column."""
    if group not in ("hr", "workingday", "weathersit", "demand_band"):
        raise ValueError("Unsupported error-analysis group")

    data = rows.copy()
    if group == "demand_band":
        data["demand_band"] = pd.cut(
            data["actual_rentals"],
            bins=[-np.inf, 100, 300, np.inf],
            labels=["low_0_99", "medium_100_299", "high_300_plus"],
            right=False,
        )

    grouped = (
        data.groupby(group, observed=True)
        .agg(
            observations=("absolute_error", "size"),
            mae=("absolute_error", "mean"),
            signed_bias=("signed_error", "mean"),
            mean_actual=("actual_rentals", "mean"),
            mean_prediction=("predicted_rentals", "mean"),
        )
        .reset_index()
    )

    results = []
    for record in grouped.to_dict(orient="records"):
        label = record[group]
        results.append({
            "group": str(label) if group == "demand_band" else int(label),
            "observations": int(record["observations"]),
            "mae": round(float(record["mae"]), 3),
            "signed_bias": round(float(record["signed_bias"]), 3),
            "mean_actual": round(float(record["mean_actual"]), 3),
            "mean_prediction": round(float(record["mean_prediction"]), 3),
        })
    return results


def analyze_errors(rows: pd.DataFrame) -> dict:
    worst = rows.nlargest(10, "absolute_error")[
        ["dteday", "hr", "workingday", "weathersit", "actual_rentals",
         "predicted_rentals", "signed_error", "absolute_error"]
    ].copy()
    for column in ("predicted_rentals", "signed_error", "absolute_error"):
        worst[column] = worst[column].round(3)

    return {
        "bias_definition": "predicted minus actual; positive means overprediction",
        "by_hour": error_profile(rows, "hr"),
        "by_workingday": error_profile(rows, "workingday"),
        "by_weather": error_profile(rows, "weathersit"),
        "by_actual_demand": error_profile(rows, "demand_band"),
        "largest_validation_errors": worst.to_dict(orient="records"),
    }


def task6_reference(train: pd.DataFrame, holdout: pd.DataFrame) -> dict | None:
    """Read the previous Task 6 score for reporting only, never for selection."""
    if not REPORT_PATH.exists():
        return None
    try:
        previous = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None

    expected = (
        previous.get("train_rows") == len(train)
        and previous.get("test_rows") == len(holdout)
        and previous.get("train_end") == str(train["dteday"].iloc[-1])
        and previous.get("test_start") == str(holdout["dteday"].iloc[0])
        and previous.get("features") == list(FEATURES)
    )
    if not expected:
        return None
    return {
        "mae": previous.get("model_mae"),
        "rmse": previous.get("model_rmse"),
        "r2": previous.get("model_r2"),
    }


def main() -> dict:
    data = load_hourly()
    full_train, holdout = chronological_split(data, train_fraction=0.8)
    fit, validation = chronological_split(full_train, train_fraction=0.8)

    x_fit, x_valid = training_features(fit), training_features(validation)
    y_fit, y_valid = fit[TARGET], validation[TARGET]

    validation_scores = []
    candidate_predictions = {}

    for name, parameters in CANDIDATES.items():
        model = make_model(parameters)
        model.fit(x_fit, y_fit)
        preds = predict_nonnegative(model, x_valid)
        score = regression_metrics(y_valid, preds)
        validation_scores.append({
            "name": name,
            "mae": float(mean_absolute_error(y_valid, preds)),
            "rmse": score["rmse"],
            "r2": round(float(r2_score(y_valid, preds)), 4),
        })
        candidate_predictions[name] = preds

    selected_name = choose_candidate(validation_scores)
    selected_params = CANDIDATES[selected_name]
    chosen_validation_preds = candidate_predictions[selected_name]

    # Diagnostic only: ablation never changes the fixed 10-feature application contract.
    without_peak = [name for name in FEATURES if name != "commute_peak"]
    ablated_model = make_model(selected_params)
    ablated_model.fit(x_fit[without_peak], y_fit)
    ablated_preds = predict_nonnegative(ablated_model, x_valid[without_peak])
    ablation = {
        "description": "Same selected settings, without commute_peak; diagnostic only",
        "with_commute_peak_mae": round(
            float(mean_absolute_error(y_valid, chosen_validation_preds)), 3
        ),
        "without_commute_peak_mae": round(
            float(mean_absolute_error(y_valid, ablated_preds)), 3
        ),
        "feature_set_changed": False,
    }

    validation_rows = prediction_frame(validation, chosen_validation_preds)
    validation_analysis = analyze_errors(validation_rows)

    # The final test period is not used to choose hyperparameters or features.
    final_model = make_model(selected_params)
    final_model.fit(training_features(full_train), full_train[TARGET])
    final_predictions = predict_nonnegative(final_model, training_features(holdout))
    final_rows = prediction_frame(holdout, final_predictions)
    global_mean, hourly_mean = baseline_predictions(full_train, holdout)

    holdout_metrics = regression_metrics(holdout[TARGET], final_predictions)
    holdout_metrics["r2"] = round(
        float(r2_score(holdout[TARGET], final_predictions)), 4
    )
    hourly_metrics = regression_metrics(holdout[TARGET], hourly_mean)
    mean_metrics = regression_metrics(holdout[TARGET], global_mean)
    holdout_metrics["mae_reduction_vs_hourly_pct"] = round(
        100 * (hourly_metrics["mae"] - holdout_metrics["mae"])
        / hourly_metrics["mae"], 2
    ) if hourly_metrics["mae"] > 0 else None

    importance = sorted(
        (
            {"feature": name, "importance": round(float(value), 5)}
            for name, value in zip(FEATURES, final_model.feature_importances_)
        ),
        key=lambda item: item["importance"],
        reverse=True,
    )

    report = {
        "dataset": "UCI Bike Sharing (hour.csv, 2011–2012)",
        "selection_rule": "lowest validation MAE, requiring >0.5% gain over Task 6 settings",
        "outer_split": {
            "training_rows": len(full_train),
            "holdout_rows": len(holdout),
            "training_end": str(full_train["dteday"].iloc[-1]),
            "holdout_start": str(holdout["dteday"].iloc[0]),
        },
        "inner_split": {
            "fit_rows": len(fit),
            "validation_rows": len(validation),
            "fit_end": str(fit["dteday"].iloc[-1]),
            "validation_start": str(validation["dteday"].iloc[0]),
        },
        "features": list(FEATURES),
        "validation_candidates": [
            {**entry, "mae": round(entry["mae"], 3)}
            for entry in validation_scores
        ],
        "selected_candidate": selected_name,
        "selected_parameters": selected_params,
        "commute_peak_ablation": ablation,
        "validation_error_analysis": validation_analysis,
        "holdout_optimized": holdout_metrics,
        "holdout_hourly_baseline": hourly_metrics,
        "holdout_global_mean_baseline": mean_metrics,
        "task6_holdout_reference": task6_reference(full_train, holdout),
        "feature_importance": importance,
        "note": (
            "Previous Task 6 holdout results were already inspected. "
            "Only inner-validation scores determined model selection; "
            "further tuning against the final holdout would bias evaluation."
        ),
    }

    for path in (
        OPTIMIZED_MODEL_PATH,
        OPTIMIZATION_REPORT_PATH,
        VALIDATION_PREDICTIONS_PATH,
        OPTIMIZED_PREDICTIONS_PATH,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(final_model, OPTIMIZED_MODEL_PATH)
    validation_rows.to_csv(
        VALIDATION_PREDICTIONS_PATH, index=False, float_format="%.4f"
    )
    final_rows.to_csv(
        OPTIMIZED_PREDICTIONS_PATH, index=False, float_format="%.4f"
    )
    OPTIMIZATION_REPORT_PATH.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(json.dumps({
        "selected_candidate": selected_name,
        "inner_split": report["inner_split"],
        "validation_candidates": report["validation_candidates"],
        "commute_peak_ablation": ablation,
        "holdout_optimized": holdout_metrics,
        "holdout_hourly_baseline": hourly_metrics,
        "task6_holdout_reference": report["task6_holdout_reference"],
        "top_features": importance[:5],
    }, indent=2))
    print(f"Optimized model saved: {OPTIMIZED_MODEL_PATH}")
    print(f"Optimization report saved: {OPTIMIZATION_REPORT_PATH}")
    print(f"Validation predictions saved: {VALIDATION_PREDICTIONS_PATH}")
    print(f"Holdout predictions saved: {OPTIMIZED_PREDICTIONS_PATH}")
    return report


if __name__ == "__main__":
    main()
