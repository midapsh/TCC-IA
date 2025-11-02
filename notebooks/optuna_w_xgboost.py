from typing import NamedTuple, List, Dict, Any

from sklearn.metrics import mean_squared_error
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb


class Split(NamedTuple):
    train: np.ndarray
    test: np.ndarray
    purge: np.ndarray
    embargo: np.ndarray


def purged_time_splits_with_details(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon_hours: int = 1,
    embargo_hours: int = 24,
) -> List[Split]:
    """
    Create purged and embargoed time series splits with detailed information.
    Returns list of tuples: (train_indices, test_indices, purged_indices, embargo_indices)
    """
    y_times = pd.to_datetime(df["datetime"]).values
    t_sorted = np.sort(y_times)
    fold_edges = np.linspace(0, len(t_sorted), n_splits + 1, dtype=int)

    splits = []

    for k in range(n_splits):
        # Define test period
        test_start_idx = fold_edges[k]
        test_end_idx = fold_edges[k + 1] - 1
        test_start_time = t_sorted[test_start_idx]
        test_end_time = t_sorted[test_end_idx]

        # Test mask
        test_mask = (y_times >= test_start_time) & (y_times <= test_end_time)
        test_indices = np.where(test_mask)[0]

        # Define purge and embargo periods
        horizon = np.timedelta64(horizon_hours, "h")
        embargo = np.timedelta64(embargo_hours, "h")

        # Purge period: horizon before and after test period
        purge_start = test_start_time - horizon
        purge_end = test_end_time + horizon

        # Embargo period: after test period
        embargo_start = test_end_time
        embargo_end = test_end_time + embargo

        # Create masks
        purge_mask = (y_times >= purge_start) & (y_times <= purge_end)
        embargo_mask = (y_times > embargo_start) & (y_times <= embargo_end)

        # Train mask: everything except test, purged, and embargoed
        train_mask = ~test_mask & ~purge_mask & ~embargo_mask

        # Get indices
        train_indices = np.where(train_mask)[0]
        purged_indices = np.where(purge_mask & ~test_mask)[0]  # Purged but not test
        embargo_indices = np.where(embargo_mask)[0]

        splits.append(
            Split(train_indices, test_indices, purged_indices, embargo_indices)
        )

    return splits


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create time-based features"""
    df_features = df.copy()
    df_features["hour"] = df_features["datetime"].dt.hour
    df_features["day_of_week"] = df_features["datetime"].dt.dayofweek
    df_features["day_of_year"] = df_features["datetime"].dt.dayofyear

    # Lag features
    df_features["pressure_lag_1"] = df_features[
        "pressao_atmosferica_ao_nivel_da_estacao_horaria"
    ].shift(1)
    df_features["pressure_lag_24"] = df_features[
        "pressao_atmosferica_ao_nivel_da_estacao_horaria"
    ].shift(24)

    # Rolling features
    df_features["pressure_rolling_mean_24"] = (
        df_features["pressao_atmosferica_ao_nivel_da_estacao_horaria"]
        .rolling(24)
        .mean()
    )
    df_features["pressure_rolling_std_24"] = (
        df_features["pressao_atmosferica_ao_nivel_da_estacao_horaria"].rolling(24).std()
    )

    return df_features


def optimize_with_optuna(
    df: pd.DataFrame,
    splits: List[Split],
    feature_cols: List[str],
    target_col: str,
    n_trials: int = 10,
) -> Dict[str, Any]:
    """
    Optimize XGBoost hyperparameters using Optuna with purged time series splits

    Returns:
        dict: Best parameters found by Optuna
    """

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }

        rmse_scores = []

        for train_idx, test_idx, _, _ in splits:
            X_train = df.iloc[train_idx][feature_cols]
            y_train = df.iloc[train_idx][target_col]
            X_test = df.iloc[test_idx][feature_cols]
            y_test = df.iloc[test_idx][target_col]

            model = xgb.XGBRegressor(**params, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            rmse_scores.append(rmse)

        return np.mean(rmse_scores)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params


def train_and_evaluate_xgboost(
    df: pd.DataFrame,
    splits: List[Split],
    feature_cols: List[str],
    target_col: str,
    model_params: Dict[str, Any] | None = None,
) -> List[Dict[str, Any]]:
    """
    Train XGBoost models using purged time splits and evaluate performance

    Args:
        model_params: Optional parameters to pass to XGBRegressor
    """
    results = []

    for i, (train_idx, test_idx, purged_idx, embargo_idx) in enumerate(splits):
        print(f"\n=== Split {i+1} ===")

        # Prepare data
        X_train = df.iloc[train_idx][feature_cols]
        y_train = df.iloc[train_idx][target_col]
        X_test = df.iloc[test_idx][feature_cols]
        y_test = df.iloc[test_idx][target_col]

        # Train XGBoost model
        if model_params:
            model = xgb.XGBRegressor(**model_params, random_state=42, verbosity=0)
        else:
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                verbosity=0,
            )

        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        # Store results
        result = {
            "split": i + 1,
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "purged_size": len(purged_idx),
            "embargo_size": len(embargo_idx),
            "rmse": rmse,
            "train_period": (
                df.iloc[train_idx]["datetime"].min(),
                df.iloc[train_idx]["datetime"].max(),
            ),
            "test_period": (
                df.iloc[test_idx]["datetime"].min(),
                df.iloc[test_idx]["datetime"].max(),
            ),
            "model": model,
            "predictions": y_pred,
            "actual": y_test.values,
        }

        results.append(result)

        print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")
        print(f"Purged samples: {len(purged_idx)}, Embargo samples: {len(embargo_idx)}")
        print(f"RMSE: {rmse:.4f}")

        # Feature importance for first split
        if i == 0:
            importance = model.feature_importances_
            feature_importance = pd.DataFrame(
                {"feature": feature_cols, "importance": importance}
            ).sort_values("importance", ascending=False)
            print(f"\nFeature Importance (Split 1):")
            print(feature_importance)

    return results


def plot_graphs(
    df_clean: pd.DataFrame,
    splits: List[Split],
    feature_cols: List[str],
    target_col: str,
    results: List[Dict[str, Any]],
    optimized: bool = False,
):
    import matplotlib.pyplot as plt

    # Visualize the splits and results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(
        "Purged Time Series Cross-Validation Results"
        + (" (Optimized)" if optimized else ""),
        fontsize=16,
    )

    # Plot 1: Time series with split visualization for first split
    ax1 = axes[0, 0]
    train_idx, test_idx, purged_idx, embargo_idx = splits[0]

    ax1.plot(
        df_clean["datetime"],
        df_clean[target_col],
        alpha=0.7,
        color="lightgray",
        label="All data",
    )
    ax1.scatter(
        df_clean.iloc[train_idx]["datetime"],
        df_clean.iloc[train_idx][target_col],
        alpha=0.6,
        s=1,
        color="blue",
        label=f"Train ({len(train_idx)})",
    )
    ax1.scatter(
        df_clean.iloc[test_idx]["datetime"],
        df_clean.iloc[test_idx][target_col],
        alpha=0.8,
        s=2,
        color="red",
        label=f"Test ({len(test_idx)})",
    )
    if len(purged_idx) > 0:
        ax1.scatter(
            df_clean.iloc[purged_idx]["datetime"],
            df_clean.iloc[purged_idx][target_col],
            alpha=0.8,
            s=3,
            color="orange",
            label=f"Purged ({len(purged_idx)})",
        )
    if len(embargo_idx) > 0:
        ax1.scatter(
            df_clean.iloc[embargo_idx]["datetime"],
            df_clean.iloc[embargo_idx][target_col],
            alpha=0.8,
            s=3,
            color="purple",
            label=f"Embargo ({len(embargo_idx)})",
        )

    ax1.set_title("Split 1: Train/Test/Purge/Embargo Visualization")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Atmospheric Pressure")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: RMSE across splits
    ax2 = axes[0, 1]
    splits_num = [r["split"] for r in results]
    rmse_values = [r["rmse"] for r in results]
    ax2.bar(splits_num, rmse_values, color="skyblue", alpha=0.7)
    ax2.axhline(
        y=np.mean(rmse_values),
        color="red",
        linestyle="--",
        label=f"Mean RMSE: {np.mean(rmse_values):.3f}",
    )
    ax2.set_title("RMSE Across Time Splits")
    ax2.set_xlabel("Split Number")
    ax2.set_ylabel("RMSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Actual vs Predicted for first split
    ax3 = axes[1, 0]
    actual = results[0]["actual"]
    predicted = results[0]["predictions"]
    ax3.scatter(actual, predicted, alpha=0.6, color="green")
    ax3.plot([actual.min(), actual.max()], [actual.min(), actual.max()], "r--", lw=2)
    ax3.set_title(f'Split 1: Actual vs Predicted (RMSE: {results[0]["rmse"]:.3f})')
    ax3.set_xlabel("Actual Pressure")
    ax3.set_ylabel("Predicted Pressure")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Feature importance
    ax4 = axes[1, 1]
    importance_data = results[0]["model"].feature_importances_
    feature_names = feature_cols
    sorted_idx = np.argsort(importance_data)
    ax4.barh(
        range(len(importance_data)), importance_data[sorted_idx], color="lightcoral"
    )
    ax4.set_yticks(range(len(importance_data)))
    ax4.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax4.set_title("Feature Importance (Split 1)")
    ax4.set_xlabel("Importance")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("purged_xgboost_results.png")
    plt.close()


def main():
    # 1. Load your data - replace this with your actual data loading
    # df = get_bauru_df()

    # For demonstration: create sample data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=1000, freq="H")
    pressure = (
        1013.25
        + np.random.normal(0, 5, 1000)
        + 10 * np.sin(np.arange(1000) * 2 * np.pi / 24)
    )
    df = pd.DataFrame(
        {"datetime": dates, "pressao_atmosferica_ao_nivel_da_estacao_horaria": pressure}
    )

    print("Sample data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # 2. Create features
    df_with_features = create_features(df)
    df_clean = df_with_features.dropna().reset_index(drop=True)

    print(f"\nClean data shape: {df_clean.shape}")
    print("\nFeature columns:")
    feature_cols = [
        col
        for col in df_clean.columns
        if col not in ["datetime", "pressao_atmosferica_ao_nivel_da_estacao_horaria"]
    ]
    print(feature_cols)

    # 3. Create purged splits
    splits = purged_time_splits_with_details(
        df_clean, n_splits=5, horizon_hours=1, embargo_hours=24
    )

    print(f"\nNumber of splits: {len(splits)}")

    # 4. Baseline model evaluation
    print("\n" + "=" * 60)
    print("BASELINE MODEL EVALUATION")
    print("=" * 60)
    target_col = "pressao_atmosferica_ao_nivel_da_estacao_horaria"
    baseline_results = train_and_evaluate_xgboost(
        df_clean, splits, feature_cols, target_col
    )

    # Baseline summary
    baseline_rmse = np.mean([r["rmse"] for r in baseline_results])
    print(f"\nBaseline Average RMSE: {baseline_rmse:.4f}")

    # 5. Hyperparameter optimization with Optuna
    print("\n" + "=" * 60)
    print("HYPERPARAMETER OPTIMIZATION WITH OPTUNA")
    print("=" * 60)
    best_params = optimize_with_optuna(
        df_clean, splits, feature_cols, target_col, n_trials=10
    )
    print(f"\nBest parameters: {best_params}")

    # 6. Optimized model evaluation
    print("\n" + "=" * 60)
    print("OPTIMIZED MODEL EVALUATION")
    print("=" * 60)
    optimized_results = train_and_evaluate_xgboost(
        df_clean, splits, feature_cols, target_col, model_params=best_params
    )

    # Optimized summary
    optimized_rmse = np.mean([r["rmse"] for r in optimized_results])
    improvement = (baseline_rmse - optimized_rmse) / baseline_rmse * 100
    print(f"\nOptimized Average RMSE: {optimized_rmse:.4f}")
    print(f"Improvement over baseline: {improvement:.2f}%")

    # 7. Visualization
    plot_graphs(
        df_clean, splits, feature_cols, target_col, baseline_results, optimized=False
    )
    plot_graphs(
        df_clean, splits, feature_cols, target_col, optimized_results, optimized=True
    )

    # Final model training on full dataset
    print("\n" + "=" * 60)
    print("FINAL MODEL TRAINING")
    print("=" * 60)
    final_model = xgb.XGBRegressor(**best_params, random_state=42)
    final_model.fit(df_clean[feature_cols], df_clean[target_col])

    # Feature importance
    importance = final_model.feature_importances_
    feature_importance = pd.DataFrame(
        {"feature": feature_cols, "importance": importance}
    ).sort_values("importance", ascending=False)

    print("\nFinal Model Feature Importance:")
    print(feature_importance)


if __name__ == "__main__":
    main()
