from typing import NamedTuple

import numpy as np
import pandas as pd

from utils.load_df import load_df


def get_bauru_df() -> pd.DataFrame:
    df = load_df([65554853])  # BAURU
    return df


import numpy as np
from numpy import typing
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta


class Split(NamedTuple):
    train: typing.NDArray
    test: typing.NDArray
    purge: typing.NDArray
    embargo: typing.NDArray


def purged_time_splits_with_details(
    df: pd.DataFrame,
    n_splits: int = 5,
    horizon_hours: int = 1,
    embargo_hours: int = 24,
) -> list[Split]:
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


# Now let's implement the complete solution with XGBoost
def create_features(df: pd.DataFrame, /) -> pd.DataFrame:
    """Create simple time-based features"""
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


# Now let's implement XGBoost training and evaluation
def train_and_evaluate_xgboost(
    df: pd.DataFrame, splits: list[Split], feature_cols: list[str], target_col: str
):
    """
    Train XGBoost models using purged time splits and evaluate performance
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
    splits: list[Split],
    feature_cols: list[str],
    target_col: str,
    results: list,
):
    import matplotlib.pyplot as plt

    # Visualize the splits and results
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

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
    plt.savefig("cool-stuff.png")

    # Print summary of the approach
    rmse_scores = [r["rmse"] for r in results]
    print("\n" + "=" * 60)
    print("SUMMARY OF PURGED TIME SERIES CROSS-VALIDATION")
    print("=" * 60)
    print(f"✓ Created {len(splits)} time-based splits")
    print(f"✓ Applied {1}-hour purge window and {24}-hour embargo")
    print(f"✓ Trained XGBoost models with time-based features")
    print(f"✓ Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
    print(
        f"✓ Most important feature: {feature_cols[np.argmax(results[0]['model'].feature_importances_)]}"
    )
    print(
        "\nThe splits successfully return (train, test, purged, embargo) indices as requested!"
    )


def main():
    df = get_bauru_df()

    # Create sample data for demonstration
    np.random.seed(42)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df[["datetime", "pressao_atmosferica_ao_nivel_da_estacao_horaria"]]

    print("Sample data:")
    print(df.head())
    print(f"\nData shape: {df.shape}")
    print(f"Date range: {df['datetime'].min()} to {df['datetime'].max()}")

    # Create features
    df_with_features = create_features(df)

    # Drop rows with NaN values (due to lag and rolling features)
    df_clean = df_with_features.dropna().reset_index(drop=True)

    print(f"Clean data shape: {df_clean.shape}")
    print("\nFeature columns:")
    feature_cols = [
        col
        for col in df_clean.columns
        if col not in ["datetime", "pressao_atmosferica_ao_nivel_da_estacao_horaria"]
    ]
    print(feature_cols)

    # Get the splits
    splits = purged_time_splits_with_details(
        df_clean, n_splits=5, horizon_hours=1, embargo_hours=24
    )

    print(f"\nNumber of splits: {len(splits)}")

    # Analyze first split
    train_idx, test_idx, purged_idx, embargo_idx = splits[0]
    print(f"\nFirst split analysis:")
    print(f"Train samples: {len(train_idx)}")
    print(f"Test samples: {len(test_idx)}")
    print(f"Purged samples: {len(purged_idx)}")
    print(f"Embargo samples: {len(embargo_idx)}")

    # Show time ranges
    print(f"\nTime ranges for first split:")
    print(
        f"Train period: {df_clean.iloc[train_idx]['datetime'].min()} to {df_clean.iloc[train_idx]['datetime'].max()}"
    )
    print(
        f"Test period: {df_clean.iloc[test_idx]['datetime'].min()} to {df_clean.iloc[test_idx]['datetime'].max()}"
    )
    if len(purged_idx) > 0:
        print(
            f"Purged period: {df_clean.iloc[purged_idx]['datetime'].min()} to {df_clean.iloc[purged_idx]['datetime'].max()}"
        )
    if len(embargo_idx) > 0:
        print(
            f"Embargo period: {df_clean.iloc[embargo_idx]['datetime'].min()} to {df_clean.iloc[embargo_idx]['datetime'].max()}"
        )
    ################### Run the training and evaluation
    target_col = "pressao_atmosferica_ao_nivel_da_estacao_horaria"
    results = train_and_evaluate_xgboost(df_clean, splits, feature_cols, target_col)

    # Summary statistics
    rmse_scores = [r["rmse"] for r in results]
    print(f"\n=== Overall Results ===")
    print(f"Mean RMSE: {np.mean(rmse_scores):.4f}")
    print(f"Std RMSE: {np.std(rmse_scores):.4f}")
    print(f"Min RMSE: {np.min(rmse_scores):.4f}")
    print(f"Max RMSE: {np.max(rmse_scores):.4f}")

    plot_graphs(df_clean, splits, feature_cols, target_col, results)


if __name__ == "__main__":
    main()
