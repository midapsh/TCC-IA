#!/usr/bin/env python3
"""
fit_and_forecast.py

Bundle: denoise + model selection + forecasting per station.

Dependencies:
  - Python 3.9+
  - numpy, pandas, statsmodels

Install:
  pip install numpy pandas statsmodels

Example:
  python fit_and_forecast.py \
    --input data/all_stations.csv \
    --time-col timestamp --station-col station_id --value-col value \
    --horizon 48 --outdir outputs --freq auto --seasonal-periods auto
"""

import argparse
import json
import os
import warnings
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.statespace.sarimax import SARIMAX


# --------------------------- Utility & metrics ---------------------------


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE in percent."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return 200.0 * np.mean(np.abs(y_true - y_pred) / denom)


def infer_freq(index: pd.DatetimeIndex) -> Optional[str]:
    """Infer frequency; return None if ambiguous."""
    try:
        f = pd.infer_freq(index)
    except Exception:
        f = None
    return f


def default_seasonal_periods(freq: Optional[str]) -> int:
    """Heuristic seasonal periods by frequency."""
    if not freq:
        return 7  # sensible default for daily-ish data
    freq = to_offset(freq).name.upper()
    # Map common cases
    if freq == "H":
        return 24
    if freq == "T" or freq == "MIN":
        return 60
    if freq == "S":
        return 3600
    if freq == "D":
        return 7
    if freq == "W-SUN" or freq.startswith("W-") or freq == "W":
        return 52
    if freq == "M":
        return 12
    if freq == "MS":
        return 12
    if freq == "Q" or freq == "QS":
        return 4
    if freq == "A" or freq == "AS" or freq.startswith("Y"):
        return 1
    # Fallback
    return 7


def next_periods(index: pd.DatetimeIndex, steps: int, freq: str) -> pd.DatetimeIndex:
    last = index[-1]
    return pd.date_range(last + to_offset(freq), periods=steps, freq=freq)


def z_score_for(alpha_two_sided: float) -> float:
    """Return z for two-sided (e.g., alpha=0.20 -> z≈1.2816)."""
    # Simple lookup to avoid scipy dependency
    table = {
        0.20: 1.2815515655446004,
        0.10: 1.6448536269514722,
        0.05: 1.959963984540054,
    }
    # Nearest match
    keys = sorted(table.keys())
    closest = min(keys, key=lambda k: abs(k - alpha_two_sided))
    return table[closest]


# --------------------------- Denoising (STL) ---------------------------


@dataclass
class DenoiseResult:
    trend: pd.Series
    seasonal: pd.Series
    resid: pd.Series

    @property
    def denoised(self) -> pd.Series:
        return self.trend + self.seasonal


def stl_denoise(y: pd.Series, sp: int) -> DenoiseResult:
    """STL to separate trend/seasonal/remainder; robust for outliers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stl = STL(y, period=max(2, sp), robust=True)
        res = stl.fit()
    return DenoiseResult(trend=res.trend, seasonal=res.seasonal, resid=res.resid)


# --------------------------- Models & forecasts ---------------------------


def seasonal_naive_forecast(y: pd.Series, h: int, sp: int) -> np.ndarray:
    if sp <= 0 or len(y) < sp:
        # Fallback: naive (last value)
        return np.repeat(y.iloc[-1], h)
    last_season = y.iloc[-sp:].to_numpy()
    reps = int(np.ceil(h / sp))
    return np.tile(last_season, reps)[:h]


def snaive_intervals(
    y_fit_resid: np.ndarray, h: int, sp: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Approx intervals for s-naive via Normal approx scaling ~ sqrt(h)."""
    sigma = np.nanstd(y_fit_resid, ddof=1) if len(y_fit_resid) > 1 else 0.0
    # For random-walk-ish error growth; simple heuristic:
    scales = np.sqrt(np.arange(1, h + 1, dtype=float))
    return sigma * scales, sigma * scales


def ets_fit_predict(
    y_train: pd.Series, h: int, sp: int, config: Dict
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Fit ETS with given config; return point forecast and residuals from in-sample one-step fit."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = ExponentialSmoothing(
            y_train,
            seasonal_periods=max(2, sp),
            trend=config.get("trend"),
            seasonal=config.get("seasonal"),
            damped_trend=config.get("damped_trend", False),
            initialization_method="estimated",
        )
        res = model.fit(optimized=True, use_brute=False)
        f = res.forecast(h)
        # Residuals from in-sample one-step ahead
        resid = (y_train - res.fittedvalues).to_numpy()
    return f.to_numpy(), resid


def ets_intervals(
    resid: np.ndarray, h: int, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Normal-approx prediction intervals for ETS using residual std ~ sqrt(h)."""
    z = z_score_for(alpha)
    sigma = np.nanstd(resid, ddof=1) if len(resid) > 1 else 0.0
    scales = np.sqrt(np.arange(1, h + 1, dtype=float))
    half_width = z * sigma * scales
    return half_width, half_width


def sarima_select_order(
    y: pd.Series, sp: int
) -> Optional[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]]:
    """
    Lightweight SARIMA order selection on a small grid by AIC.
    Returns ((p,d,q), (P,D,Q,s)) or None on failure.
    """
    # Small grid that works well in practice without pmdarima
    pdq_grid = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
    PDQ_grid = [(0, 1, 1), (1, 1, 0), (1, 1, 1)]
    if sp <= 1:
        PDQ_grid = [(0, 0, 0)]  # no seasonality

    best = None
    best_aic = np.inf

    for p, d, q in pdq_grid:
        for P, D, Q in PDQ_grid:
            seasonal_order = (P, D, Q, max(1, sp))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    mod = SARIMAX(
                        y,
                        order=(p, d, q),
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    res = mod.fit(disp=False)
                aic = res.aic
                if np.isfinite(aic) and aic < best_aic:
                    best_aic = aic
                    best = ((p, d, q), seasonal_order)
            except Exception:
                continue
    return best


def sarima_fit_predict(
    y_train: pd.Series,
    h: int,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False)
        f = res.get_forecast(steps=h)
        # In-sample residuals (one-step)
        resid = res.resid
    return f.predicted_mean.to_numpy(), np.asarray(resid)


def sarima_intervals(
    y_train: pd.Series, h: int, order, seasonal_order, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = SARIMAX(
            y_train,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False)
        forecast_res = res.get_forecast(steps=h)
        ci = forecast_res.conf_int(alpha=alpha)
        lower = ci.iloc[:, 0].to_numpy()
        upper = ci.iloc[:, 1].to_numpy()
        # Half-widths around mean forecast (symmetric approx)
        mean = forecast_res.predicted_mean.to_numpy()
        return (mean - lower), (upper - mean)


# --------------------------- Rolling CV ---------------------------


def rolling_cv(
    y: pd.Series,
    sp: int,
    horizon: int,
    n_splits: Optional[int],
    initial_window: Optional[int],
    model_family: str,
    ets_config: Optional[Dict] = None,
    sarima_orders: Optional[
        Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = None,
) -> float:
    """
    Rolling-origin evaluation: mean sMAPE over splits, each predicting 'horizon'.
    """
    n = len(y)
    if n < max(10, 2 * sp + horizon):
        # Too short—fall back to simple holdout
        n_splits = 1
        initial = max(sp + horizon, int(n * 0.6))
    else:
        initial = initial_window or max(2 * sp, 24)
        if initial + horizon >= n:
            initial = max(int(n * 0.5), sp + horizon)
        if n_splits is None:
            n_splits = min(5, max(1, (n - initial) // horizon))

    scores: List[float] = []

    for i in range(n_splits):
        end = initial + i * horizon
        y_train = y.iloc[:end]
        y_valid = y.iloc[end : end + horizon]
        if len(y_valid) < horizon:
            break

        if model_family == "snaive":
            pred = seasonal_naive_forecast(y_train, horizon, sp)
        elif model_family == "ets":
            pred, _ = ets_fit_predict(y_train, horizon, sp, ets_config or {})
        elif model_family == "sarima":
            if sarima_orders is None:
                # If order not given, skip
                scores.append(np.inf)
                continue
            (order, seasonal_order) = sarima_orders
            pred, _ = sarima_fit_predict(y_train, horizon, order, seasonal_order)
        else:
            raise ValueError(f"Unknown model family: {model_family}")

        scores.append(smape(y_valid.to_numpy(), pred))

    return float(np.mean(scores)) if scores else np.inf


# --------------------------- Station processing ---------------------------


@dataclass
class ModelChoice:
    family: str
    details: Dict
    cv_smape: float


@dataclass
class StationArtifacts:
    station_id: str
    freq: str
    seasonal_periods: int
    chosen_model: ModelChoice
    candidates: Dict[str, Dict]  # raw scores/config per family


def prepare_series(
    df: pd.DataFrame,
    time_col: str,
    value_col: str,
    freq: Optional[str],
    impute: str,
) -> Tuple[pd.Series, str]:
    """
    Parse, sort, set freq, and impute gaps.
    Returns (y, freq_str)
    """
    s = df[[time_col, value_col]].copy()
    s[time_col] = pd.to_datetime(s[time_col], utc=False)
    s = s.sort_values(time_col).set_index(time_col)
    y = s[value_col].astype(float)

    # If freq unspecified, infer
    if not freq or freq == "auto":
        f = infer_freq(y.index)
        if f is None:
            # attempt fallback: use median diff
            diffs = np.diff(y.index.view("i8"))
            if len(diffs):
                step_ns = int(np.median(diffs))
                f = to_offset(pd.to_timedelta(step_ns, unit="ns")).freqstr
        freq_str = f or "D"  # safe fallback
    else:
        freq_str = freq

    # Align to freq and impute
    y = y.asfreq(freq_str)

    if impute == "ffill":
        y = y.ffill()
    elif impute == "bfill":
        y = y.bfill()
    elif impute == "linear":
        y = y.interpolate(method="time", limit_direction="both")
    # else: "none" → leave NaNs (models may fail if NaNs remain)
    y = y.dropna()

    return y, freq_str


def process_station(
    station_id: str,
    df_station: pd.DataFrame,
    args: argparse.Namespace,
) -> StationArtifacts:
    # Prepare series (align, impute)
    y, freq_str = prepare_series(
        df_station,
        time_col=args.time_col,
        value_col=args.value_col,
        freq=args.freq,
        impute=args.impute,
    )

    # Decide seasonal periods
    sp = args.seasonal_periods
    if isinstance(sp, str) and sp.lower() == "auto":
        sp = default_seasonal_periods(freq_str)
    sp = int(max(1, sp))

    # Denoise via STL
    den = stl_denoise(y, sp)
    den_df = pd.DataFrame(
        {
            "timestamp": y.index,
            "y": y.values,
            "trend": den.trend.values,
            "seasonal": den.seasonal.values,
            "resid": den.resid.values,
            "denoised": den.denoised.values,
        }
    )

    # Candidate models
    ets_candidates = [
        {"trend": "add", "seasonal": "add", "damped_trend": True},
        {"trend": "add", "seasonal": "add", "damped_trend": False},
        {"trend": None, "seasonal": "add", "damped_trend": False},
    ]

    # Select SARIMA orders on full series
    sarima_orders = sarima_select_order(y, sp)

    # Evaluate by rolling CV
    candidates_summary: Dict[str, Dict] = {}

    # Seasonal Naive
    cv_snaive = rolling_cv(
        y=y,
        sp=sp,
        horizon=args.horizon,
        n_splits=args.n_splits,
        initial_window=args.initial_window,
        model_family="snaive",
    )
    candidates_summary["snaive"] = {"cv_smape": float(cv_snaive)}

    # ETS (pick best config via CV)
    ets_scores = []
    for cfg in ets_candidates:
        score = rolling_cv(
            y=y,
            sp=sp,
            horizon=args.horizon,
            n_splits=args.n_splits,
            initial_window=args.initial_window,
            model_family="ets",
            ets_config=cfg,
        )
        ets_scores.append((score, cfg))
    best_ets_score, best_ets_cfg = min(ets_scores, key=lambda t: t[0])
    candidates_summary["ets"] = {
        "cv_smape": float(best_ets_score),
        "config": best_ets_cfg,
    }

    # SARIMA
    if sarima_orders is not None:
        cv_sarima = rolling_cv(
            y=y,
            sp=sp,
            horizon=args.horizon,
            n_splits=args.n_splits,
            initial_window=args.initial_window,
            model_family="sarima",
            sarima_orders=sarima_orders,
        )
        candidates_summary["sarima"] = {
            "cv_smape": float(cv_sarima),
            "order": sarima_orders[0],
            "seasonal_order": sarima_orders[1],
        }
    else:
        candidates_summary["sarima"] = {
            "cv_smape": float(np.inf),
            "order": None,
            "seasonal_order": None,
        }

    # Choose best family
    best_family = min(candidates_summary.items(), key=lambda kv: kv[1]["cv_smape"])[0]

    # Fit chosen on full data & forecast + intervals
    horizon = args.horizon
    fut_index = next_periods(y.index, horizon, freq_str)

    if best_family == "snaive":
        yhat = seasonal_naive_forecast(y, horizon, sp)
        # Residuals for (approx) intervals
        # using one-step seasonal naive errors on training
        if len(y) > sp:
            snaive_in_sample = seasonal_naive_forecast(y.iloc[:-sp], sp, sp)
            resid = (
                y.iloc[-sp:].to_numpy() - snaive_in_sample
            )  # small sample; fine as heuristic
        else:
            resid = np.array([0.0])
        lo80_hw, hi80_hw = snaive_intervals(resid, horizon, sp)
        lo95_hw, hi95_hw = snaive_intervals(resid, horizon, sp)
        # symmetric around mean
        lo80 = yhat - lo80_hw
        hi80 = yhat + hi80_hw
        lo95 = yhat - lo95_hw * (z_score_for(0.05) / z_score_for(0.20))
        hi95 = yhat + hi95_hw * (z_score_for(0.05) / z_score_for(0.20))
        details = {}
    elif best_family == "ets":
        yhat, resid = ets_fit_predict(y, horizon, sp, best_ets_cfg)
        lo80_hw, hi80_hw = ets_intervals(resid, horizon, alpha=0.20)
        lo95_hw, hi95_hw = ets_intervals(resid, horizon, alpha=0.05)
        lo80 = yhat - lo80_hw
        hi80 = yhat + hi80_hw
        lo95 = yhat - lo95_hw
        hi95 = yhat + hi95_hw
        details = {"config": best_ets_cfg}
    else:
        # SARIMA with model-based intervals
        (order, seasonal_order) = sarima_orders  # type: ignore
        yhat, resid = sarima_fit_predict(y, horizon, order, seasonal_order)
        # Use model's conf_int at 80% and 95%
        # We recompute intervals by refitting once per alpha to get exact CI (cheap on small models)
        lo80_hw, hi80_hw = sarima_intervals(
            y, horizon, order, seasonal_order, alpha=0.20
        )
        lo95_hw, hi95_hw = sarima_intervals(
            y, horizon, order, seasonal_order, alpha=0.05
        )
        lo80 = yhat - lo80_hw
        hi80 = yhat + hi80_hw
        lo95 = yhat - lo95_hw
        hi95 = yhat + hi95_hw
        details = {"order": order, "seasonal_order": seasonal_order}

    forecast_df = pd.DataFrame(
        {
            "timestamp": fut_index,
            "y_pred": yhat,
            "lo_80": lo80,
            "hi_80": hi80,
            "lo_95": lo95,
            "hi_95": hi95,
        }
    )

    # Write artifacts
    outdir = os.path.join(args.outdir, str(station_id))
    ensure_dir(outdir)
    den_df.to_csv(os.path.join(outdir, "denoised.csv"), index=False)
    forecast_df.to_csv(os.path.join(outdir, "forecast.csv"), index=False)

    choice = ModelChoice(
        family=best_family,
        details=details,
        cv_smape=float(candidates_summary[best_family]["cv_smape"]),
    )
    report = {
        "station_id": station_id,
        "freq": freq_str,
        "seasonal_periods": sp,
        "chosen_model": {
            "family": choice.family,
            "details": choice.details,
            "cv_smape": choice.cv_smape,
        },
        "candidates": candidates_summary,
        "n_obs": int(len(y)),
        "horizon": int(horizon),
    }
    with open(os.path.join(outdir, "model_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return StationArtifacts(
        station_id=str(station_id),
        freq=freq_str,
        seasonal_periods=sp,
        chosen_model=choice,
        candidates=candidates_summary,
    )


# --------------------------- CLI ---------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fit best model per station, export denoised series and forecast intervals."
    )
    p.add_argument(
        "--input",
        required=True,
        help="Path to CSV containing time series for one or many stations.",
    )
    p.add_argument("--time-col", required=True, help="Timestamp column name.")
    p.add_argument("--value-col", required=True, help="Value column name.")
    p.add_argument(
        "--station-col",
        required=False,
        default=None,
        help="Station ID column; if omitted, all rows treated as one station.",
    )
    p.add_argument(
        "--horizon", type=int, required=True, help="Forecast horizon (number of steps)."
    )
    p.add_argument(
        "--outdir", required=True, help="Output directory for per-station artifacts."
    )
    p.add_argument(
        "--freq",
        default="auto",
        help="Pandas freq string (e.g., H, D, W, M) or 'auto'.",
    )
    p.add_argument(
        "--seasonal-periods", default="auto", help="Integer seasonal period or 'auto'."
    )
    p.add_argument(
        "--impute",
        choices=["none", "ffill", "bfill", "linear"],
        default="linear",
        help="Gap imputation method after aligning to frequency.",
    )
    p.add_argument(
        "--n-splits", type=int, default=None, help="Rolling CV splits (default: auto)."
    )
    p.add_argument(
        "--initial-window",
        type=int,
        default=None,
        help="Initial training window for CV (default: auto).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.outdir)

    df = pd.read_csv(args.input)
    if args.station_col and args.station_col in df.columns:
        stations = sorted(df[args.station_col].dropna().unique())
        groups = df.groupby(args.station_col)
    else:
        stations = ["__ALL__"]
        groups = [(stations[0], df)]

    run_summary: Dict[str, Dict] = {}

    for sid, g in groups if isinstance(groups, list) else groups:
        try:
            artifacts = process_station(str(sid), g, args)
            run_summary[str(sid)] = {
                "freq": artifacts.freq,
                "seasonal_periods": artifacts.seasonal_periods,
                "chosen_model": {
                    "family": artifacts.chosen_model.family,
                    "details": artifacts.chosen_model.details,
                    "cv_smape": artifacts.chosen_model.cv_smape,
                },
                "candidates": artifacts.candidates,
            }
            print(
                f"[OK] {sid}: {artifacts.chosen_model.family} (CV sMAPE={artifacts.chosen_model.cv_smape:.2f})"
            )
        except Exception as e:
            run_summary[str(sid)] = {"error": str(e)}
            print(f"[FAIL] {sid}: {e}")

    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump(run_summary, f, indent=2)


if __name__ == "__main__":
    # Avoid noisy solver warnings from statsmodels
    warnings.filterwarnings("ignore")
    main()
