# -*- coding: utf-8 -*-
"""
Weather GRU + Physics (Gray-box) with GRU-D missing-data handling,
station & cluster embeddings, hierarchical clustering dendrogram,
denoising options, and leakage-safe backtesting (purge + embargo).

This is a single-file implementation. It includes:
- load_df() (as provided by the user) to read from SQLite
- Feature building with physics-informed variables
- Optional denoising (None/FFT/Wavelet/SARIMAX-residual)
- GRU-D-style missing data handling (masks + time-since-last-seen decays)
- Station embeddings and optional cluster embeddings
- Hierarchical clustering of stations with dendrogram plot
- Purged + embargoed cross-validation for leakage-safe backtests
- Quantile heads (P10/P50/P90) to produce confidence bands
- Automatic model selection (heuristic or GradientBoosting classifier)

Outputs:
- models/model_<tag>_foldK.pt      (trained weights)
- outputs/val_predictions_<tag>.csv
- outputs/metrics_<tag>.txt
- outputs/cv_metrics_<tag>.csv
- outputs/plot_<tag>_<station>.png
- outputs/dendrogram_<tag>.png
- outputs/station_clusters_<tag>.csv

Hardware: Optimized for multi-core CPU; uses CUDA if available.
"""

import os, math, gc, warnings, sqlite3
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional libs (used if present)
try:
    from scipy.signal import butter, filtfilt

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    import pywt

    HAVE_PYWT = True
except Exception:
    HAVE_PYWT = False

try:
    import statsmodels.api as sm

    HAVE_STATSMODELS = True
except Exception:
    HAVE_STATSMODELS = False

try:
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import AgglomerativeClustering

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

try:
    import scipy.cluster.hierarchy as sch

    HAVE_SCIPY_CLUSTER = True
except Exception:
    HAVE_SCIPY_CLUSTER = False

# ================
# Environment
# ================
torch.set_num_threads(min(16, os.cpu_count() or 8))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 0) User-provided loader (kept intact) + helpers for metadata
# ============================================================

from pathlib import Path

# ======================
# CONFIG
# ======================
DATA_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/data")
DATABASE_URI = str(DATA_FOLDER / "database.db")


# ======================
# SQL QUERY
# ======================
def load_df() -> pd.DataFrame:
    """
    Original user-provided loader. Returns a DataFrame with 'datetime' and the
    Portuguese-named columns from the 'medicoes' table for BAURU station only.
    """
    STMT = """SELECT * FROM medicoes a;"""

    with sqlite3.connect(DATABASE_URI) as conn:
        df = pd.read_sql_query(STMT, conn)

    # ======================
    # DATETIME CLEANING
    # ======================
    def standard_format(datetime_hour: str, /) -> str:
        datetime_hour = datetime_hour.replace("-", "/")
        datetime_hour = str(datetime_hour).replace("UTC", "").strip()
        if ":" in datetime_hour:
            return datetime_hour.replace(":", "")
        return datetime_hour

    df["datetime"] = pd.to_datetime(
        (df["data"] + df["hora"]).apply(standard_format), format="%Y/%m/%d%H%M"
    )

    # Select useful columns
    df = df[
        [
            "datetime",
            "id_fk",
            "precipitacao_total_horario",
            "pressao_atmosferica_ao_nivel_da_estacao_horaria",
            "pressao_atmosferica_max_na_hora_ant",
            "pressao_atmosferica_min_na_hora_ant",
            "radiacao_global",
            "temperatura_do_ar_bulbo_seco_horaria",
            "temperatura_do_ponto_de_orvalho",
            "temperatura_maxima_na_hora_ant",
            "temperatura_minima_na_hora_ant",
            "temperatura_orvalho_max_na_hora_ant",
            "temperatura_orvalho_min_na_hora_ant",
            "umidade_relativa_max_na_hora_ant",
            "umidade_relativa_min_na_hora_ant",
            "umidade_relativa_do_ar_horaria",
            "vento_direcao_horaria",
            "vento_rajada_maxima",
            "vento_velocidade_horaria",
        ]
    ]

    df.sort_values("datetime", inplace=True)

    return df


def augment_with_station_meta(
    df: pd.DataFrame, db_uri: str = DATABASE_URI
) -> pd.DataFrame:
    """
    Augment the loaded measurements with station metadata (lat, lon, altitude, names).
    Expects 'id_fk' in df. Adds: 'station', 'lat', 'lon', 'altitude'.
    """
    if "id_fk" not in df.columns:
        df = df.copy()
        df["id_fk"] = -1  # single-station fallback

    with sqlite3.connect(db_uri) as conn:
        ids = tuple(sorted(set(df["id_fk"].dropna().astype(int).tolist())))
        if len(ids) == 0:
            # Attempt to pull BAURU id anyway
            st_meta = pd.read_sql_query(
                'SELECT id, nome_do_arquivo, regiao, uf, estacao, codigo, latitude, longitude, altitude FROM estacoes WHERE nome_do_arquivo LIKE "%BAURU%";',
                conn,
            )
        else:
            qmarks = ",".join(["?"] * len(ids))
            st_meta = pd.read_sql_query(
                f"SELECT id, nome_do_arquivo, regiao, uf, estacao, codigo, latitude, longitude, altitude FROM estacoes WHERE id IN ({qmarks});",
                conn,
                params=list(ids),
            )

    if st_meta.empty:
        # Graceful fallback
        df = df.copy()
        df["station"] = "UNKNOWN"
        df["lat"] = 0.0
        df["lon"] = 0.0
        df["altitude"] = 0.0
        return df

    # Build display name preference
    st_meta["station_name"] = st_meta["estacao"].fillna("").astype(str)
    empty = st_meta["station_name"].eq("")
    st_meta.loc[empty, "station_name"] = st_meta.loc[empty, "nome_do_arquivo"].fillna(
        ""
    )
    empty = st_meta["station_name"].eq("")
    st_meta.loc[empty, "station_name"] = st_meta.loc[empty, "codigo"].fillna("")
    st_meta.loc[st_meta["station_name"].eq(""), "station_name"] = "station_" + st_meta[
        "id"
    ].astype(str)

    merged = df.merge(
        st_meta.rename(columns={"id": "id_fk", "latitude": "lat", "longitude": "lon"})[
            ["id_fk", "station_name", "lat", "lon", "altitude"]
        ],
        on="id_fk",
        how="left",
    )
    merged["station"] = merged["station_name"].fillna(
        "station_" + merged["id_fk"].astype(str)
    )
    merged.drop(columns=["station_name"], inplace=True)
    return merged


# ============================================================
# 1) Column normalization & unit conversions (PT -> canonical)
# ============================================================
SIGMA = 5.670374419e-8  # W m^-2 K^-4
CP = 1004.0  # J kg^-1 K^-1
RD = 287.0  # J kg^-1 K^-1
EPS = 0.622

# Portuguese -> canonical feature names
RENAME_PT = {
    "precipitacao_total_horario": "precip_mm",
    "pressao_atmosferica_ao_nivel_da_estacao_horaria": "pressure_mb",
    # Previous-hour extrema exist, but we compute previous via shift per station
    "radiacao_global": "global_rad_kj_m2",
    "temperatura_do_ar_bulbo_seco_horaria": "temp_c",
    "temperatura_do_ponto_de_orvalho": "dewpoint_c",
    "umidade_relativa_do_ar_horaria": "rh_pct",
    "vento_direcao_horaria": "wind_dir_deg",
    "vento_rajada_maxima": "wind_gust_ms",
    "vento_velocidade_horaria": "wind_speed_ms",
}

# Scaling correction for radiation if needed (keep at 1.0 for kJ/m² input)
RAD_SCALE = 1.0


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Map Portuguese column names to canonical names, ensure types, and create
    previous-hour features by station (shifted).
    Requires 'datetime', 'station', 'lat', 'lon' to be present (use augment_with_station_meta first).
    """
    df = df.rename(columns=RENAME_PT).copy()
    if "datetime" not in df.columns:
        raise ValueError("DataFrame must have 'datetime' column.")
    if (
        "station" not in df.columns
        or "lat" not in df.columns
        or "lon" not in df.columns
    ):
        raise ValueError(
            "Please call augment_with_station_meta() before standardize_columns()."
        )

    # Enforce dtypes
    to_num = [
        "lat",
        "lon",
        "precip_mm",
        "pressure_mb",
        "global_rad_kj_m2",
        "temp_c",
        "dewpoint_c",
        "rh_pct",
        "wind_dir_deg",
        "wind_gust_ms",
        "wind_speed_ms",
    ]
    for col in to_num:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort properly
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.sort_values(["station", "datetime"]).reset_index(drop=True)

    # Compute previous-hour values via shift (per station) to avoid using MAX/MIN previous-hour variants
    def add_prev(g: pd.DataFrame) -> pd.DataFrame:
        for src, new in [
            ("pressure_mb", "pressure_prev_mb"),
            ("temp_c", "temp_prev_c"),
            ("dewpoint_c", "dewpoint_prev_c"),
            ("rh_pct", "rh_prev_pct"),
        ]:
            if src in g.columns:
                g[new] = g[src].shift(1)
        return g

    df = df.groupby("station", group_keys=False).apply(add_prev)

    return df


# ============================================================
# 2) Physics feature builders (robust to missing)
# ============================================================
def es_hPa_from_Tc(Tc: np.ndarray) -> np.ndarray:
    # Alduchov & Eskridge (1996) Magnus form
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))


def Lv_J_kg(Tc: np.ndarray) -> np.ndarray:
    return 2.501e6 - 2361.0 * Tc


def safe_psychrometrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute psychrometrics even when dewpoint or RH is missing."""
    Tc = df["temp_c"].values
    Td = (
        df["dewpoint_c"].values
        if "dewpoint_c" in df.columns
        else np.full(len(df), np.nan)
    )
    RH = df["rh_pct"].values if "rh_pct" in df.columns else np.full(len(df), np.nan)
    p_pa = (df["pressure_mb"].values) * 100.0

    es_T = es_hPa_from_Tc(Tc) * 100.0  # Pa
    # If dewpoint present, use it; else fall back to RH% * es(T)
    ea_from_Td = es_hPa_from_Tc(Td) * 100.0
    ea_from_RH = (np.nan_to_num(RH, nan=np.nan) / 100.0) * es_T
    ea = np.where(np.isfinite(ea_from_Td), ea_from_Td, ea_from_RH)
    # Final fallback: small fraction of es_T to avoid NaNs
    ea = np.where(np.isfinite(ea), ea, 0.5 * es_T)

    vpd = np.maximum(es_T - ea, 0.0)  # Pa
    w = EPS * ea / np.maximum(p_pa - ea, 1.0)
    q = w / (1.0 + w)
    Tv = (Tc + 273.15) * (1.0 + 0.61 * q)
    rho = p_pa / (RD * Tv)
    Lv = Lv_J_kg(Tc)
    gamma = CP * p_pa / (EPS * Lv)
    out = pd.DataFrame(
        {
            "p_pa": p_pa,
            "es_pa": es_T,
            "ea_pa": ea,
            "vpd_pa": vpd,
            "q_kgkg": q,
            "rho_air": rho,
            "Tv_K": Tv,
            "Lv_Jkg": Lv,
            "gamma": gamma,
        },
        index=df.index,
    )
    return out


def fao56_extraterrestrial_radiation_hourly(ts, lat_deg, lon_deg):
    """FAO-56 hourly approximation of extraterrestrial radiation (Ra) and cos(zenith)."""
    lat = np.radians(lat_deg)
    # Day of year
    n = ts.timetuple().tm_yday
    dr = 1 + 0.033 * np.cos(2 * np.pi * n / 365.0)
    delta = 0.409 * np.sin(2 * np.pi * n / 365.0 - 1.39)

    # Approximate local solar time using longitude (timezone ≈ round(lon/15))
    tz = np.round(lon_deg / 15.0)
    lst = ts + pd.to_timedelta((lon_deg / 15.0 - tz), unit="h")
    omega = (lst.hour + lst.minute / 60.0 + lst.second / 3600.0 - 12.0) * (
        np.pi / 12.0
    )  # center hour angle
    omega1 = omega - np.pi / 24.0
    omega2 = omega + np.pi / 24.0

    Gsc = 0.0820  # MJ m^-2 min^-1
    Ra_MJ = (
        (12 * 60 / np.pi)
        * Gsc
        * dr
        * (
            (np.cos(lat) * np.cos(delta)) * (np.sin(omega2) - np.sin(omega1))
            + (np.pi * (omega2 - omega1) / 180.0) * np.sin(lat) * np.sin(delta)
        )
    )
    Ra_Wm2 = max(Ra_MJ * 1e6 / 3600.0, 0.0)
    cosz = max(
        np.sin(lat) * np.sin(delta) + np.cos(lat) * np.cos(delta) * np.cos(omega), 0.0
    )
    return Ra_Wm2, cosz


def radiation_features(df: pd.DataFrame) -> pd.DataFrame:
    # If radiacao_global is kJ/m² per hour, convert to W/m²
    S_in = (df["global_rad_kj_m2"].values * RAD_SCALE) / 3.6  # kJ/h/m2 -> W/m2
    Ra = np.empty(len(df))
    cosz = np.empty(len(df))
    for i, (ts, lat, lon) in enumerate(zip(df["datetime"], df["lat"], df["lon"])):
        ra_i, cosz_i = fao56_extraterrestrial_radiation_hourly(ts, lat, lon)
        Ra[i] = ra_i
        cosz[i] = cosz_i
    Kt = np.divide(S_in, np.maximum(Ra, 1.0), out=np.zeros_like(S_in), where=Ra > 1.0)
    out = pd.DataFrame(
        {"S_in_Wm2": S_in, "Ra_Wm2": Ra, "cos_zenith": cosz, "clearness_index": Kt},
        index=df.index,
    )
    return out


def longwave_downward_brutsaert(df, psych, rad):
    Tk = df["temp_c"].values + 273.15
    e_kPa = psych["ea_pa"].values / 1000.0
    # Brutsaert clear-sky emissivity ~ 1.24*(e/T)^1/7
    eps_clear = 1.24 * np.power(
        np.maximum(e_kPa / np.maximum(Tk, 1.0), 1e-8), 1.0 / 7.0
    )
    eps_clear = np.clip(eps_clear, 0.5, 1.0)
    cloud = np.clip(1.0 - np.clip(rad["clearness_index"].values, 0.0, 1.0), 0.0, 1.0)
    eps_sky = eps_clear * (1.0 + 0.22 * cloud**2)
    Ld = eps_sky * SIGMA * Tk**4
    return pd.DataFrame({"eps_sky": eps_sky, "L_down_Wm2": Ld}, index=df.index)


def energy_balance_proxies(df, psych, rad, eps_s=0.98, alpha0=0.23):
    Rns = (1.0 - alpha0) * rad["S_in_Wm2"].values
    Rnl = rad["L_down_Wm2"].values - eps_s * SIGMA * (df["temp_c"].values + 273.15) ** 4
    Rn = Rns + Rnl
    U = np.nan_to_num(df["wind_speed_ms"].values, nan=0.0)
    VPD = psych["vpd_pa"].values
    out = pd.DataFrame(
        {"Rns_Wm2": Rns, "Rnl_Wm2": Rnl, "Rn_Wm2": Rn, "U_ms": U, "VPD_pa": VPD},
        index=df.index,
    )
    return out


# ============================================================
# 3) Denoising / residualization
# ============================================================
def fft_lowpass(x, cutoff_hours=6, dt_hours=1.0):
    x = pd.Series(x).astype(float)
    n = len(x)
    if n < 8:
        return x.values
    X = np.fft.rfft(x.fillna(method="ffill").fillna(method="bfill").values)
    freqs = np.fft.rfftfreq(n, d=dt_hours)
    X[freqs > 1.0 / cutoff_hours] = 0.0
    y = np.fft.irfft(X, n=n)
    return y


def butter_lowpass(x, cutoff_hours=6, dt_hours=1.0, order=4):
    if not HAVE_SCIPY:
        return x
    fs = 1.0 / dt_hours
    b, a = butter(order, (1.0 / cutoff_hours) / (0.5 * fs), btype="low")
    return filtfilt(
        b, a, pd.Series(x).fillna(method="ffill").fillna(method="bfill").values
    )


def wavelet_denoise(x, wave="db4", level=2):
    if not HAVE_PYWT:
        return x
    coeffs = pywt.wavedec(
        pd.Series(x).fillna(method="ffill").fillna(method="bfill").values,
        wave,
        mode="periodization",
    )
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    uth = sigma * np.sqrt(2 * np.log(len(x)))
    coeffs[1:] = [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
    return pywt.waverec(coeffs, wave, mode="periodization")[: len(x)]


def sarimax_residuals(y, order=(2, 0, 2), seasonal_order=(1, 0, 1, 24 * 7)):
    if not HAVE_STATSMODELS:
        return y - pd.Series(y).rolling(24, min_periods=1).mean().values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = sm.tsa.statespace.SARIMAX(
            y,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = mod.fit(disp=False, maxiter=200)
    fitted = res.fittedvalues
    resid = y - fitted
    return (resid + pd.Series(y).rolling(24, min_periods=1).mean()).values


# ============================================================
# 4) Full feature assembly (per-row, no leakage)
# ============================================================
def build_features(raw_df: pd.DataFrame, denoise: Optional[str] = None) -> pd.DataFrame:
    """
    raw_df: DataFrame produced by load_df() and augmented by augment_with_station_meta().
    """
    df = standardize_columns(raw_df)
    # Psychrometrics & radiation (robust)
    psy = safe_psychrometrics(df)
    rad_base = radiation_features(df)
    lw = longwave_downward_brutsaert(df, psy, rad_base)
    eb = energy_balance_proxies(df, psy, pd.concat([rad_base, lw], axis=1))

    # Time encodings
    ts = df["datetime"]
    hod = (ts.dt.hour + ts.dt.minute / 60.0).values
    hod_sin = np.sin(2 * np.pi * hod / 24.0)
    hod_cos = np.cos(2 * np.pi * hod / 24.0)
    doy = ts.dt.dayofyear.values.astype(float)
    doy_sin = np.sin(2 * np.pi * (doy / 365.25))
    doy_cos = np.cos(2 * np.pi * (doy / 365.25))

    # Wind direction sin/cos
    wd_rad = np.radians(np.nan_to_num(df["wind_dir_deg"].values, nan=0.0))
    wind_sin, wind_cos = np.sin(wd_rad), np.cos(wd_rad)

    features = pd.concat(
        [
            df[
                [
                    "station",
                    "datetime",
                    "lat",
                    "lon",
                    "temp_c",
                    "dewpoint_c",
                    "rh_pct",
                    "precip_mm",
                    "pressure_mb",
                    "global_rad_kj_m2",
                    "wind_speed_ms",
                    "wind_gust_ms",
                ]
            ],
            psy,
            rad_base,
            lw,
            eb,
        ],
        axis=1,
    )

    # Stationwise rolling ranges (past-only)
    def add_rolling_ranges(g):
        g = g.sort_values("datetime")
        for col in ["temp_c", "pressure_mb", "S_in_Wm2", "VPD_pa", "Rn_Wm2"]:
            if col in g.columns:
                rmax = g[col].rolling(24, min_periods=6).max()
                rmin = g[col].rolling(24, min_periods=6).min()
                g[f"{col}_rng24"] = (rmax - rmin).values
        return g

    features = features.groupby("station", group_keys=False).apply(add_rolling_ranges)

    # Denoise target (optional)
    base_y = features["temp_c"].values
    if denoise == "fft":
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(
            lambda s: fft_lowpass(s.values, cutoff_hours=6)
        )
    elif denoise == "butter" and HAVE_SCIPY:
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(
            lambda s: butter_lowpass(s.values, cutoff_hours=6)
        )
    elif denoise == "wavelet" and HAVE_PYWT:
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(
            lambda s: wavelet_denoise(s.values)
        )
    elif denoise == "sarimax":
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(
            lambda s: sarimax_residuals(s.values)
        )
    else:
        y_smooth = base_y
    features["temp_c_smooth"] = y_smooth

    # Fourier time features
    features["hod_sin"], features["hod_cos"] = hod_sin, hod_cos
    features["doy_sin"], features["doy_cos"] = doy_sin, doy_cos
    features["wind_sin"], features["wind_cos"] = wind_sin, wind_cos

    # Physics nominal tendency (first guess; learned gate will rescale)
    Ce_nom = 1.5e6
    kH_nom, kLE_nom, kG_nom = 5.0, 5e-7, 0.05
    T = features["temp_c_smooth"].values
    U = np.nan_to_num(features["wind_speed_ms"].values, nan=0.0)
    Rns = features["Rns_Wm2"].values
    Rnl = features["Rnl_Wm2"].values
    VPD = features["VPD_pa"].values
    T_ref = (
        features.groupby("station")["temp_c_smooth"]
        .transform(lambda s: s.rolling(24, min_periods=6).mean())
        .fillna(method="bfill")
        .values
    )
    H = kH_nom * U * (T - T_ref)
    LE = kLE_nom * U * VPD
    G = kG_nom * Rns
    dTdt_phys = (Rns + Rnl - H - LE - G) / Ce_nom
    features["dTdt_phys"] = dTdt_phys

    return features


# ============================================================
# 5) Missing-data aware sequence dataset (GRU-D style)
# ============================================================
@dataclass
class SeqConfig:
    input_cols: List[str]
    target_col: str = "temp_c"
    seq_len: int = 48
    horizon: int = 1
    stride: int = 1


class SeqDatasetGRUD(Dataset):
    """
    Returns:
      x_seq, m_seq, d_seq   : (L, D) inputs, masks, deltas
      dTdt_phys_last, T_last: scalars
      y                     : target (may be NaN)
      sid, cid              : station id (long), cluster id (long; 0 if not clustered)
      meta                  : dict with indices
    """

    def __init__(
        self,
        feat_df: pd.DataFrame,
        cfg: SeqConfig,
        stations: Optional[List[str]] = None,
    ):
        self.df = feat_df.copy()
        if stations is not None:
            self.df = self.df[self.df["station"].isin(stations)].copy()
        self.df = self.df.sort_values(["station", "datetime"]).reset_index(drop=True)
        self.cfg = cfg
        self.D = len(cfg.input_cols)

        # Station & cluster codes
        cats = self.df["station"].astype("category")
        self.station_code_all = cats.cat.codes.values.astype(np.int64)
        self.stations_ = list(cats.cat.categories)
        self.num_stations = len(self.stations_)
        if "cluster_id" in self.df.columns:
            cc = self.df["cluster_id"].fillna(0).astype(int).astype("category")
            self.cluster_code_all = cc.cat.codes.values.astype(np.int64)
            self.clusters_ = list(cc.cat.categories)
            self.num_clusters = len(self.clusters_)
        else:
            self.cluster_code_all = np.zeros(len(self.df), dtype=np.int64)
            self.clusters_ = [0]
            self.num_clusters = 1

        # Input arrays, masks, deltas
        self.X = self.df[self.cfg.input_cols].values.astype(np.float32)
        self.M = (~pd.isna(self.df[self.cfg.input_cols])).values.astype(np.float32)

        self.DELTA = np.zeros_like(self.X, dtype=np.float32)
        for st, g in self.df.groupby("station"):
            idx = g.index.to_numpy()
            t = (
                self.df.loc[idx, "datetime"]
                .values.astype("datetime64[h]")
                .astype(np.int64)
            )
            dt_hours = np.diff(t, prepend=t[0]).astype(np.float32)
            dt_hours[0] = 1.0
            d_running = np.zeros((len(idx), self.D), dtype=np.float32)
            last_obs = np.full(self.D, -np.inf, dtype=np.float32)
            for k, ridx in enumerate(idx):
                obs = self.M[ridx] == 1.0
                last_obs[obs] = 0.0
                last_obs[~obs] += dt_hours[k]
                d_running[k, :] = np.maximum(last_obs, 0.0)
            self.DELTA[idx, :] = d_running

        self.X = np.nan_to_num(self.X, nan=0.0)

        # Build windows
        self.rows = []
        for st, g in self.df.groupby("station"):
            n = len(g)
            for i in range(
                0, n - self.cfg.seq_len - self.cfg.horizon + 1, self.cfg.stride
            ):
                start, end = i, i + self.cfg.seq_len
                y_t = i + self.cfg.seq_len + self.cfg.horizon - 1
                self.rows.append(
                    (g.index[start:end].to_numpy(), g.index[end - 1], g.index[y_t])
                )
        self.rows = np.array(self.rows, dtype=object)

        self.sample_meta = pd.DataFrame(
            {"x_end_idx": [r[1] for r in self.rows], "y_idx": [r[2] for r in self.rows]}
        )
        self.sample_meta["datetime_x_end"] = self.df.loc[
            self.sample_meta["x_end_idx"], "datetime"
        ].values
        self.sample_meta["datetime_y"] = self.df.loc[
            self.sample_meta["y_idx"], "datetime"
        ].values

        # Mean vector for imputer
        means = np.nanmean(
            np.where(self.M == 1.0, self.df[self.cfg.input_cols].values, np.nan), axis=0
        )
        self.input_means = np.nan_to_num(means, nan=0.0).astype(np.float32)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        idx_seq, idx_xend, idx_y = self.rows[i]
        X = self.X[idx_seq, :]
        M = self.M[idx_seq, :]
        Dlt = self.DELTA[idx_seq, :]

        dTdt_phys_last = self.df.loc[idx_xend, "dTdt_phys"]
        T_last = self.df.loc[idx_xend, "temp_c"]
        y = self.df.loc[idx_y, self.cfg.target_col]

        sid = self.station_code_all[idx_xend]
        cid = self.cluster_code_all[idx_xend]

        meta = {"x_end_idx": int(idx_xend), "y_idx": int(idx_y)}
        return (
            torch.from_numpy(X).float(),
            torch.from_numpy(M).float(),
            torch.from_numpy(Dlt).float(),
            torch.tensor(np.float32(dTdt_phys_last)),
            torch.tensor(np.float32(T_last if pd.notna(T_last) else 0.0)),
            torch.tensor(np.float32(y if pd.notna(y) else np.nan)),
            torch.tensor(sid, dtype=torch.long),
            torch.tensor(cid, dtype=torch.long),
            meta,
        )


# ============================================================
# 6) Purged & embargoed time split
# ============================================================
def purged_time_splits(
    meta_times: pd.DataFrame, n_splits=5, horizon_hours=1, embargo_hours=24
):
    y_times = pd.to_datetime(meta_times["datetime_y"]).values
    t_sorted = np.sort(y_times)
    fold_edges = np.linspace(0, len(t_sorted), n_splits + 1, dtype=int)
    splits = []
    for k in range(n_splits):
        val_lo_t = t_sorted[fold_edges[k]]
        val_hi_t = t_sorted[fold_edges[k + 1] - 1]
        val_mask = (y_times >= val_lo_t) & (y_times <= val_hi_t)

        horizon = np.timedelta64(horizon_hours, "h")
        embargo = np.timedelta64(embargo_hours, "h")
        purge_lo = val_lo_t - horizon
        purge_hi = val_hi_t + horizon
        embargo_lo = val_hi_t
        embargo_hi = val_hi_t + embargo

        train_mask = ~val_mask
        x_times = pd.to_datetime(meta_times["datetime_x_end"]).values
        overlap = ((y_times >= purge_lo) & (y_times <= purge_hi)) | (
            (x_times >= purge_lo) & (x_times <= purge_hi)
        )
        train_mask &= ~overlap
        train_mask &= ~((y_times > embargo_lo) & (y_times <= embargo_hi))
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits


# ============================================================
# 7) GRU-D imputer + GRU + Station/Cluster embeddings + Physics + Quantile heads
# ============================================================
class GRUDImputer(nn.Module):
    """GRU-D style imputation (Che et al., 2018) with learnable decays."""

    def __init__(self, input_dim):
        super().__init__()
        self.input_dim = input_dim
        self.w_x = nn.Parameter(torch.ones(input_dim))
        self.b_x = nn.Parameter(torch.zeros(input_dim))
        self.w_h = nn.Parameter(torch.ones(input_dim))
        self.b_h = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("x_mean", torch.zeros(input_dim))

    def set_x_mean(self, x_mean: np.ndarray):
        with torch.no_grad():
            self.x_mean.copy_(torch.from_numpy(x_mean).float())

    def forward(self, x, m, d, h_prev=None):
        B, L, D = x.shape
        x_imp = torch.zeros_like(x)
        h_decay_seq = torch.zeros(B, L, 1, device=x.device)
        x_last = self.x_mean.unsqueeze(0).unsqueeze(0).expand(B, 1, D)  # (B,1,D)
        for t in range(L):
            dt = d[:, t, :]  # (B,D)
            gamma_x = torch.exp(-torch.relu(self.w_x * dt + self.b_x))  # (B,D)
            x_hat = m[:, t, :] * x[:, t, :] + (1 - m[:, t, :]) * (
                gamma_x * (x_last[:, -1, :]) + (1 - gamma_x) * self.x_mean
            )
            x_imp[:, t, :] = x_hat

            gamma_h = torch.exp(-torch.relu((self.w_h * dt + self.b_h))).mean(
                dim=-1, keepdim=True
            )  # (B,1)
            h_decay_seq[:, t, :] = gamma_h
            x_last = torch.cat([x_last[:, -1:, :], x_hat.unsqueeze(1)], dim=1)
        return x_imp, h_decay_seq


class PhysGRUDQuant(nn.Module):
    """
    GRU-D imputer + GRU encoder + station/cluster embeddings + physics tendency + quantile heads.
    """

    def __init__(
        self,
        input_dim,
        num_stations,
        station_emb_dim=16,
        num_clusters=1,
        cluster_emb_dim=0,
        hidden=128,
        layers=2,
        dropout=0.1,
    ):
        super().__init__()
        self.imp = GRUDImputer(input_dim)
        aug_in = (
            input_dim
            + station_emb_dim
            + (cluster_emb_dim if (num_clusters > 1 and cluster_emb_dim > 0) else 0)
        )

        self.st_emb = nn.Embedding(num_stations, station_emb_dim)
        self.has_cluster = num_clusters > 1 and cluster_emb_dim > 0
        if self.has_cluster:
            self.cl_emb = nn.Embedding(num_clusters, cluster_emb_dim)

        self.gru = nn.GRU(
            aug_in, hidden, num_layers=layers, batch_first=True, dropout=dropout
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 3)
        )  # q10,q50,q90 deltas
        self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())

    def set_x_mean(self, mean_vec: np.ndarray):
        self.imp.set_x_mean(mean_vec)

    def forward(self, x, m, d, dTdt_phys_last, T_last, sid, cid=None):
        x_imp, _ = self.imp(x, m, d)  # (B,L,D)
        B, L, _ = x_imp.shape

        # Embeddings
        e = self.st_emb(sid)  # (B, E_st)
        e_seq = e.unsqueeze(1).expand(-1, L, -1)
        if self.has_cluster and cid is not None:
            c = self.cl_emb(cid)
            c_seq = c.unsqueeze(1).expand(-1, L, -1)
            x_aug = torch.cat([x_imp, e_seq, c_seq], dim=-1)
        else:
            x_aug = torch.cat([x_imp, e_seq], dim=-1)

        out, _ = self.gru(x_aug)  # (B,L,H)
        h_last = out[:, -1, :]

        dT_data = self.fc(h_last)  # (B,3)
        g = self.gate(h_last).squeeze(-1)  # (B,)
        dT_phys = g * dTdt_phys_last  # (B,)
        T_next = T_last.view(-1, 1) + dT_data + dT_phys.view(-1, 1)

        return T_next, dT_data, dT_phys, g


# ============================================================
# 8) Losses, training, evaluation, plotting
# ============================================================
def pinball_loss(pred, target, quantile):
    diff = target - pred
    return torch.mean(torch.maximum(quantile * diff, (quantile - 1) * diff))


def train_one_fold(
    model, train_loader, val_loader, epochs=8, lr=1e-3, weight_decay=1e-6, grad_clip=1.0
):
    device = DEVICE
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val_mae": np.inf, "state": None}

    for ep in range(1, epochs + 1):
        model.train()
        tr_losses = []
        for X, M, Dlt, dTdt, Tlast, y, sid, cid, _ in train_loader:
            nan_mask = torch.isnan(y)
            if nan_mask.all():
                continue
            X, M, Dlt = X.to(device), M.to(device), Dlt.to(device)
            dTdt, Tlast, y = dTdt.to(device), Tlast.to(device), y.to(device)
            sid, cid = sid.to(device), cid.to(device)

            opt.zero_grad()
            yhat_q, _, _, _ = model(X, M, Dlt, dTdt, Tlast, sid, cid)
            y10, y50, y90 = yhat_q[:, 0], yhat_q[:, 1], yhat_q[:, 2]

            mask = ~torch.isnan(y)
            l50 = nn.L1Loss()(y50[mask], y[mask])
            l10 = pinball_loss(y10[mask], y[mask], 0.10)
            l90 = pinball_loss(y90[mask], y[mask], 0.90)
            order_pen = torch.mean(torch.relu(y10 - y50) + torch.relu(y50 - y90))

            loss = l50 + 0.5 * (l10 + l90) + 0.1 * order_pen
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())

        # ---- validation
        model.eval()
        val_abs, val_sq, n = 0.0, 0.0, 0
        with torch.no_grad():
            for X, M, Dlt, dTdt, Tlast, y, sid, cid, _ in val_loader:
                X, M, Dlt = X.to(device), M.to(device), Dlt.to(device)
                dTdt, Tlast, y = dTdt.to(device), Tlast.to(device), y.to(device)
                sid, cid = sid.to(device), cid.to(device)
                yhat_q, _, _, _ = model(X, M, Dlt, dTdt, Tlast, sid, cid)
                y50 = yhat_q[:, 1]
                mask = ~torch.isnan(y)
                err = y50[mask] - y[mask]
                val_abs += torch.abs(err).sum().item()
                val_sq += (err**2).sum().item()
                n += mask.sum().item()
        if n > 0:
            val_mae = val_abs / n
            val_rmse = math.sqrt(val_sq / n)
        else:
            val_mae, val_rmse = np.inf, np.inf

        if val_mae < best["val_mae"]:
            best = {
                "val_mae": val_mae,
                "state": {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                },
            }
        print(
            f"Epoch {ep:02d} | train_loss={np.mean(tr_losses):.4f} | val_mae={val_mae:.3f} | val_rmse={val_rmse:.3f}"
        )

    model.load_state_dict(best["state"])
    return model, best["val_mae"]


def evaluate_and_plot(
    model, ds_val, val_indices, cfg: SeqConfig, tag: str, max_plots=6
):
    """Run predictions on validation subset, save CSV + plots with bands."""
    device = DEVICE
    loader = DataLoader(
        torch.utils.data.Subset(ds_val, val_indices),
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    model = model.to(device)
    model.eval()

    preds, preds_q10, preds_q90, trues, times, stations = [], [], [], [], [], []
    with torch.no_grad():
        for X, M, Dlt, dTdt, Tlast, y, sid, cid, meta in loader:
            X, M, Dlt = X.to(device), M.to(device), Dlt.to(device)
            dTdt, Tlast = dTdt.to(device), Tlast.to(device)
            sid, cid = sid.to(device), cid.to(device)
            yhat_q, _, _, _ = model(X, M, Dlt, dTdt, Tlast, sid, cid)
            preds.append(yhat_q[:, 1].cpu().numpy())
            preds_q10.append(yhat_q[:, 0].cpu().numpy())
            preds_q90.append(yhat_q[:, 2].cpu().numpy())
            trues.append(y.numpy())
            y_idx = meta["y_idx"]
            times.extend(ds_val.df.loc[y_idx, "datetime"].astype(str).tolist())
            stations.extend(ds_val.df.loc[y_idx, "station"].astype(str).tolist())

    yhat = np.concatenate(preds)
    y10 = np.concatenate(preds_q10)
    y90 = np.concatenate(preds_q90)
    ytrue = np.concatenate(trues)

    out = pd.DataFrame(
        {
            "datetime": pd.to_datetime(times),
            "station": stations,
            "y_true": ytrue,
            "y_pred_p50": yhat,
            "y_pred_p10": y10,
            "y_pred_p90": y90,
        }
    ).sort_values(["station", "datetime"])
    out.to_csv(f"outputs/val_predictions_{tag}.csv", index=False)

    # coverage & metrics
    mask = np.isfinite(out["y_true"])
    mae = np.mean(np.abs(out.loc[mask, "y_true"] - out.loc[mask, "y_pred_p50"]))
    rmse = math.sqrt(
        np.mean((out.loc[mask, "y_true"] - out.loc[mask, "y_pred_p50"]) ** 2)
    )
    coverage = np.mean(
        (out.loc[mask, "y_true"] >= out.loc[mask, "y_pred_p10"])
        & (out.loc[mask, "y_true"] <= out.loc[mask, "y_pred_p90"])
    )
    with open(f"outputs/metrics_{tag}.txt", "w") as f:
        f.write(f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nP10-P90 coverage: {coverage:.3f}\n")

    # plots
    import matplotlib.pyplot as plt

    for st in out["station"].dropna().unique()[:max_plots]:
        df_st = (
            out[out["station"] == st].sort_values("datetime").tail(7 * 24)
        )  # last week
        if len(df_st) < 24:
            continue
        plt.figure(figsize=(12, 4))
        t = df_st["datetime"]
        plt.plot(t, df_st["y_true"], label="Actual")
        plt.plot(t, df_st["y_pred_p50"], label="Pred (P50)")
        plt.fill_between(
            t, df_st["y_pred_p10"], df_st["y_pred_p90"], alpha=0.25, label="P10–P90"
        )
        plt.title(f"Station {st} — Actual vs Forecast (with bands) [{tag}]")
        plt.xlabel("Time")
        plt.ylabel("Temperature (°C)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"outputs/plot_{tag}_{st}.png", dpi=150)
        plt.close()

    return mae, rmse, coverage


# ============================================================
# 9) Station clustering + dendrogram
# ============================================================
def _station_feature_table_for_clustering(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Build per-station feature vectors (robust to missing values)."""
    rows = []
    for st, g in df_feat.groupby("station"):
        g = g.sort_values("datetime")
        lat = float(g["lat"].dropna().iloc[0]) if g["lat"].notna().any() else 0.0
        lon = float(g["lon"].dropna().iloc[0]) if g["lon"].notna().any() else 0.0
        T = g["temp_c"].astype(float)
        Sin = (
            g["S_in_Wm2"].astype(float)
            if "S_in_Wm2" in g
            else pd.Series(np.nan, index=g.index)
        )
        VPD = (
            g["VPD_pa"].astype(float)
            if "VPD_pa" in g
            else pd.Series(np.nan, index=g.index)
        )
        W = (
            g["wind_speed_ms"].astype(float)
            if "wind_speed_ms" in g
            else pd.Series(np.nan, index=g.index)
        )
        P = (
            g["precip_mm"].astype(float)
            if "precip_mm" in g
            else pd.Series(np.nan, index=g.index)
        )

        tmean, tstd = float(T.mean()), float(T.std())
        monthly = T.groupby(g["datetime"].dt.month).mean()
        seas_amp = (
            float(monthly.max() - monthly.min()) if monthly.notna().any() else 0.0
        )
        daily_rng = T.groupby(g["datetime"].dt.floor("D")).agg(
            lambda s: s.max() - s.min()
        )
        diurnal_amp = float(daily_rng.mean()) if daily_rng.notna().any() else 0.0
        corr_TS = (
            float(
                np.corrcoef(
                    T.fillna(method="ffill").fillna(method="bfill"), Sin.fillna(0.0)
                )[0, 1]
            )
            if Sin.notna().any()
            else 0.0
        )
        vpd_mean = float(VPD.mean()) if VPD.notna().any() else 0.0
        wind_mean = float(W.mean()) if W.notna().any() else 0.0
        precip_sum = float(P.sum()) if P.notna().any() else 0.0

        rows.append(
            {
                "station": st,
                "lat": lat,
                "lon": lon,
                "t_mean": tmean,
                "t_std": tstd,
                "season_amp": seas_amp,
                "diurnal_amp": diurnal_amp,
                "corr_T_Sin": corr_TS,
                "vpd_mean": vpd_mean,
                "wind_mean": wind_mean,
                "precip_sum": precip_sum,
            }
        )
    return pd.DataFrame(rows).set_index("station")


def _plot_dendrogram_from_sklearn_model(
    model, labels, filename="outputs/station_dendrogram.png"
):
    """Recreate sklearn example to plot a dendrogram from AgglomerativeClustering."""
    if not HAVE_SCIPY_CLUSTER:
        print("[Dendrogram] SciPy not available; skipping dendrogram plot.")
        return
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    sch.dendrogram(linkage_matrix, labels=labels, truncate_mode="level", p=6)
    plt.title("Hierarchical Clustering Dendrogram (stations)")
    plt.xlabel("Stations")
    plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def compute_station_clusters_and_dendrogram(
    df_feat: pd.DataFrame,
    n_clusters: int = 8,
    method: str = "ward",
    tag: str = "stations",
):
    """
    Build station feature table, scale, fit AgglomerativeClustering with distance_threshold=0 for dendrogram,
    save figure, then refit with n_clusters to assign labels.
    Returns:
      mapping: dict station -> cluster_id  (0..n_clusters-1)
      feat_table: the per-station feature dataframe
    """
    assert (
        HAVE_SKLEARN
    ), "scikit-learn is required for clustering. Please install scikit-learn."
    feats = _station_feature_table_for_clustering(df_feat)  # index = station
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)

    # Fit full tree for dendrogram
    model = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage=method, compute_full_tree=True
    )
    try:
        model.set_params(compute_distances=True)
    except Exception:
        pass
    model = model.fit(X)

    _plot_dendrogram_from_sklearn_model(
        model, labels=feats.index.tolist(), filename=f"outputs/dendrogram_{tag}.png"
    )

    # Cut the tree to get labels
    cut = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
    labels = cut.fit_predict(X)
    mapping = {st: int(lbl) for st, lbl in zip(feats.index.tolist(), labels)}
    pd.DataFrame({"station": feats.index, "cluster_id": labels}).to_csv(
        f"outputs/station_clusters_{tag}.csv", index=False
    )
    return mapping, feats


# ============================================================
# 10) Auto model selection (heuristics + optional tree classifier)
# ============================================================
def acf_at_lag(series: np.ndarray, lag: int) -> float:
    s = pd.Series(series).astype(float)
    if s.isna().all():
        return 0.0
    s = s.fillna(method="ffill").fillna(method="bfill")
    v = s.values
    if len(v) <= lag:
        return 0.0
    v = v - v.mean()
    num = np.dot(v[:-lag], v[lag:])
    den = np.dot(v, v)
    return float(num / den) if den > 0 else 0.0


def dataset_meta_signature(df: pd.DataFrame) -> Dict[str, float]:
    s = {}
    y = df["temp_c"].values
    s["missing_rate_y"] = float(np.mean(~np.isfinite(y)))
    max_gap = 0
    for _, g in df.groupby("station"):
        t = (
            pd.to_datetime(g["datetime"])
            .values.astype("datetime64[h]")
            .astype(np.int64)
        )
        if len(t) < 2:
            continue
        gap = np.max(np.diff(t))
        max_gap = max(max_gap, int(gap))
    s["max_gap_hours"] = max_gap
    s["acf_24h"] = acf_at_lag(y, 24)
    y_filled = pd.Series(y).fillna(method="ffill").fillna(method="bfill").values
    s["diff_std"] = float(np.std(np.diff(y_filled))) if len(y_filled) > 1 else 0.0
    return s


@dataclass
class PipelineChoice:
    denoise: Optional[str]
    use_grud: bool
    reason: str


def choose_pipeline_heuristic(
    df_feat: pd.DataFrame, requested_denoise: Optional[str]
) -> PipelineChoice:
    meta = dataset_meta_signature(df_feat)
    denoise = requested_denoise
    if requested_denoise is None:
        if meta["acf_24h"] >= 0.6 and meta["missing_rate_y"] < 0.05:
            denoise = "sarimax"
        elif meta["diff_std"] > 1.5:
            denoise = "wavelet"
        else:
            denoise = None
    use_grud = (meta["missing_rate_y"] >= 0.05) or (meta["max_gap_hours"] >= 6)
    reason = f"meta={meta}"
    return PipelineChoice(denoise=denoise, use_grud=use_grud, reason=reason)


def choose_pipeline_classifier(
    df_feat: pd.DataFrame, requested_denoise: Optional[str]
) -> PipelineChoice:
    if not HAVE_SKLEARN:
        return choose_pipeline_heuristic(df_feat, requested_denoise)
    metas, labels = [], []
    for st, g in df_feat.groupby("station"):
        m = dataset_meta_signature(g)
        metas.append(
            [m["missing_rate_y"], m["max_gap_hours"], m["acf_24h"], m["diff_std"]]
        )
        lbl = 1 if (m["missing_rate_y"] >= 0.05 or m["max_gap_hours"] >= 6) else 0
        labels.append(lbl)
    X = np.array(metas)
    y = np.array(labels)
    if len(np.unique(y)) < 2:
        return choose_pipeline_heuristic(df_feat, requested_denoise)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = GradientBoostingClassifier(random_state=42)
    clf.fit(Xtr, ytr)
    m_all = dataset_meta_signature(df_feat)
    x = np.array(
        [
            [
                m_all["missing_rate_y"],
                m_all["max_gap_hours"],
                m_all["acf_24h"],
                m_all["diff_std"],
            ]
        ]
    )
    pred = clf.predict(x)[0]
    use_grud = bool(pred == 1)
    heur = choose_pipeline_heuristic(df_feat, requested_denoise)
    return PipelineChoice(
        denoise=heur.denoise,
        use_grud=use_grud,
        reason=f"classifier->use_grud={use_grud}, {heur.reason}",
    )


# ============================================================
# 11) Backtesting driver (with clustering + embeddings)
# ============================================================
def run_backtest(
    df_feat: pd.DataFrame,
    base_inputs: List[str],
    horizon: int = 1,
    seq_len: int = 48,
    n_splits: int = 5,
    embargo_hours: int = 24,
    epochs: int = 8,
    batch_size: int = 512,
    use_classifier: bool = False,
    requested_denoise: Optional[str] = None,
    tag_prefix: str = "none",
    do_cluster: bool = True,
    n_clusters: int = 8,
    station_emb_dim: int = 16,
    cluster_emb_dim: int = 8,
):

    # Model selection
    choice = (
        choose_pipeline_classifier(df_feat, requested_denoise)
        if use_classifier
        else choose_pipeline_heuristic(df_feat, requested_denoise)
    )
    print(
        f"[ModelSelect] denoise={choice.denoise}, use_grud={choice.use_grud} | {choice.reason}"
    )

    # If selector changed denoise, rebuild features (only affects temp_c_smooth)
    if choice.denoise != requested_denoise:
        df_feat = build_features(df_feat, denoise=choice.denoise)

    # Clustering & dendrogram
    if do_cluster and HAVE_SKLEARN:
        try:
            mapping, feats = compute_station_clusters_and_dendrogram(
                df_feat, n_clusters=n_clusters, method="ward", tag=tag_prefix
            )
            df_feat = df_feat.copy()
            df_feat["cluster_id"] = (
                df_feat["station"].map(mapping).fillna(0).astype(int)
            )
        except Exception as e:
            print(f"[Clustering] Failed with error: {e}. Proceeding without clusters.")
            df_feat = df_feat.copy()
            df_feat["cluster_id"] = 0
    else:
        df_feat = df_feat.copy()
        df_feat["cluster_id"] = 0

    cfg = SeqConfig(
        input_cols=base_inputs,
        target_col="temp_c",
        seq_len=seq_len,
        horizon=horizon,
        stride=1,
    )
    ds = SeqDatasetGRUD(df_feat, cfg)
    splits = purged_time_splits(
        ds.sample_meta,
        n_splits=n_splits,
        horizon_hours=horizon,
        embargo_hours=embargo_hours,
    )

    # Model with embeddings
    hidden = 192 if choice.use_grud else 128
    model = PhysGRUDQuant(
        input_dim=len(base_inputs),
        num_stations=ds.num_stations,
        station_emb_dim=station_emb_dim,
        num_clusters=ds.num_clusters,
        cluster_emb_dim=(cluster_emb_dim if ds.num_clusters > 1 else 0),
        hidden=hidden,
        layers=2,
        dropout=0.1,
    )
    model.set_x_mean(ds.input_means)

    fold_rows = []
    for k, (tr_idx, va_idx) in enumerate(splits, 1):
        print(
            f"\n=== Fold {k}/{n_splits} | train={len(tr_idx)} | val={len(va_idx)} ==="
        )
        tr_loader = DataLoader(
            torch.utils.data.Subset(ds, tr_idx),
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        va_loader = DataLoader(
            torch.utils.data.Subset(ds, va_idx),
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )

        model, val_mae = train_one_fold(
            model,
            tr_loader,
            va_loader,
            epochs=epochs,
            lr=1e-3,
            weight_decay=1e-6,
            grad_clip=1.0,
        )
        tag = f"{tag_prefix}_fold{k}"
        mae, rmse, cov = evaluate_and_plot(model, ds, va_idx, cfg, tag=tag, max_plots=6)
        fold_rows.append(
            {
                "fold": k,
                "val_mae": val_mae,
                "eval_mae": mae,
                "eval_rmse": rmse,
                "p10p90_cov": cov,
            }
        )

        torch.save(model.state_dict(), f"models/model_{tag}.pt")
        gc.collect()
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(f"outputs/cv_metrics_{tag_prefix}.csv", index=False)
    print(fold_df)
    return fold_df


# ============================================================
# 12) Inputs list & FOUR MAIN FUNCTIONS (denoising variants)
# ============================================================
BASE_INPUTS = [
    # central signals
    "temp_c_smooth",
    "dewpoint_c",
    "rh_pct",
    "p_pa",
    # radiation & solar geometry
    "S_in_Wm2",
    "Ra_Wm2",
    "clearness_index",
    "cos_zenith",
    # physics proxies
    "VPD_pa",
    "Rn_Wm2",
    "Rns_Wm2",
    "Rnl_Wm2",
    # wind
    "wind_speed_ms",
    "wind_gust_ms",
    "wind_sin",
    "wind_cos",
    # precip
    "precip_mm",
    # time encodings
    "hod_sin",
    "hod_cos",
    "doy_sin",
    "doy_cos",
    # ranges (past-only)
    "temp_c_rng24",
    "S_in_Wm2_rng24",
    "VPD_pa_rng24",
    "Rn_Wm2_rng24",
]


def _prepare_features(denoise: Optional[str]):
    # Load raw data
    df_raw = load_df()
    # Augment with station metadata (lat, lon)
    df_aug = augment_with_station_meta(df_raw, DATABASE_URI)
    # Build physics-informed features (+ optional denoise)
    df_feat = build_features(df_aug, denoise=denoise)
    return df_feat


def main_run_none(
    horizon=1,
    seq_len=48,
    epochs=8,
    use_classifier=True,
    do_cluster=True,
    n_clusters=8,
    station_emb_dim=16,
    cluster_emb_dim=8,
):
    df_feat = _prepare_features(denoise=None)
    return run_backtest(
        df_feat,
        BASE_INPUTS,
        horizon=horizon,
        seq_len=seq_len,
        epochs=epochs,
        tag_prefix="none",
        use_classifier=use_classifier,
        do_cluster=do_cluster,
        n_clusters=n_clusters,
        station_emb_dim=station_emb_dim,
        cluster_emb_dim=cluster_emb_dim,
    )


def main_run_fft(
    horizon=1,
    seq_len=48,
    epochs=8,
    use_classifier=True,
    do_cluster=True,
    n_clusters=8,
    station_emb_dim=16,
    cluster_emb_dim=8,
):
    df_feat = _prepare_features(denoise="fft")
    return run_backtest(
        df_feat,
        BASE_INPUTS,
        horizon=horizon,
        seq_len=seq_len,
        epochs=epochs,
        tag_prefix="fft",
        use_classifier=use_classifier,
        requested_denoise="fft",
        do_cluster=do_cluster,
        n_clusters=n_clusters,
        station_emb_dim=station_emb_dim,
        cluster_emb_dim=cluster_emb_dim,
    )


def main_run_wavelet(
    horizon=1,
    seq_len=48,
    epochs=8,
    use_classifier=True,
    do_cluster=True,
    n_clusters=8,
    station_emb_dim=16,
    cluster_emb_dim=8,
):
    df_feat = _prepare_features(denoise="wavelet")
    return run_backtest(
        df_feat,
        BASE_INPUTS,
        horizon=horizon,
        seq_len=seq_len,
        epochs=epochs,
        tag_prefix="wavelet",
        use_classifier=use_classifier,
        requested_denoise="wavelet",
        do_cluster=do_cluster,
        n_clusters=n_clusters,
        station_emb_dim=station_emb_dim,
        cluster_emb_dim=cluster_emb_dim,
    )


def main_run_sarimax(
    horizon=1,
    seq_len=48,
    epochs=8,
    use_classifier=True,
    do_cluster=True,
    n_clusters=8,
    station_emb_dim=16,
    cluster_emb_dim=8,
):
    df_feat = _prepare_features(denoise="sarimax")
    return run_backtest(
        df_feat,
        BASE_INPUTS,
        horizon=horizon,
        seq_len=seq_len,
        epochs=epochs,
        tag_prefix="sarimax",
        use_classifier=use_classifier,
        requested_denoise="sarimax",
        do_cluster=do_cluster,
        n_clusters=n_clusters,
        station_emb_dim=station_emb_dim,
        cluster_emb_dim=cluster_emb_dim,
    )


# ============================================================
# 13) Station clustering entry-point (optional utility)
# ============================================================
def build_clusters_only(
    denoise: Optional[str] = None, n_clusters: int = 10, tag: str = "stations"
):
    df_raw = load_df()
    df_aug = augment_with_station_meta(df_raw, DATABASE_URI)
    df_feat = build_features(df_aug, denoise=denoise)
    mapping, feats = compute_station_clusters_and_dendrogram(
        df_feat, n_clusters=n_clusters, method="ward", tag=tag
    )
    return mapping, feats


# ============================================================
# 14) Main guard
# ============================================================
if __name__ == "__main__":
    # Example: run a short CV with no denoising, 1-hour horizon
    # metrics = main_run_none(
    metrics = main_run_sarimax(
        horizon=1,
        seq_len=48,
        epochs=4,
        use_classifier=True,
        do_cluster=True,
        n_clusters=8,
        station_emb_dim=16,
        cluster_emb_dim=8,
    )
    pass
