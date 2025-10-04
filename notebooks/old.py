# ============================================================
# 0) Imports & helpers
# ============================================================
import math, warnings, gc
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional filters
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

# ============================================================
# 1) Column normalization & unit conversions
# ============================================================
RENAME = {
    "Station": "station", "Station latitude": "lat", "Station longitude": "lon",
    "TOTAL PRECIPITATION, HOURLY (mm)": "precip_mm",
    "ATMOSPHERIC PRESSURE AT STATION LEVEL, HOURLY (mB)": "pressure_mb",
    "ATMOSPHERIC PRESSURE PREVIOUS HOUR (AUT) (mB)": "pressure_prev_mb",
    "GLOBAL RADIATION (kJ/m²)": "global_rad_kj_m2",
    "AIR TEMPERATURE – DRY BULB, HOURLY (°C)": "temp_c",
    "DEW POINT TEMPERATURE (°C)": "dewpoint_c",
    "AIR TEMPERATURE PREVIOUS HOUR (AUT) (°C)": "temp_prev_c",
    "DEW POINT TEMPERATURE PREVIOUS HOUR (AUT) (°C)": "dewpoint_prev_c",
    "RELATIVE HUMIDITY PREVIOUS HOUR (AUT) (%)": "rh_prev_pct",
    "RELATIVE HUMIDITY, HOURLY (%)": "rh_pct",
    "WIND DIRECTION, HOURLY (deg) (°)": "wind_dir_deg",
    "WIND, MAXIMUM GUST (m/s)": "wind_gust_ms",
    "WIND, HOURLY SPEED (m/s)": "wind_speed_ms"
}

def standardize_columns(df):
    df = df.rename(columns=RENAME).copy()
    if "timestamp" not in df.columns:
        # ensure there's a datetime column called 'timestamp'
        if "date" in df.columns:
            df["timestamp"] = pd.to_datetime(df["date"])
        elif df.index.dtype == "datetime64[ns]":
            df["timestamp"] = df.index
        else:
            raise ValueError("Please add/rename a datetime column to 'timestamp'.")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    # basic types
    for col in ["lat","lon","precip_mm","pressure_mb","pressure_prev_mb",
                "global_rad_kj_m2","temp_c","temp_prev_c","dewpoint_c","dewpoint_prev_c",
                "rh_pct","rh_prev_pct","wind_dir_deg","wind_gust_ms","wind_speed_ms"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

# ============================================================
# 2) Physics feature builders
# ============================================================
SIGMA = 5.670374419e-8  # W m^-2 K^-4
CP = 1004.0             # J kg^-1 K^-1
RD = 287.0              # J kg^-1 K^-1
EPS = 0.622

def es_hPa_from_Tc(Tc):
    # Alduchov & Eskridge (1996) Magnus form
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

def Lv_J_kg(Tc):
    return 2.501e6 - 2361.0 * Tc

def psychrometrics(df):
    Tc = df["temp_c"].values
    Td = df["dewpoint_c"].values
    p_pa = (df["pressure_mb"].values) * 100.0

    es_T = es_hPa_from_Tc(Tc) * 100.0    # Pa
    ea = es_hPa_from_Tc(Td) * 100.0      # Pa, using dewpoint
    vpd = np.maximum(es_T - ea, 0.0)     # Pa
    w = EPS * ea / np.maximum(p_pa - ea, 1.0)
    q = w / (1.0 + w)
    Tv = (Tc + 273.15) * (1.0 + 0.61 * q)
    rho = p_pa / (RD * Tv)
    Lv = Lv_J_kg(Tc)
    gamma = CP * p_pa / (EPS * Lv)
    out = pd.DataFrame({
        "p_pa": p_pa, "es_pa": es_T, "ea_pa": ea, "vpd_pa": vpd,
        "q_kgkg": q, "rho_air": rho, "Tv_K": Tv, "Lv_Jkg": Lv, "gamma": gamma
    }, index=df.index)
    return out

# --- Solar geometry (hourly extraterrestrial radiation Ra and zenith) ---
# Using FAO-56 hourly approximation (sufficient for features)
def fao56_extraterrestrial_radiation_hourly(ts_utc, lat_deg, lon_deg):
    # Returns Ra (W/m^2) approximate for the hour centered at ts.
    # Steps follow FAO-56 (MJ/m2/h -> convert to W/m2)
    lat = np.radians(lat_deg)
    # Julian day
    n = ts_utc.timetuple().tm_yday
    # inverse relative distance Earth-Sun
    dr = 1 + 0.033 * np.cos(2 * np.pi * n / 365.0)
    # solar declination
    delta = 0.409 * np.sin(2 * np.pi * n / 365.0 - 1.39)
    # Approximate local solar time using longitude (timezone ≈ round(lon/15))
    tz = np.round(lon_deg / 15.0)
    lst = ts_utc + pd.to_timedelta((lon_deg/15.0 - tz), unit="h")
    omega = (lst.hour + lst.minute/60.0 + lst.second/3600.0 - 12.0) * (np.pi/12.0)  # hour angle center
    # use 1-hour width
    omega1 = omega - np.pi/24.0; omega2 = omega + np.pi/24.0

    Gsc = 0.0820  # MJ m^-2 min^-1 (solar constant)
    # Ra in MJ/m2/h (FAO-56 Eq. 28-33 style)
    Ra_MJ = (12*60/np.pi) * Gsc * dr * (
        (np.cos(lat)*np.cos(delta))*(np.sin(omega2) - np.sin(omega1)) +
        (np.pi*(omega2 - omega1)/180.0)*np.sin(lat)*np.sin(delta)
    )
    Ra_Wm2 = np.maximum(Ra_MJ * 1e6 / 3600.0, 0.0)  # to W/m2
    # Cosine of zenith (clip)
    cosz = np.maximum(np.sin(lat)*np.sin(delta) + np.cos(lat)*np.cos(delta)*np.cos(omega), 0.0)
    return Ra_Wm2, cosz

def radiation_features(df):
    S_in = df["global_rad_kj_m2"].values / 3.6  # to W/m^2
    Ra = np.empty(len(df))
    cosz = np.empty(len(df))
    for i, (ts, lat, lon) in enumerate(zip(df["timestamp"], df["lat"], df["lon"])):
        ra_i, cosz_i = fao56_extraterrestrial_radiation_hourly(ts, lat, lon)
        Ra[i] = ra_i; cosz[i] = cosz_i
    Kt = np.divide(S_in, np.maximum(Ra, 1.0), out=np.zeros_like(S_in), where=Ra>1.0)
    out = pd.DataFrame({"S_in_Wm2": S_in, "Ra_Wm2": Ra, "cos_zenith": cosz, "clearness_index": Kt}, index=df.index)
    return out

def longwave_downward_brutsaert(df, psych, rad):
    # Clear-sky emissivity (Brutsaert 1975) and cloud adjustment (Crawford & Duchon-like)
    Tk = df["temp_c"].values + 273.15
    e_kPa = psych["ea_pa"].values / 1000.0
    eps_clear = 1.24 * np.power(np.maximum(e_kPa / np.maximum(Tk, 1.0), 1e-8), 1.0/7.0)
    eps_clear = np.clip(eps_clear, 0.5, 1.0)
    cloud = np.clip(1.0 - np.clip(rad["clearness_index"].values, 0.0, 1.0), 0.0, 1.0)
    eps_sky = eps_clear * (1.0 + 0.22 * cloud**2)
    Ld = eps_sky * SIGMA * Tk**4
    return pd.DataFrame({"eps_sky": eps_sky, "L_down_Wm2": Ld}, index=df.index)

def energy_balance_proxies(df, psych, rad, eps_s=0.98, alpha0=0.23):
    Rns = (1.0 - alpha0) * rad["S_in_Wm2"].values
    Rnl = rad["L_down_Wm2"].values - eps_s * SIGMA * (df["temp_c"].values + 273.15)**4
    Rn = Rns + Rnl
    # simple closures; coefficients learned later in NN. Here we just pass proxies:
    U = np.nan_to_num(df["wind_speed_ms"].values, nan=0.0)
    VPD = psych["vpd_pa"].values
    out = pd.DataFrame({
        "Rns_Wm2": Rns, "Rnl_Wm2": Rnl, "Rn_Wm2": Rn,
        "U_ms": U, "VPD_pa": VPD
    }, index=df.index)
    return out

# ============================================================
# 3) Optional denoising / residualization
# ============================================================
def fft_lowpass(x, cutoff_hours=6, dt_hours=1.0):
    x = pd.Series(x).astype(float)
    n = len(x)
    if n < 8: return x.values
    X = np.fft.rfft(x.fillna(method="ffill").fillna(method="bfill").values)
    freqs = np.fft.rfftfreq(n, d=dt_hours)
    X[freqs > 1.0/cutoff_hours] = 0.0
    y = np.fft.irfft(X, n=n)
    return y

def butter_lowpass(x, cutoff_hours=6, dt_hours=1.0, order=4):
    if not HAVE_SCIPY: return x
    fs = 1.0/dt_hours
    from scipy.signal import butter, filtfilt
    b, a = butter(order, (1.0/cutoff_hours)/(0.5*fs), btype="low")
    return filtfilt(b, a, pd.Series(x).fillna(method="ffill").fillna(method="bfill").values)

def wavelet_denoise(x, wave="db4", level=2):
    if not HAVE_PYWT: return x
    coeffs = pywt.wavedec(pd.Series(x).fillna(method="ffill").fillna(method="bfill").values, wave, mode='periodization')
    # universal soft threshold
    sigma = np.median(np.abs(coeffs[-1]))/0.6745
    uth = sigma*np.sqrt(2*np.log(len(x)))
    coeffs[1:] = [pywt.threshold(c, value=uth, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wave, mode='periodization')[:len(x)]

def sarimax_residuals(y, order=(2,0,2), seasonal_order=(1,0,1,24*7)):
    if not HAVE_STATSMODELS: return y - pd.Series(y).rolling(24, min_periods=1).mean().values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mod = sm.tsa.statespace.SARIMAX(y, order=order, seasonal_order=seasonal_order, enforce_stationarity=False, enforce_invertibility=False)
        res = mod.fit(disp=False, maxiter=200)
    fitted = res.fittedvalues
    resid = y - fitted
    return resid.values

# ============================================================
# 4) Full feature assembly (per-row, no leakage)
# ============================================================
def build_features(df, denoise=None):
    df = standardize_columns(df)
    df = df.sort_values(["station", "timestamp"]).reset_index(drop=True)

    # Psychrometrics & radiation
    psy = psychrometrics(df)
    rad_base = radiation_features(df)
    lw = longwave_downward_brutsaert(df, psy, rad_base)
    eb = energy_balance_proxies(df, psy, pd.concat([rad_base, lw], axis=1))

    # Time encodings (UTC-based; local offset is approximated in Ra)
    ts = df["timestamp"]
    hod = (ts.dt.hour + ts.dt.minute/60.0).values
    hod_sin = np.sin(2*np.pi*hod/24.0); hod_cos = np.cos(2*np.pi*hod/24.0)
    doy = ts.dt.dayofyear.values.astype(float)
    doy_sin = np.sin(2*np.pi*(doy/365.25)); doy_cos = np.cos(2*np.pi*(doy/365.25))

    # Wind direction as sin/cos
    wd_rad = np.radians(np.nan_to_num(df["wind_dir_deg"].values, nan=0.0))
    wind_sin, wind_cos = np.sin(wd_rad), np.cos(wd_rad)

    # Ranges (last 24h) — compute per station to avoid leakage
    features = pd.concat([df[["station","timestamp","lat","lon","temp_c","dewpoint_c","rh_pct",
                               "precip_mm","pressure_mb","global_rad_kj_m2","wind_speed_ms","wind_gust_ms"]],
                          psy, rad_base, lw, eb], axis=1)

    def add_rolling_ranges(g):
        g = g.sort_values("timestamp")
        for col in ["temp_c","pressure_mb","S_in_Wm2","VPD_pa","Rn_Wm2"]:
            if col in g.columns:
                rmax = g[col].rolling(24, min_periods=6).max()
                rmin = g[col].rolling(24, min_periods=6).min()
                g[f"{col}_rng24"] = (rmax - rmin).values
        return g

    features = features.groupby("station", group_keys=False).apply(add_rolling_ranges)

    # Denoise target OPTIONAL (one method only)
    y = df["temp_c"].values
    if denoise == "fft":
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(lambda s: fft_lowpass(s.values, cutoff_hours=6))
        features["temp_c_smooth"] = y_smooth
    elif denoise == "butter" and HAVE_SCIPY:
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(lambda s: butter_lowpass(s.values, cutoff_hours=6))
        features["temp_c_smooth"] = y_smooth
    elif denoise == "wavelet" and HAVE_PYWT:
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(lambda s: wavelet_denoise(s.values))
        features["temp_c_smooth"] = y_smooth
    elif denoise == "sarimax":
        y_smooth = features.groupby("station", group_keys=False)["temp_c"].transform(lambda s: sarimax_residuals(s.values) + s.rolling(24, min_periods=1).mean().values)
        features["temp_c_smooth"] = y_smooth
    else:
        features["temp_c_smooth"] = features["temp_c"]

    # Fourier time features
    features["hod_sin"], features["hod_cos"] = hod_sin, hod_cos
    features["doy_sin"], features["doy_cos"] = doy_sin, doy_cos
    features["wind_sin"], features["wind_cos"] = wind_sin, wind_cos

    # Pre-compute a physics tendency proxy (no learnable params yet)
    # Use nominal constants to create a "first guess" dT/dt
    Ce_nom = 1.5e6  # J m^-2 K^-1 (tunable; NN will reweight)
    kH_nom, kLE_nom, kG_nom = 5.0, 5e-7, 0.05
    T = features["temp_c_smooth"].values
    U = np.nan_to_num(features["wind_speed_ms"].values, nan=0.0)
    Rns = features["Rns_Wm2"].values
    Rnl = features["Rnl_Wm2"].values
    VPD = features["VPD_pa"].values
    H = kH_nom * U * (T - pd.Series(T).rolling(24, min_periods=6).mean().fillna(method="bfill").values)
    LE = kLE_nom * U * VPD
    G = kG_nom * Rns
    dTdt_phys = (Rns + Rnl - H - LE - G) / Ce_nom
    features["dTdt_phys"] = dTdt_phys

    return features

# ============================================================
# 5) Windowed dataset (multi-station)
# ============================================================
class SeqDataset(Dataset):
    def __init__(self, feat_df, input_cols, target_col="temp_c", horizon=1, seq_len=48, stride=1, stations=None, time_index=None):
        self.df = feat_df.copy()
        if stations is not None:
            self.df = self.df[self.df["station"].isin(stations)].copy()
        self.df = self.df.sort_values(["timestamp","station"]).reset_index(drop=True)
        self.input_cols = input_cols
        self.target_col = target_col
        self.horizon = horizon
        self.seq_len = seq_len
        self.stride = stride

        # Build index of valid windows (by station to avoid crossing boundaries)
        self.rows = []
        for st, g in self.df.groupby("station"):
            g = g.sort_values("timestamp")
            vals = g[self.input_cols + [self.target_col, "dTdt_phys"]].values
            n = len(g)
            for i in range(0, n - seq_len - horizon + 1, stride):
                start, end = i, i + seq_len
                y_t = i + seq_len + horizon - 1
                # Save (global row indices)
                self.rows.append((g.index[start:end].to_numpy(), g.index[end-1], g.index[y_t]))
        self.rows = np.array(self.rows, dtype=object)

        # For purged CV, keep mapping to times
        self.sample_meta = pd.DataFrame({
            "x_end_idx": [r[1] for r in self.rows],
            "y_idx": [r[2] for r in self.rows]
        })
        self.sample_meta["timestamp_x_end"] = self.df.loc[self.sample_meta["x_end_idx"], "timestamp"].values
        self.sample_meta["timestamp_y"] = self.df.loc[self.sample_meta["y_idx"], "timestamp"].values

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        idx_seq, idx_xend, idx_y = self.rows[i]
        X = self.df.loc[idx_seq, self.input_cols].values.astype(np.float32)
        # physics tendency at last step (additive path)
        dTdt_phys = self.df.loc[idx_xend, "dTdt_phys"].astype(np.float32)
        # last temperature (for delta update)
        T_last = self.df.loc[idx_xend, "temp_c"].astype(np.float32)
        y = self.df.loc[idx_y, self.target_col].astype(np.float32)
        return torch.from_numpy(X), torch.tensor(dTdt_phys), torch.tensor(T_last), torch.tensor(y)

# ============================================================
# 6) Purged & embargoed time split
# ============================================================
def purged_time_splits(meta_times, n_splits=5, horizon_hours=1, embargo_hours=24):
    """
    meta_times: DataFrame with columns ['timestamp_x_end','timestamp_y'] for each sample
    Splits by timestamp_y (label time). Purge +/- horizon and embargo after val.
    """
    y_times = pd.to_datetime(meta_times["timestamp_y"]).values
    # Build time-based fold boundaries
    t_sorted = np.sort(y_times)
    fold_edges = np.linspace(0, len(t_sorted), n_splits+1, dtype=int)
    splits = []
    for k in range(n_splits):
        val_lo_t = t_sorted[fold_edges[k]]
        val_hi_t = t_sorted[fold_edges[k+1]-1]
        # boolean masks
        val_mask = (y_times >= val_lo_t) & (y_times <= val_hi_t)
        # Purge +/- horizon
        horizon = np.timedelta64(horizon_hours, 'h')
        embargo = np.timedelta64(embargo_hours, 'h')
        purge_lo = val_lo_t - horizon; purge_hi = val_hi_t + horizon
        # Embargo after validation
        embargo_lo = val_hi_t; embargo_hi = val_hi_t + embargo

        train_mask = ~val_mask
        x_times = pd.to_datetime(meta_times["timestamp_x_end"]).values
        # Remove samples whose input or label overlap purge window
        overlap = ((y_times >= purge_lo) & (y_times <= purge_hi)) | ((x_times >= purge_lo) & (x_times <= purge_hi))
        train_mask &= ~overlap
        # Remove embargoed region after validation
        train_mask &= ~((y_times > embargo_lo) & (y_times <= embargo_hi))
        splits.append((np.where(train_mask)[0], np.where(val_mask)[0]))
    return splits

# ============================================================
# 7) GRU + Physics additive model (one-step; extendable to multi-horizon)
# ============================================================
class PhysGRU(nn.Module):
    def __init__(self, input_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden, num_layers=layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))  # predicts ΔT_data
        # Physics gate and scaling (positivity via softplus)
        self.log_Ce = nn.Parameter(torch.log(torch.tensor(1.5e6)))     # J m^-2 K^-1
        self.kH = nn.Parameter(torch.tensor(5.0))
        self.kLE = nn.Parameter(torch.tensor(5e-7))
        self.kG = nn.Parameter(torch.tensor(0.05))
        self.gate = nn.Sequential(nn.Linear(hidden, 1), nn.Sigmoid())  # [0,1] weight on physics term

    def forward(self, x_seq, dTdt_phys, T_last):
        # x_seq: (B, L, D)
        out, _ = self.gru(x_seq)             # (B, L, H)
        h_last = out[:, -1, :]               # (B, H)
        dT_data = self.head(h_last).squeeze(-1)   # (B,)
        g = self.gate(h_last).squeeze(-1)         # (B,)

        # Calibrate physics tendency via learned coefficients (implicitly)
        # Here dTdt_phys was computed with nominal constants; we let the gate scale it
        dT_phys = g * dTdt_phys              # (B,)

        # Predict next temperature
        T_next = T_last + dT_data + dT_phys
        return T_next, dT_data, dT_phys

# ============================================================
# 8) Training & evaluation
# ============================================================
def train_one_fold(model, train_loader, val_loader, device="cuda" if torch.cuda.is_available() else "cpu",
                   epochs=10, lr=1e-3, weight_decay=1e-6, grad_clip=1.0):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best = {"val_mae": np.inf, "state": None}
    loss_fn = nn.L1Loss()   # MAE; switch to Huber if outliers
    for ep in range(1, epochs+1):
        model.train()
        tr_losses = []
        for X, dTdt, Tlast, y in train_loader:
            X, dTdt, Tlast, y = X.to(device), dTdt.to(device), Tlast.to(device), y.to(device)
            opt.zero_grad()
            yhat, _, _ = model(X, dTdt, Tlast)
            loss = loss_fn(yhat, y)
            loss.backward()
            if grad_clip: nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            tr_losses.append(loss.item())
        # validation
        model.eval()
        with torch.no_grad():
            val_mae, val_rmse, n = 0.0, 0.0, 0
            for X, dTdt, Tlast, y in val_loader:
                X, dTdt, Tlast, y = X.to(device), dTdt.to(device), Tlast.to(device), y.to(device)
                yhat, _, _ = model(X, dTdt, Tlast)
                err = (yhat - y).detach()
                val_mae += torch.abs(err).sum().item()
                val_rmse += (err**2).sum().item()
                n += len(y)
            val_mae /= n; val_rmse = math.sqrt(val_rmse/n)
        if val_mae < best["val_mae"]:
            best = {"val_mae": val_mae, "state": {k: v.cpu().clone() for k, v in model.state_dict().items()}}
        print(f"Epoch {ep:3d} | train_mae={np.mean(tr_losses):.3f} | val_mae={val_mae:.3f} | val_rmse={val_rmse:.3f}")
    model.load_state_dict(best["state"])
    return model, best["val_mae"]

def backtest_cv(feat_df, input_cols, target_col="temp_c", seq_len=48, horizon=1,
                batch_size=256, n_splits=5, embargo_hours=24, denoise=None, epochs=10):
    ds = SeqDataset(feat_df, input_cols=input_cols, target_col=target_col, seq_len=seq_len, horizon=horizon)
    splits = purged_time_splits(ds.sample_meta, n_splits=n_splits, horizon_hours=horizon, embargo_hours=embargo_hours)
    fold_metrics = []
    for k, (tr_idx, va_idx) in enumerate(splits, 1):
        tr_loader = DataLoader(torch.utils.data.Subset(ds, tr_idx), batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        va_loader = DataLoader(torch.utils.data.Subset(ds, va_idx), batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
        model = PhysGRU(input_dim=len(input_cols), hidden=128, layers=2, dropout=0.2)
        model, val_mae = train_one_fold(model, tr_loader, va_loader, epochs=epochs)
        # test on validation fold (already computed)
        fold_metrics.append({"fold": k, "val_mae": val_mae})
        # free CUDA mem between folds
        del model; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
    return pd.DataFrame(fold_metrics)

# ============================================================
# 9) How to call this on your DataFrame
# ============================================================
# df_raw = ...  # Your long-form dataframe with the exact columns; include 'timestamp'
# features = build_features(df_raw, denoise=None)  # or 'fft'/'wavelet'/'butter'/'sarimax'
# Choose inputs (avoid direct collinearity; include physics & time encodings)
# You can expand this list after checking correlations:
# base_inputs = [
#   "temp_c_smooth","dewpoint_c","rh_pct","p_pa","S_in_Wm2","Ra_Wm2","clearness_index","cos_zenith",
#   "VPD_pa","Rn_Wm2","Rns_Wm2","Rnl_Wm2","wind_speed_ms","wind_gust_ms","wind_sin","wind_cos",
#   "precip_mm","hod_sin","hod_cos","doy_sin","doy_cos",
#   "temp_c_rng24","S_in_Wm2_rng24","VPD_pa_rng24","Rn_Wm2_rng24"
# ]
# metrics = backtest_cv(features, input_cols=base_inputs, target_col="temp_c", seq_len=48, horizon=1,
#                       batch_size=512, n_splits=5, embargo_hours=24, epochs=8)
# print(metrics)
