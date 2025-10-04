# Weather GRU + Physics (Gray‑Box) — Single‑File Pipeline

This repository contains a **single Python module** that builds **physics‑informed features** and trains a **GRU‑D**‑style neural forecaster with **station embeddings** and optional **hierarchical clustering** of stations. It supports **leakage‑safe backtesting** (purging & embargoing), **denoising options** (FFT, Wavelet, SARIMAX‑residual), and **uncertainty bands** (P10–P90).

> Works with your `sqlite3` database and the provided `load_df()` function. The pipeline automatically augments station metadata (lat/lon) from the `estacoes` table and handles **missing values** (NaNs) natively.

---

## Quickstart

1. **Install dependencies (recommended)**
   ```bash
   pip install numpy pandas torch matplotlib scikit-learn statsmodels pywavelets scipy
   ```

2. **Ensure the database path** in the script matches your environment:
   ```python
   DATA_FOLDER = Path("/opt/project/data")
   DATABASE_URI = str(DATA_FOLDER / "database.db")
   ```

3. **Run one of the four main functions** (choose the denoising strategy):
   ```python
   from weather_gru_phys_full import (
       main_run_none, main_run_fft, main_run_wavelet, main_run_sarimax
   )

   # Example: 1h ahead, 48h lookback, 5-fold CV, station & cluster embeddings
   metrics = main_run_none(horizon=1, seq_len=48, epochs=6,
                           use_classifier=True, do_cluster=True,
                           n_clusters=10, station_emb_dim=24, cluster_emb_dim=8)
   ```

4. **Outputs** (created under the working directory):
   - `models/model_<tag>_foldK.pt` — trained weights per fold  
   - `outputs/val_predictions_<tag>.csv` — validation predictions with P10/P50/P90  
   - `outputs/metrics_<tag>.txt` — MAE/RMSE and P10–P90 coverage  
   - `outputs/cv_metrics_<tag>.csv` — per‑fold scores  
   - `outputs/plot_<tag>_<station>.png` — Actual vs. Predicted with **confidence bands**  
   - `outputs/dendrogram_<tag>.png` — **Hierarchical Clustering Dendrogram** (if SciPy installed)  
   - `outputs/station_clusters_<tag>.csv` — station → cluster mapping

---

## What the pipeline does

- **Physics features**:
  - **Solar geometry & radiation**: clearness index, extraterrestrial radiation, solar zenith.
  - **Longwave radiation** via **Brutsaert** emissivity with a cloudiness factor.
  - **Psychrometrics**: saturation vapor pressure, vapor pressure deficit (VPD), virtual temperature, air density.
  - **Energy‑balance proxy**: simple slab model tendency added to the NN via a learned gate.
- **Denoising**: FFT low‑pass, Wavelet soft‑thresholding, or SARIMAX residualization.
- **Missing data**: **GRU‑D** mechanism with masks and time‑since‑last‑seen decays; target NaNs are masked out in the loss.
- **Embeddings**: **Station** embeddings (always); **Cluster** embeddings (optional, from hierarchical clustering).
- **Backtesting**: **Purged** cross‑validation with **embargo** to avoid temporal leakage.
- **Uncertainty**: Quantile heads (P10/P50/P90) used to draw confidence bands and compute coverage.

---

## Notes on your data

- The provided `load_df()` reads BAURU station and returns Portuguese‑named columns plus `datetime` and `id_fk`.
- The module **adds lat/lon** by joining `medicoes.id_fk` to `estacoes.id` (see `augment_with_station_meta()`).
- If you later load **multiple stations**, the pipeline scales automatically; clustering will become more informative.

---

## Dendrogram (Hierarchical Clustering)

We reproduce the **official scikit‑learn dendrogram recipe** by fitting an `AgglomerativeClustering` with `distance_threshold=0` and converting to a linkage matrix for `scipy.cluster.hierarchy.dendrogram`. The resulting PNG is saved in `outputs/`.

---

## References

### Physics & Meteorology
- **FAO‑56** — Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration — Guidelines for computing crop water requirements* (FAO Irrigation and drainage paper 56). (Solar geometry, net radiation, Penman–Monteith.)
- **Brutsaert, W.** (1975). On a derivable formula for long-wave radiation from clear skies. *Water Resources Research*, 11(5), 742–744.

### Missing Data & RNNs
- **Che, Z.**, Purushotham, S., Cho, K., Sontag, D., & Liu, Y. (2018). *Recurrent Neural Networks for Multivariate Time Series with Missing Values*. Scientific Reports 8, 6085. (GRU‑D)

### Denoising
- **Donoho, D. L., & Johnstone, I. M.** (1994). *Ideal spatial adaptation by wavelet shrinkage*. Biometrika.
- **Statsmodels SARIMAX** documentation.

### Clustering & Dendrogram
- **scikit‑learn Agglomerative Clustering Dendrogram example**:  
  https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html

---

## License

This file is provided as-is for research and internal prototyping.
