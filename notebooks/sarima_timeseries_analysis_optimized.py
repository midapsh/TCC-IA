#!/usr/bin/env python3
"""
Optimized SARIMA Time Series Analysis Script
Enhanced with:
- Memory management and garbage collection
- Parallel processing for CPU optimization
- Batch processing to reduce memory footprint
- Efficient data structures
"""

import gc
import os
import sys
import pickle
import warnings
from os import makedirs
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from functools import partial
import multiprocessing as mp
from typing import Dict, List, Tuple, Any
import psutil

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend to save memory
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.mixture import GaussianMixture

# Statistical and time series imports
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

warnings.filterwarnings("ignore")

# Memory management settings
gc.enable()
gc.set_threshold(700, 10, 10)  # More aggressive garbage collection

# Configuration
DIMS = [
    # "precipitacao_total_horario",
    # "pressao_atmosferica_ao_nivel_da_estacao_horaria",
    # "pressao_atmosferica_max_na_hora_ant",
    # "pressao_atmosferica_min_na_hora_ant",
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

DIMENSION_NAMES = {
    # "precipitacao_total_horario": "Hourly Precipitation",
    # "pressao_atmosferica_ao_nivel_da_estacao_horaria": "Atmospheric Pressure (Station Level)",
    # "pressao_atmosferica_max_na_hora_ant": "Max Atmospheric Pressure (Previous Hour)",
    # "pressao_atmosferica_min_na_hora_ant": "Min Atmospheric Pressure (Previous Hour)",
    "radiacao_global": "Global Radiation",
    "temperatura_do_ar_bulbo_seco_horaria": "Air Temperature (Dry Bulb)",
    "temperatura_do_ponto_de_orvalho": "Dew Point Temperature",
    "temperatura_maxima_na_hora_ant": "Max Temperature (Previous Hour)",
    "temperatura_minima_na_hora_ant": "Min Temperature (Previous Hour)",
    "temperatura_orvalho_max_na_hora_ant": "Max Dew Point (Previous Hour)",
    "temperatura_orvalho_min_na_hora_ant": "Min Dew Point (Previous Hour)",
    "umidade_relativa_max_na_hora_ant": "Max Relative Humidity (Previous Hour)",
    "umidade_relativa_min_na_hora_ant": "Min Relative Humidity (Previous Hour)",
    "umidade_relativa_do_ar_horaria": "Hourly Relative Humidity",
    "vento_direcao_horaria": "Wind Direction",
    "vento_rajada_maxima": "Max Wind Gust",
    "vento_velocidade_horaria": "Wind Speed",
}

# Output folders
BASE_OUTPUT_FOLDER = Path("sarima_analysis_optimized")
MODELS_FOLDER = BASE_OUTPUT_FOLDER / "models"
PLOTS_FOLDER = BASE_OUTPUT_FOLDER / "plots"
TABLES_FOLDER = BASE_OUTPUT_FOLDER / "tables"
MULTIMODAL_FOLDER = BASE_OUTPUT_FOLDER / "multimodal_analysis"

# Create folders
for folder in [MODELS_FOLDER, PLOTS_FOLDER, TABLES_FOLDER, MULTIMODAL_FOLDER]:
    makedirs(folder, exist_ok=True)


def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024


def clean_memory():
    """Force garbage collection and clear memory"""
    gc.collect()


class OptimizedSARIMAAnalyzer:
    """Memory-optimized SARIMA analyzer with parallel processing support"""

    def __init__(self, dimension_name: str, friendly_name: str):
        self.dimension_name = dimension_name
        self.friendly_name = friendly_name
        self.results = {}
        self.multimodal_info = {}

    def analyze_dimension(self, data: pd.Series) -> Dict:
        """Main analysis function optimized for memory"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.friendly_name}")
        print(f"Initial memory: {get_memory_usage():.1f} MB")
        print(f"{'='*60}")

        try:
            # Clean data
            clean_data = data.dropna()
            if len(clean_data) < 100:
                return {"error": f"Insufficient data: {len(clean_data)} points"}

            print(f"Data points: {len(clean_data)}")

            # 1. Basic statistics (low memory)
            stats_results = self._calculate_statistics(clean_data)
            self.results["statistics"] = stats_results

            # 2. ADF test (low memory)
            adf_results = self._perform_adf_test(clean_data)
            self.results["adf"] = adf_results

            # 3. Frequency analysis (moderate memory)
            freq_results = self._detect_frequencies(clean_data)
            self.results["frequency_components"] = freq_results
            clean_memory()

            # 4. Multimodal detection (moderate memory)
            multimodal_results = self._detect_multimodal(clean_data)
            self.results["multimodal_info"] = multimodal_results
            clean_memory()

            # 5. Create plots (high memory - done in batches)
            self._create_plots_batch(clean_data)
            clean_memory()

            # 6. Fit SARIMA model (high memory)
            sarima_results = self._fit_sarima_optimized(clean_data)
            self.results["sarima_params"] = sarima_results
            clean_memory()

            print(f"Final memory: {get_memory_usage():.1f} MB")
            return self.results

        except Exception as e:
            print(f"Error: {e}")
            return {"error": str(e)}
        finally:
            clean_memory()

    def _calculate_statistics(self, data: pd.Series) -> Dict:
        """Calculate statistics with minimal memory usage"""
        # Use numpy for efficiency
        data_array = data.values

        stats_dict = {
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "std_dev": float(np.std(data_array)),
            "variance": float(np.var(data_array)),
            "skewness": float(stats.skew(data_array)),
            "kurtosis": float(stats.kurtosis(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "q25": float(np.percentile(data_array, 25)),
            "q75": float(np.percentile(data_array, 75)),
        }
        stats_dict["iqr"] = stats_dict["q75"] - stats_dict["q25"]

        # Save to CSV
        pd.DataFrame(stats_dict, index=[0]).T.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_statistics.csv"
        )

        print(
            f"Stats - Skew: {stats_dict['skewness']:.3f}, Kurt: {stats_dict['kurtosis']:.3f}"
        )
        return stats_dict

    def _perform_adf_test(self, data: pd.Series) -> Dict:
        """Perform ADF test efficiently"""
        adf_result = adfuller(data.values, autolag="AIC")

        adf_dict = {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "used_lag": int(adf_result[2]),
            "n_obs": int(adf_result[3]),
            "is_stationary": bool(adf_result[1] < 0.05),
        }

        if len(adf_result) > 4:
            adf_dict["critical_values"] = adf_result[4]

        # Save results
        pd.DataFrame([adf_dict]).to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_adf_test.csv", index=False
        )

        print(
            f"ADF - Stationary: {adf_dict['is_stationary']} (p={adf_dict['p_value']:.4f})"
        )
        return adf_dict

    def _detect_frequencies(self, data: pd.Series) -> Dict:
        """Detect frequency components using FFT"""
        data_array = data.values
        n = len(data_array)

        # Use rfft for real data (more efficient)
        fft_vals = np.fft.rfft(data_array)
        freqs = np.fft.rfftfreq(n)

        # Power spectrum
        power = np.abs(fft_vals) ** 2

        # Find significant frequencies
        threshold = np.percentile(power[1:], 90)  # Skip DC component
        sig_mask = power > threshold
        sig_freqs = freqs[sig_mask]

        # Classify frequencies
        low_freq = np.sum(sig_freqs < 1 / 168)
        high_freq = np.sum(sig_freqs > 1 / 12)
        mid_freq = np.sum((sig_freqs >= 1 / 168) & (sig_freqs <= 1 / 12))

        # Find dominant frequency
        dom_idx = np.argmax(power[1:]) + 1
        dom_freq = float(freqs[dom_idx]) if dom_idx < len(freqs) else 0

        freq_dict = {
            "low_frequency_count": int(low_freq),
            "high_frequency_count": int(high_freq),
            "mid_frequency_count": int(mid_freq),
            "dominant_frequency": dom_freq,
        }

        # Save results
        pd.DataFrame([freq_dict]).to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_frequency_components.csv",
            index=False,
        )

        print(f"Frequencies - Low: {low_freq}, Mid: {mid_freq}, High: {high_freq}")

        # Clear large arrays
        del fft_vals, power
        return freq_dict

    def _detect_multimodal(self, data: pd.Series) -> Dict:
        """Detect multimodal distribution efficiently"""
        # Downsample if data is too large
        if len(data) > 10000:
            sample_data = data.sample(n=10000, random_state=42).values
        else:
            sample_data = data.values

        sample_data = sample_data.reshape(-1, 1)

        # Test for multimodality
        bic_scores = []
        for n_comp in range(1, min(6, len(np.unique(sample_data)))):
            try:
                gmm = GaussianMixture(
                    n_components=n_comp, random_state=42, max_iter=100
                )
                gmm.fit(sample_data)
                bic_scores.append(gmm.bic(sample_data))
            except:
                bic_scores.append(np.inf)

        optimal_components = np.argmin(bic_scores) + 1

        multimodal_dict = {
            "is_multimodal": bool(optimal_components > 1),
            "n_modes": int(optimal_components),
        }

        if optimal_components > 1:
            # Fit final model
            gmm = GaussianMixture(n_components=optimal_components, random_state=42)
            gmm.fit(sample_data)
            multimodal_dict["mode_means"] = gmm.means_.flatten().tolist()
            multimodal_dict["mode_weights"] = gmm.weights_.tolist()

            print(f"Multimodal: {optimal_components} modes detected")
        else:
            print("Unimodal distribution")

        # Save results
        pd.DataFrame(
            [
                {
                    "is_multimodal": multimodal_dict["is_multimodal"],
                    "n_modes": multimodal_dict["n_modes"],
                }
            ]
        ).to_csv(TABLES_FOLDER / f"{self.dimension_name}_multimodal.csv", index=False)

        return multimodal_dict

    def _create_plots_batch(self, data: pd.Series):
        """Create plots in batches to manage memory"""
        # ACF/PACF plot
        self._create_acf_pacf_plot(data)
        clean_memory()

        # Distribution plot
        self._create_distribution_plot(data)
        clean_memory()

    def _create_acf_pacf_plot(self, data: pd.Series):
        """Create ACF/PACF plots efficiently"""
        try:
            fig, axes = plt.subplots(2, 1, figsize=(10, 6))

            # Limit lags for memory efficiency
            max_lags = min(40, len(data) // 4)

            plot_acf(data.values, lags=max_lags, ax=axes[0], alpha=0.05)
            axes[0].set_title(f"ACF - {self.friendly_name}", fontsize=10)

            plot_pacf(data.values, lags=max_lags, ax=axes[1], alpha=0.05)
            axes[1].set_title(f"PACF - {self.friendly_name}", fontsize=10)

            plt.tight_layout()
            plt.savefig(
                PLOTS_FOLDER / f"{self.dimension_name}_acf_pacf.png",
                dpi=150,  # Reduced DPI for memory
                bbox_inches="tight",
            )
            plt.close("all")

            print("ACF/PACF plots created")
        except Exception as e:
            print(f"Error creating ACF/PACF plots: {e}")
        finally:
            plt.close("all")

    def _create_distribution_plot(self, data: pd.Series):
        """Create distribution analysis plot efficiently"""
        try:
            # Downsample for plotting if needed
            if len(data) > 5000:
                plot_data = data.sample(n=5000, random_state=42)
            else:
                plot_data = data

            fig = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=(
                    "Histogram",
                    "Q-Q Plot",
                    "Box Plot",
                    "Time Series Sample",
                ),
            )

            # Histogram
            fig.add_trace(
                go.Histogram(x=plot_data.values, nbinsx=30, name="Hist"), row=1, col=1
            )

            # Q-Q plot
            theoretical = stats.norm.ppf(
                np.linspace(0.01, 0.99, min(100, len(plot_data)))
            )
            sample = np.sort(plot_data.values)
            sample_interp = np.interp(
                np.linspace(0, len(sample) - 1, len(theoretical)),
                np.arange(len(sample)),
                sample,
            )

            fig.add_trace(
                go.Scatter(
                    x=theoretical, y=sample_interp, mode="markers", marker=dict(size=3)
                ),
                row=1,
                col=2,
            )

            # Box plot
            fig.add_trace(go.Box(y=plot_data.values, name="Box"), row=2, col=1)

            # Time series sample (first 1000 points)
            sample_size = min(1000, len(data))
            fig.add_trace(
                go.Scatter(
                    x=np.arange(sample_size),
                    y=data.iloc[:sample_size].values,
                    mode="lines",
                    line=dict(width=0.5),
                ),
                row=2,
                col=2,
            )

            fig.update_layout(
                title=f"Distribution - {self.friendly_name}",
                height=600,
                showlegend=False,
            )

            fig.write_html(PLOTS_FOLDER / f"{self.dimension_name}_distribution.html")

            print("Distribution plots created")
        except Exception as e:
            print(f"Error creating distribution plots: {e}")

    def _fit_sarima_optimized(self, data: pd.Series) -> Dict:
        """Fit SARIMA model with memory optimization"""
        print(f"Fitting SARIMA model...")

        try:
            # Downsample for initial parameter search if data is large
            if len(data) > 5000:
                sample_data = data[::2]  # Take every other point
                print(
                    f"Using downsampled data ({len(sample_data)} points) for parameter search"
                )
            else:
                sample_data = data

            # Use more conservative parameters for memory
            model = pm.auto_arima(
                sample_data.values,
                seasonal=True,
                m=24,  # Daily seasonality
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=2,
                max_q=2,  # Reduced from 3
                max_P=1,
                max_Q=1,  # Reduced from 2
                max_order=5,  # Reduced from 10
                n_jobs=1,  # Single thread per model
                trace=False,
                maxiter=50,  # Limit iterations
            )

            # Get parameters
            order = model.order
            seasonal_order = model.seasonal_order

            # If we downsampled, refit on full data with found parameters
            if len(data) > 5000:
                print(f"Refitting on full data with SARIMA{order}x{seasonal_order}")
                from statsmodels.tsa.statespace.sarimax import SARIMAX

                final_model = SARIMAX(
                    data.values, order=order, seasonal_order=seasonal_order
                ).fit(disp=False, maxiter=100)

                aic = final_model.aic
                bic = final_model.bic
            else:
                final_model = model
                aic = model.aic()
                bic = model.bic()

            # Save model
            model_path = MODELS_FOLDER / f"{self.dimension_name}_sarima.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"order": order, "seasonal_order": seasonal_order}, f)

            sarima_dict = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": float(aic),
                "bic": float(bic),
            }

            # Save parameters
            pd.DataFrame([sarima_dict]).to_csv(
                TABLES_FOLDER / f"{self.dimension_name}_sarima_params.csv", index=False
            )

            print(f"SARIMA{order}x{seasonal_order} - AIC: {aic:.2f}, BIC: {bic:.2f}")

            # Clear model from memory
            del model
            if len(data) > 5000:
                del final_model

            return sarima_dict

        except Exception as e:
            print(f"Error fitting SARIMA: {e}")
            return {"error": str(e)}


def process_dimension_parallel(dimension: str, df: pd.DataFrame) -> Tuple[str, Dict]:
    """Process a single dimension (for parallel execution)"""
    if dimension not in df.columns:
        return dimension, {"error": f"Dimension not found"}

    try:
        data = df[dimension]
        friendly_name = DIMENSION_NAMES.get(dimension, dimension)

        analyzer = OptimizedSARIMAAnalyzer(dimension, friendly_name)
        results = analyzer.analyze_dimension(data)

        # Force cleanup
        del analyzer
        clean_memory()

        return dimension, results

    except Exception as e:
        print(f"Error processing {dimension}: {e}")
        return dimension, {"error": str(e)}


def process_batch(dimensions: List[str], df: pd.DataFrame, batch_num: int) -> Dict:
    """Process a batch of dimensions"""
    print(f"\nProcessing batch {batch_num} with {len(dimensions)} dimensions")
    results = {}

    for dim in dimensions:
        dim_name, dim_results = process_dimension_parallel(dim, df)
        results[dim_name] = dim_results
        clean_memory()

    return results


def create_summary_report(all_results: Dict):
    """Create summary report from all results"""
    summary_data = []

    for dim, results in all_results.items():
        if "error" not in results:
            summary_data.append(
                {
                    "Dimension": DIMENSION_NAMES.get(dim, dim),
                    "Mean": results.get("statistics", {}).get("mean", np.nan),
                    "Std Dev": results.get("statistics", {}).get("std_dev", np.nan),
                    "Skewness": results.get("statistics", {}).get("skewness", np.nan),
                    "Kurtosis": results.get("statistics", {}).get("kurtosis", np.nan),
                    "Stationary": results.get("adf", {}).get("is_stationary", False),
                    "ADF p-value": results.get("adf", {}).get("p_value", np.nan),
                    "Multimodal": results.get("multimodal_info", {}).get(
                        "is_multimodal", False
                    ),
                    "N Modes": results.get("multimodal_info", {}).get("n_modes", 1),
                    "Low Freq": results.get("frequency_components", {}).get(
                        "low_frequency_count", 0
                    ),
                    "High Freq": results.get("frequency_components", {}).get(
                        "high_frequency_count", 0
                    ),
                    "SARIMA Order": str(
                        results.get("sarima_params", {}).get("order", "N/A")
                    ),
                    "AIC": results.get("sarima_params", {}).get("aic", np.nan),
                    "BIC": results.get("sarima_params", {}).get("bic", np.nan),
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(TABLES_FOLDER / "summary.csv", index=False)
        summary_df.to_excel(TABLES_FOLDER / "summary.xlsx", index=False)
        print("\nSummary report saved")
        return summary_df

    return None


def main():
    """Main execution with parallel processing and memory management"""
    print("=" * 60)
    print("OPTIMIZED SARIMA TIME SERIES ANALYSIS")
    print("=" * 60)
    print(f"CPU cores available: {mp.cpu_count()}")
    print(f"Initial memory: {get_memory_usage():.1f} MB")

    # Import data loader
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from utils.load_df import load_df

    print("Loading data...")
    df = load_df()

    if df.empty:
        print("Error: Empty dataframe")
        return
    df = df[["datetime"] + DIMS]

    print(f"Data shape: {df.shape}")

    # Determine processing strategy based on available memory
    available_memory = psutil.virtual_memory().available / 1024 / 1024 / 1024  # GB
    print(f"Available memory: {available_memory:.1f} GB")

    # Process dimensions
    all_results = {}

    # Determine batch size based on available resources
    if available_memory > 8:
        batch_size = 4  # Process 4 dimensions at a time
        n_workers = min(4, mp.cpu_count())
    elif available_memory > 4:
        batch_size = 2  # Process 2 dimensions at a time
        n_workers = min(2, mp.cpu_count())
    else:
        batch_size = 1  # Process 1 dimension at a time
        n_workers = 1

    print(f"Processing strategy: batch_size={batch_size}, workers={n_workers}")

    # Filter available dimensions
    available_dims = [d for d in DIMS if d in df.columns]
    print(f"Found {len(available_dims)} dimensions to analyze")

    # Process in batches
    for i in range(0, len(available_dims), batch_size):
        batch = available_dims[i : i + batch_size]
        batch_num = i // batch_size + 1

        print(f"\n{'='*60}")
        print(f"BATCH {batch_num}/{(len(available_dims)-1)//batch_size + 1}")
        print(f"{'='*60}")

        if n_workers > 1 and len(batch) > 1:
            # Parallel processing within batch
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(process_dimension_parallel, dim, df): dim
                    for dim in batch
                }

                for future in futures:
                    dim = futures[future]
                    try:
                        dim_name, results = future.result(timeout=300)  # 5 min timeout
                        all_results[dim_name] = results
                    except Exception as e:
                        print(f"Error processing {dim}: {e}")
                        all_results[dim] = {"error": str(e)}
        else:
            # Sequential processing
            batch_results = process_batch(batch, df, batch_num)
            all_results.update(batch_results)

        # Clean memory after each batch
        clean_memory()
        print(f"Memory after batch: {get_memory_usage():.1f} MB")

    # Create summary report
    print("\n" + "=" * 60)
    print("Creating summary report...")
    summary_df = create_summary_report(all_results)

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    successful = sum(1 for r in all_results.values() if "error" not in r)
    print(f"Successfully analyzed: {successful}/{len(available_dims)} dimensions")
    print(f"Final memory usage: {get_memory_usage():.1f} MB")
    print(f"\nResults saved to: {BASE_OUTPUT_FOLDER}")
    print(f"  - Models: {MODELS_FOLDER}")
    print(f"  - Plots: {PLOTS_FOLDER}")
    print(f"  - Tables: {TABLES_FOLDER}")


if __name__ == "__main__":
    # Set multiprocessing start method
    if sys.platform != "win32":
        mp.set_start_method("spawn", force=True)

    main()
