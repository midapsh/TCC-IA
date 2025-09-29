# Create optimized SARIMA analysis script with memory management and multi-threading
"""
Optimized SARIMA Time Series Analysis Script
- Memory efficient with garbage collection
- Multi-threaded GaussianMixture computations
- Batch processing to reduce memory footprint
"""

import gc
import os
import sys
import pickle
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from os import makedirs
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to save memory
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pmdarima as pm
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.mixture import GaussianMixture
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, adfuller, pacf

warnings.filterwarnings("ignore")

# Configuration
DIMS = [
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

DIMENSION_NAMES = {
    "precipitacao_total_horario": "Hourly Precipitation",
    "pressao_atmosferica_ao_nivel_da_estacao_horaria": "Atmospheric Pressure (Station Level)",
    "pressao_atmosferica_max_na_hora_ant": "Max Atmospheric Pressure (Previous Hour)",
    "pressao_atmosferica_min_na_hora_ant": "Min Atmospheric Pressure (Previous Hour)",
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
BASE_OUTPUT_FOLDER = Path("sarima_analysis")
MODELS_FOLDER = BASE_OUTPUT_FOLDER / "models"
PLOTS_FOLDER = BASE_OUTPUT_FOLDER / "plots"
TABLES_FOLDER = BASE_OUTPUT_FOLDER / "tables"
MULTIMODAL_FOLDER = BASE_OUTPUT_FOLDER / "multimodal_analysis"

# Create folders
for folder in [MODELS_FOLDER, PLOTS_FOLDER, TABLES_FOLDER, MULTIMODAL_FOLDER]:
    makedirs(folder, exist_ok=True)

# Memory management settings
MAX_WORKERS = 4  # Number of threads for parallel processing
CHUNK_SIZE = 10000  # Process data in chunks to reduce memory


def fit_gmm_component(data_array, n_components):
    """Fit a single GMM model - used for parallel processing"""
    try:
        gmm = GaussianMixture(n_components=n_components, random_state=42, max_iter=100)
        gmm.fit(data_array)
        return n_components, gmm.bic(data_array), gmm.aic(data_array)
    except Exception as e:
        print(f"Error fitting GMM with {n_components} components: {e}")
        return n_components, float("inf"), float("inf")


class OptimizedSARIMAAnalyzer:
    """Memory-optimized SARIMA analyzer with multi-threading"""

    def __init__(self, data, dimension_name, friendly_name):
        self.data = data
        self.dimension_name = dimension_name
        self.friendly_name = friendly_name
        self.model = None
        self.results = {}
        self.multimodal_info = {}
        self.clean_data_series = None

    def __del__(self):
        """Cleanup when object is destroyed"""
        self.cleanup_memory()

    def cleanup_memory(self):
        """Force garbage collection and clear unnecessary data"""
        # Clear large objects
        if hasattr(self, "data"):
            del self.data
        if hasattr(self, "model") and self.model is not None:
            del self.model
        # Force garbage collection
        gc.collect()

    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Remove NaN values
        self.clean_data_series = self.data.dropna()

        # Clear original data to save memory
        del self.data
        gc.collect()

        # Check for sufficient data
        if len(self.clean_data_series) < 100:
            raise ValueError(
                f"Insufficient data for {self.dimension_name}: {len(self.clean_data_series)} points"
            )

        # If data is too large, downsample for some analyses
        if len(self.clean_data_series) > 50000:
            print(
                f"Large dataset detected ({len(self.clean_data_series)} points). Will use sampling for some analyses."
            )

        print(f"Data cleaned: {len(self.clean_data_series)} valid points")
        return self.clean_data_series

    def perform_adf_test(self):
        """Perform Augmented Dickey-Fuller test"""
        # Use a subset if data is too large
        test_data = self.clean_data_series
        if len(test_data) > 10000:
            test_data = self.clean_data_series.iloc[:10000]

        adf_result = adfuller(test_data, autolag="AIC")

        self.results["adf"] = {
            "statistic": float(adf_result[0]),
            "p_value": float(adf_result[1]),
            "used_lag": int(adf_result[2]),
            "n_obs": int(adf_result[3]),
            "critical_values": adf_result[4],
            "is_stationary": adf_result[1] < 0.05,
        }

        # Save results
        adf_df = pd.DataFrame(
            {
                "Metric": [
                    "ADF Statistic",
                    "p-value",
                    "Used Lag",
                    "Number of Observations",
                    "Critical Value (1%)",
                    "Critical Value (5%)",
                    "Critical Value (10%)",
                    "Is Stationary (p<0.05)",
                ],
                "Value": [
                    self.results["adf"]["statistic"],
                    self.results["adf"]["p_value"],
                    self.results["adf"]["used_lag"],
                    self.results["adf"]["n_obs"],
                    self.results["adf"]["critical_values"]["1%"],
                    self.results["adf"]["critical_values"]["5%"],
                    self.results["adf"]["critical_values"]["10%"],
                    self.results["adf"]["is_stationary"],
                ],
            }
        )
        adf_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_adf_test.csv", index=False
        )

        print(
            f"ADF Test - Stationary: {self.results['adf']['is_stationary']} (p-value: {self.results['adf']['p_value']:.4f})"
        )

        # Clean up
        del adf_df
        gc.collect()

        return self.results["adf"]

    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        # Convert to numpy array for efficiency
        data_array = np.array(self.clean_data_series)

        self.results["statistics"] = {
            "mean": float(np.mean(data_array)),
            "median": float(np.median(data_array)),
            "mode": float(stats.mode(data_array, keepdims=True)[0][0]),
            "std_dev": float(np.std(data_array)),
            "variance": float(np.var(data_array)),
            "skewness": float(stats.skew(data_array)),
            "kurtosis": float(stats.kurtosis(data_array)),
            "min": float(np.min(data_array)),
            "max": float(np.max(data_array)),
            "q25": float(np.percentile(data_array, 25)),
            "q75": float(np.percentile(data_array, 75)),
            "iqr": float(np.percentile(data_array, 75) - np.percentile(data_array, 25)),
        }

        # Save statistics
        stats_df = pd.DataFrame(self.results["statistics"], index=[0]).T
        stats_df.columns = ["Value"]
        stats_df.to_csv(TABLES_FOLDER / f"{self.dimension_name}_statistics.csv")

        print(
            f"Statistics - Skewness: {self.results['statistics']['skewness']:.3f}, Kurtosis: {self.results['statistics']['kurtosis']:.3f}"
        )

        # Clean up
        del data_array, stats_df
        gc.collect()

        return self.results["statistics"]

    def detect_frequency_components(self):
        """Detect low and high frequency components with memory optimization"""
        # Use subset for FFT if data is too large
        fft_data = self.clean_data_series
        if len(fft_data) > 20000:
            fft_data = self.clean_data_series.iloc[:20000]

        # FFT analysis
        fft_vals = np.fft.fft(fft_data)
        freqs = np.fft.fftfreq(len(fft_data))
        power = np.abs(fft_vals) ** 2

        # Find significant frequencies
        positive_freqs = freqs > 0
        if np.sum(positive_freqs) > 0:
            threshold = np.percentile(power[positive_freqs], 90)
            significant_mask = power > threshold
            significant_freqs = freqs[significant_mask]

            # Classify frequencies
            low_freq_threshold = 1 / 168  # Weekly
            high_freq_threshold = 1 / 12  # Sub-daily

            low_freqs = significant_freqs[
                np.abs(significant_freqs) < low_freq_threshold
            ]
            high_freqs = significant_freqs[
                np.abs(significant_freqs) > high_freq_threshold
            ]
            mid_freqs = significant_freqs[
                (np.abs(significant_freqs) >= low_freq_threshold)
                & (np.abs(significant_freqs) <= high_freq_threshold)
            ]

            # Find dominant frequency
            dominant_idx = (
                np.argmax(power[1 : len(power) // 2]) + 1 if len(freqs) > 1 else 0
            )
            dominant_freq = freqs[dominant_idx] if dominant_idx > 0 else 0
        else:
            low_freqs = high_freqs = mid_freqs = []
            dominant_freq = 0

        self.results["frequency_components"] = {
            "low_frequency_count": len(low_freqs),
            "high_frequency_count": len(high_freqs),
            "mid_frequency_count": len(mid_freqs),
            "dominant_frequency": float(dominant_freq),
            "low_freq_periods_hours": [1 / f for f in low_freqs if f != 0][
                :10
            ],  # Limit to 10
            "high_freq_periods_hours": [1 / f for f in high_freqs if f != 0][
                :10
            ],  # Limit to 10
        }

        # Save frequency analysis
        freq_df = pd.DataFrame(
            {
                "Component": [
                    "Low Frequency",
                    "Mid Frequency",
                    "High Frequency",
                    "Dominant Frequency",
                ],
                "Count": [
                    self.results["frequency_components"]["low_frequency_count"],
                    self.results["frequency_components"]["mid_frequency_count"],
                    self.results["frequency_components"]["high_frequency_count"],
                    1,
                ],
                "Description": [
                    "Periods > 168 hours (weekly+)",
                    "Periods 12-168 hours (sub-weekly)",
                    "Periods < 12 hours (sub-daily)",
                    f"Frequency: {dominant_freq:.6f}",
                ],
            }
        )
        freq_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_frequency_components.csv",
            index=False,
        )

        print(
            f"Frequency components - Low: {len(low_freqs)}, Mid: {len(mid_freqs)}, High: {len(high_freqs)}"
        )

        # Clean up
        del fft_vals, freqs, power, freq_df
        gc.collect()

        return self.results["frequency_components"]

    def detect_multimodal_distribution(self):
        """Detect multimodal distributions using parallel processing"""
        # Sample data if too large
        sample_data = self.clean_data_series
        if len(sample_data) > 10000:
            sample_data = self.clean_data_series.sample(n=10000, random_state=42)

        data_array = np.array(sample_data).reshape(-1, 1)

        # Parallel GMM fitting
        n_components_range = range(1, 6)

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit all GMM fitting tasks
            futures = {
                executor.submit(fit_gmm_component, data_array, n): n
                for n in n_components_range
            }

            results = {}
            for future in as_completed(futures):
                n_comp, bic, aic = future.result()
                results[n_comp] = {"bic": bic, "aic": aic}

        # Find optimal number of components
        bic_scores = [results[n]["bic"] for n in sorted(results.keys())]
        optimal_components = np.argmin(bic_scores) + 1

        if optimal_components > 1:
            print(f"Multimodal distribution detected with {optimal_components} modes")

            # Fit final GMM with optimal components
            gmm = GaussianMixture(
                n_components=optimal_components, random_state=42, max_iter=100
            )
            gmm.fit(data_array)

            # Get predictions
            labels = gmm.predict(data_array)

            # Find transition points
            transition_points = []
            for i in range(1, min(len(labels), 1000)):  # Limit transition point search
                if labels[i] != labels[i - 1]:
                    transition_points.append(i)

            self.multimodal_info = {
                "is_multimodal": True,
                "n_modes": int(optimal_components),
                "transition_points": transition_points[:20],  # Limit stored transitions
                "mode_means": gmm.means_.flatten().tolist(),
                "mode_weights": gmm.weights_.tolist(),
            }

            # Save limited multimodal segments
            if len(transition_points) > 0:
                self._save_multimodal_segments_limited(
                    labels, transition_points[:5]
                )  # Only first 5
        else:
            print("Unimodal distribution detected")
            self.multimodal_info = {"is_multimodal": False, "n_modes": 1}

        # Save multimodal analysis
        multimodal_df = pd.DataFrame(
            {
                "Property": [
                    "Is Multimodal",
                    "Number of Modes",
                    "Number of Transitions",
                ],
                "Value": [
                    self.multimodal_info["is_multimodal"],
                    self.multimodal_info.get("n_modes", 1),
                    len(self.multimodal_info.get("transition_points", [])),
                ],
            }
        )
        multimodal_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_multimodal_analysis.csv",
            index=False,
        )

        self.results["multimodal_info"] = self.multimodal_info

        # Clean up
        del data_array, multimodal_df
        if optimal_components > 1:
            del gmm, labels
        gc.collect()

        return self.multimodal_info

    def _save_multimodal_segments_limited(self, labels, transition_points):
        """Save limited multimodal segments to conserve memory"""
        dim_folder = MULTIMODAL_FOLDER / self.dimension_name
        makedirs(dim_folder, exist_ok=True)

        # Only process first few segments
        segments = [0] + transition_points[:5] + [len(labels)]

        for i in range(min(len(segments) - 1, 3)):  # Max 3 segments
            start_idx = segments[i]
            end_idx = segments[i + 1]

            # Use subset of segment data
            segment_size = min(end_idx - start_idx, 1000)
            segment_data = self.clean_data_series.iloc[
                start_idx : start_idx + segment_size
            ]

            # Calculate and save statistics
            segment_stats = {
                "segment": i + 1,
                "start_index": start_idx,
                "end_index": start_idx + segment_size,
                "length": segment_size,
                "mean": float(np.mean(segment_data)),
                "std": float(np.std(segment_data)),
                "skewness": float(stats.skew(segment_data)),
                "kurtosis": float(stats.kurtosis(segment_data)),
            }

            pd.DataFrame(segment_stats, index=[0]).to_csv(
                dim_folder / f"segment_{i+1}_statistics.csv", index=False
            )

            # Clean up after each segment
            del segment_data
            gc.collect()

    def plot_acf_pacf(self):
        """Create ACF and PACF plots with memory optimization"""
        # Use subset for large datasets
        plot_data = self.clean_data_series
        if len(plot_data) > 5000:
            plot_data = self.clean_data_series.iloc[:5000]

        fig, axes = plt.subplots(2, 1, figsize=(10, 6))

        # ACF plot
        plot_acf(plot_data, lags=min(40, len(plot_data) // 4), ax=axes[0], alpha=0.05)
        axes[0].set_title(f"ACF - {self.friendly_name}")

        # PACF plot
        plot_pacf(plot_data, lags=min(40, len(plot_data) // 4), ax=axes[1], alpha=0.05)
        axes[1].set_title(f"PACF - {self.friendly_name}")

        plt.tight_layout()
        plt.savefig(
            PLOTS_FOLDER / f"{self.dimension_name}_acf_pacf.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close("all")

        # Clean up
        del fig, axes
        gc.collect()

        print("ACF and PACF plots created")

    def plot_distribution_analysis(self):
        """Create distribution plots with memory optimization"""
        # Sample data if too large
        plot_data = self.clean_data_series
        if len(plot_data) > 5000:
            plot_data = self.clean_data_series.sample(n=5000, random_state=42)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Histogram", "Q-Q Plot", "Box Plot", "Time Series Sample"),
        )

        # Histogram
        fig.add_trace(
            go.Histogram(x=plot_data.values, nbinsx=30, name="Histogram"), row=1, col=1
        )

        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, min(len(plot_data), 100))
        )
        sample_quantiles = np.percentile(
            plot_data.values, np.linspace(1, 99, min(len(plot_data), 100))
        )

        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Q-Q",
                marker=dict(size=3),
            ),
            row=1,
            col=2,
        )

        # Box plot
        fig.add_trace(go.Box(y=plot_data.values, name="Box Plot"), row=2, col=1)

        # Time series sample
        sample_size = min(1000, len(self.clean_data_series))
        fig.add_trace(
            go.Scatter(
                x=np.arange(sample_size),
                y=self.clean_data_series.iloc[:sample_size].values,
                mode="lines",
                name="Time Series",
                line=dict(width=0.5),
            ),
            row=2,
            col=2,
        )

        # Add statistics annotation
        stats_text = f"""
        Mean: {self.results['statistics']['mean']:.3f}
        Median: {self.results['statistics']['median']:.3f}
        Std Dev: {self.results['statistics']['std_dev']:.3f}
        Skewness: {self.results['statistics']['skewness']:.3f}
        Kurtosis: {self.results['statistics']['kurtosis']:.3f}
        """

        fig.add_annotation(
            text=stats_text,
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            showarrow=False,
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=10),
        )

        fig.update_layout(
            title=f"Distribution Analysis - {self.friendly_name}",
            height=900,
            showlegend=False,
        )

        fig.write_html(PLOTS_FOLDER / f"{self.dimension_name}_distribution.html")

        # Clean up
        del fig
        gc.collect()

        print("Distribution analysis plots created")

    def fit_sarima_model(self, seasonal_period=24):
        """Fit SARIMA model with memory optimization"""
        print(f"Fitting SARIMA model for {self.friendly_name}...")

        # Use subset for very large datasets
        model_data = self.clean_data_series
        if len(model_data) > 10000:
            model_data = self.clean_data_series.iloc[-10000:]  # Use most recent data
            print(f"Using last 10000 points for SARIMA fitting")

        try:
            # Simplified auto_arima for memory efficiency
            self.model = pm.auto_arima(
                model_data,
                seasonal=True,
                m=seasonal_period,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3,
                max_q=3,
                max_P=2,
                max_Q=2,
                max_order=10,
                trace=False,
            )

            # Get parameters
            order = self.model.order
            seasonal_order = self.model.seasonal_order

            self.results["sarima_params"] = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": float(self.model.aic()),
                "bic": float(self.model.bic()),
            }

            # Save model
            model_path = MODELS_FOLDER / f"{self.dimension_name}_sarima.pkl"
            with open(model_path, "wb") as f:
                pickle.dump({"order": order, "seasonal_order": seasonal_order}, f)

            # Save parameters
            params_df = pd.DataFrame(
                {
                    "Parameter": ["p", "d", "q", "P", "D", "Q", "s", "AIC", "BIC"],
                    "Value": [
                        order[0],
                        order[1],
                        order[2],
                        seasonal_order[0],
                        seasonal_order[1],
                        seasonal_order[2],
                        seasonal_order[3],
                        self.model.aic(),
                        self.model.bic(),
                    ],
                }
            )
            params_df.to_csv(
                TABLES_FOLDER / f"{self.dimension_name}_sarima_params.csv", index=False
            )

            print(f"SARIMA{order}x{seasonal_order} fitted")
            print(f"AIC: {self.model.aic():.2f}, BIC: {self.model.bic():.2f}")

            # Simple residual analysis
            self._analyze_residuals_simple()

            # Clean up model after saving parameters
            del self.model
            self.model = None
            gc.collect()

        except Exception as e:
            print(f"Error fitting SARIMA: {e}")
            self.results["sarima_params"] = {"error": str(e)}

    def _analyze_residuals_simple(self):
        """Simplified residual analysis"""
        if self.model is None:
            return

        try:
            # residuals = self.model.resid()  # [:1000]  # Limit residuals
            residuals = self.model.resid()[:1000]  # Limit residuals

            # Basic Ljung-Box test
            lb_test = acorr_ljungbox(
                residuals, lags=min(10, len(residuals) // 4), return_df=True
            )
            lb_test.to_csv(TABLES_FOLDER / f"{self.dimension_name}_ljungbox.csv")

            print("Residual analysis completed")

            # Clean up
            del residuals, lb_test
            gc.collect()

        except Exception as e:
            print(f"Error in residual analysis: {e}")

    def create_forecast_plot(self, n_periods=168):
        """Create simple forecast plot"""
        # Skip if model wasn't saved properly
        print(f"Forecast plotting skipped (model cleared for memory)")

    def run_complete_analysis(self):
        """Run optimized analysis pipeline"""
        print(f"\n{'='*60}")
        print(f"Analyzing: {self.friendly_name}")
        print(f"{'='*60}")

        try:
            # Core analyses
            self.clean_data()
            self.calculate_statistics()
            self.perform_adf_test()
            self.detect_frequency_components()

            # Memory-intensive analyses
            self.detect_multimodal_distribution()
            self.plot_acf_pacf()
            self.plot_distribution_analysis()
            self.fit_sarima_model()

            print(f"Analysis completed for {self.friendly_name}")

            # Final cleanup
            self.cleanup_memory()

            return self.results

        except Exception as e:
            print(f"Error in analysis: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}


def create_summary_report(all_results):
    """Create summary report"""
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
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(TABLES_FOLDER / "summary.csv", index=False)
        summary_df.to_excel(TABLES_FOLDER / "summary.xlsx", index=False)
        print("\nSummary report saved")
        return summary_df

    return None


def process_dimension_batch(df, dimensions_batch):
    """Process a batch of dimensions"""
    results = {}

    for dimension in dimensions_batch:
        if dimension not in df.columns:
            print(f"Warning: '{dimension}' not found")
            continue

        try:
            data = df[dimension]
            friendly_name = DIMENSION_NAMES.get(dimension, dimension)

            # Create analyzer
            analyzer = OptimizedSARIMAAnalyzer(data, dimension, friendly_name)

            # Run analysis
            results[dimension] = analyzer.run_complete_analysis()

            # Force cleanup
            del analyzer
            gc.collect()

        except Exception as e:
            print(f"Error analyzing {dimension}: {e}")
            results[dimension] = {"error": str(e)}

    return results


def main():
    """Main execution with batch processing"""
    print("Starting Optimized SARIMA Analysis")
    print("=" * 60)

    # Import load_df
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.load_df import load_df

    # Load data
    print("Loading data...")
    df = load_df()

    if df.empty:
        print("Error: Empty dataframe")
        return

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Process dimensions in batches to manage memory
    batch_size = 3
    all_results = {}

    for i in range(0, len(DIMS), batch_size):
        batch = DIMS[i : i + batch_size]
        print(f"\nProcessing batch {i//batch_size + 1}/{(len(DIMS)-1)//batch_size + 1}")

        batch_results = process_dimension_batch(df, batch)
        all_results.update(batch_results)

        # Force garbage collection between batches
        gc.collect()

    # Create summary
    print("\n" + "=" * 60)
    print("Creating summary report...")
    summary_df = create_summary_report(all_results)

    # Final summary
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    successful = sum(1 for r in all_results.values() if "error" not in r)
    print(f"Successfully analyzed: {successful}/{len(DIMS)} dimensions")
    print(f"Results saved to: {BASE_OUTPUT_FOLDER}")

    # Final cleanup
    del df, all_results
    gc.collect()


if __name__ == "__main__":
    main()
