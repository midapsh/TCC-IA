#!/usr/bin/env python3
"""
SARIMA Time Series Analysis Script
Based on frequency_analysis.py structure
Performs comprehensive time series analysis including:
- SARIMA model fitting
- Dickey-Fuller tests
- ACF/PACF analysis
- Distribution analysis (skewness, kurtosis, etc.)
- Multimodal distribution detection
- Frequency component analysis
"""

from os import makedirs
from pathlib import Path
import pickle
import warnings

from plotly.subplots import make_subplots
from scipy import stats
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

# Statistical and time series imports
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
import pmdarima as pm

warnings.filterwarnings("ignore")

# Configuration based on frequency_analysis.py
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
BASE_OUTPUT_FOLDER = Path(
    "/home/dolores/Documents/matheus-ferreira/TCC-IA/models/sarima_analysis"
)
MODELS_FOLDER = BASE_OUTPUT_FOLDER / "models"
PLOTS_FOLDER = BASE_OUTPUT_FOLDER / "plots"
TABLES_FOLDER = BASE_OUTPUT_FOLDER / "tables"
MULTIMODAL_FOLDER = BASE_OUTPUT_FOLDER / "multimodal_analysis"

# Create all necessary folders
for folder in [MODELS_FOLDER, PLOTS_FOLDER, TABLES_FOLDER, MULTIMODAL_FOLDER]:
    makedirs(folder, exist_ok=True)


class SARIMAAnalyzer:
    """Complete SARIMA analysis for time series data"""

    def __init__(self, data, dimension_name, friendly_name):
        self.data = data
        self.dimension_name = dimension_name
        self.friendly_name = friendly_name
        self.model = None
        self.results = {}
        self.multimodal_info = {}

    def clean_data(self):
        """Clean and prepare data for analysis"""
        # Remove NaN values
        self.clean_data_series = self.data.dropna()

        # Check for sufficient data
        if len(self.clean_data_series) < 100:
            raise ValueError(
                f"Insufficient data for {self.dimension_name}: {len(self.clean_data_series)} points"
            )

        print(
            f"Data cleaned: {len(self.clean_data_series)} valid points from {len(self.data)} total"
        )
        return self.clean_data_series

    def perform_adf_test(self):
        """Perform Augmented Dickey-Fuller test"""
        adf_result = adfuller(self.clean_data_series, autolag="AIC")

        self.results["adf"] = {
            "statistic": adf_result[0],
            "p_value": adf_result[1],
            "used_lag": adf_result[2],
            "n_obs": adf_result[3],
            "critical_values": adf_result[4] if len(adf_result) == 5 else None,
            "is_stationary": adf_result[1] < 0.05,
        }

        # Create ADF results table
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
                    self.results.get("adf", {}).get("statistic"),
                    self.results.get("adf", {}).get("p_value")
                    or self.results.get("adf", {}).get("pvalue"),
                    self.results.get("adf", {}).get("used_lag")
                    or self.results.get("adf", {}).get("lags"),
                    self.results.get("adf", {}).get("n_obs")
                    or self.results.get("adf", {}).get("nobs"),
                    (self.results.get("adf", {}).get("critical_values") or {}).get(
                        "1%"
                    ),
                    (self.results.get("adf", {}).get("critical_values") or {}).get(
                        "5%"
                    ),
                    (self.results.get("adf", {}).get("critical_values") or {}).get(
                        "10%"
                    ),
                    self.results.get("adf", {}).get("is_stationary", False),
                ],
            }
        )

        # Save ADF results
        _ = adf_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_adf_test.csv", index=False
        )

        print(
            f"ADF Test - Stationary: {self.results['adf']['is_stationary']} (p-value: {self.results['adf']['p_value']:.4f})"
        )
        return self.results["adf"]

    def calculate_statistics(self):
        """Calculate comprehensive statistics"""
        self.results["statistics"] = {
            "mean": np.mean(self.clean_data_series),
            "median": np.median(self.clean_data_series),
            "mode": stats.mode(self.clean_data_series, keepdims=True)[0][0],
            "std_dev": np.std(self.clean_data_series),
            "variance": np.var(self.clean_data_series),
            "skewness": stats.skew(self.clean_data_series),
            "kurtosis": stats.kurtosis(self.clean_data_series),
            "min": np.min(self.clean_data_series),
            "max": np.max(self.clean_data_series),
            "q25": np.percentile(self.clean_data_series, 25),
            "q75": np.percentile(self.clean_data_series, 75),
            "iqr": np.percentile(self.clean_data_series, 75)
            - np.percentile(self.clean_data_series, 25),
        }

        # Create statistics table
        stats_df = pd.DataFrame(self.results["statistics"], index=[0]).T
        stats_df.columns = ["Value"]
        _ = stats_df.to_csv(TABLES_FOLDER / f"{self.dimension_name}_statistics.csv")

        print(
            f"Statistics calculated - Skewness: {self.results['statistics']['skewness']:.3f}, Kurtosis: {self.results['statistics']['kurtosis']:.3f}"
        )
        return self.results["statistics"]

    def detect_frequency_components(self):
        """Detect low and high frequency components"""
        # Use FFT to identify frequency components
        fft_vals = np.fft.fft(self.clean_data_series)
        freqs = np.fft.fftfreq(len(self.clean_data_series))

        # Get power spectrum
        power = np.abs(fft_vals) ** 2

        # Identify significant frequencies (top 10% power)
        threshold = np.percentile(power[freqs > 0], 90)
        significant_freqs = freqs[power > threshold]

        # Classify frequencies
        low_freq_threshold = 1 / 168  # Weekly or longer periods (168 hours = 1 week)
        high_freq_threshold = 1 / 12  # Sub-daily periods (12 hours)

        low_freqs = significant_freqs[np.abs(significant_freqs) < low_freq_threshold]
        high_freqs = significant_freqs[np.abs(significant_freqs) > high_freq_threshold]
        mid_freqs = significant_freqs[
            (np.abs(significant_freqs) >= low_freq_threshold)
            & (np.abs(significant_freqs) <= high_freq_threshold)
        ]

        self.results["frequency_components"] = {
            "low_frequency_count": len(low_freqs),
            "high_frequency_count": len(high_freqs),
            "mid_frequency_count": len(mid_freqs),
            "dominant_frequency": (
                freqs[np.argmax(power[1 : len(power) // 2]) + 1]
                if len(freqs) > 1
                else 0
            ),
            "low_freq_periods_hours": [1 / f for f in low_freqs if f != 0],
            "high_freq_periods_hours": [1 / f for f in high_freqs if f != 0],
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
                    self.results.get("frequency_components", {}).get(
                        "low_frequency_count"
                    ),
                    self.results.get("frequency_components", {}).get(
                        "mid_frequency_count"
                    ),
                    self.results.get("frequency_components", {}).get(
                        "high_frequency_count"
                    ),
                    1,
                ],
                "Description": [
                    f"Periods > 168 hours (weekly+)",
                    f"Periods 12-168 hours (sub-weekly)",
                    f"Periods < 12 hours (sub-daily)",
                    f"Frequency: {self.results.get("frequency_components", {}).get('dominant_frequency'):.6f}",
                ],
            }
        )
        _ = freq_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_frequency_components.csv",
            index=False,
        )

        print(
            f"Frequency components - Low: {len(low_freqs)}, Mid: {len(mid_freqs)}, High: {len(high_freqs)}"
        )
        return self.results["frequency_components"]

    def detect_multimodal_distribution(self):
        """Detect and analyze multimodal distributions"""
        data_array = np.array(self.clean_data_series).reshape(-1, 1)

        # Try different numbers of components
        n_components_range = range(1, 6)
        bic_scores = []
        aic_scores = []

        for n_components in n_components_range:
            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(data_array)
            bic_scores.append(gmm.bic(data_array))
            aic_scores.append(gmm.aic(data_array))

        # Find optimal number of components
        optimal_components = np.argmin(bic_scores) + 1

        if optimal_components > 1:
            print(f"Multimodal distribution detected with {optimal_components} modes")

            # Fit GMM with optimal components
            gmm = GaussianMixture(n_components=optimal_components, random_state=42)
            gmm.fit(data_array)

            # Get predictions
            labels = gmm.predict(data_array)

            # Find transition points
            transition_points = []
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    transition_points.append(i)

            self.multimodal_info = {
                "is_multimodal": True,
                "n_modes": optimal_components,
                "transition_points": transition_points,
                "mode_means": gmm.means_.flatten().tolist(),
                "mode_weights": gmm.weights_.tolist(),
                "labels": labels,
            }

            # Save multimodal segments
            if len(transition_points) > 0:
                self._save_multimodal_segments(labels, transition_points)
        else:
            print("Unimodal distribution detected")
            self.multimodal_info = {"is_multimodal": False, "n_modes": 1}

        # Save multimodal analysis results
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
        _ = multimodal_df.to_csv(
            TABLES_FOLDER / f"{self.dimension_name}_multimodal_analysis.csv",
            index=False,
        )

        self.results["multimodal_info"] = self.multimodal_info
        return self.multimodal_info

    def _save_multimodal_segments(self, labels, transition_points):
        """Save data segments before and after multimodal transitions"""
        dim_folder = MULTIMODAL_FOLDER / self.dimension_name
        makedirs(dim_folder, exist_ok=True)

        # Add start and end points
        segments = [0] + transition_points + [len(labels)]

        for i in range(len(segments) - 1):
            start_idx = segments[i]
            end_idx = segments[i + 1]

            segment_data = self.clean_data_series.iloc[start_idx:end_idx]

            # Save segment statistics
            segment_stats = {
                "segment": i + 1,
                "start_index": start_idx,
                "end_index": end_idx,
                "length": end_idx - start_idx,
                "mean": np.mean(segment_data),
                "std": np.std(segment_data),
                "skewness": stats.skew(segment_data),
                "kurtosis": stats.kurtosis(segment_data),
            }

            # Save segment data
            _ = pd.DataFrame(segment_stats, index=[0]).to_csv(
                dim_folder / f"segment_{i+1}_statistics.csv", index=False
            )

            # Create segment plot
            self._plot_segment(segment_data, i + 1, dim_folder)

    def _plot_segment(self, segment_data, segment_num, output_folder):
        """Plot individual segment with distribution"""
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"Segment {segment_num} Time Series",
                f"Segment {segment_num} Distribution",
                f"Segment {segment_num} ACF",
                f"Segment {segment_num} Q-Q Plot",
            ),
        )

        # Time series
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(segment_data)),
                y=segment_data.values,
                mode="lines",
                name="Time Series",
            ),
            row=1,
            col=1,
        )

        # Distribution
        fig.add_trace(
            go.Histogram(x=segment_data.values, nbinsx=30, name="Distribution"),
            row=1,
            col=2,
        )

        # ACF
        acf_values = acf(segment_data, nlags=min(40, len(segment_data) // 4))
        fig.add_trace(
            go.Bar(x=np.arange(len(acf_values)), y=acf_values, name="ACF"), row=2, col=1
        )

        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(segment_data))
        )
        sample_quantiles = np.sort(segment_data.values)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode="markers",
                name="Q-Q Plot",
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"{self.friendly_name} - Segment {segment_num} Analysis",
            height=800,
            showlegend=False,
        )

        _ = fig.write_html(output_folder / f"segment_{segment_num}_analysis.html")

    def plot_acf_pacf(self):
        """Create ACF and PACF plots"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # ACF plot
        plot_acf(
            self.clean_data_series,
            lags=min(50, len(self.clean_data_series) // 4),
            ax=axes[0],
            alpha=0.05,
        )
        axes[0].set_title(f"Autocorrelation Function - {self.friendly_name}")

        # PACF plot
        plot_pacf(
            self.clean_data_series,
            lags=min(50, len(self.clean_data_series) // 4),
            ax=axes[1],
            alpha=0.05,
        )
        axes[1].set_title(f"Partial Autocorrelation Function - {self.friendly_name}")

        plt.tight_layout()
        plt.savefig(
            PLOTS_FOLDER / f"{self.dimension_name}_acf_pacf.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print("ACF and PACF plots created")

    def plot_distribution_analysis(self):
        """Create comprehensive distribution plots"""
        fig = make_subplots(
            rows=2,
            cols=3,
            subplot_titles=(
                "Distribution Histogram",
                "Q-Q Plot",
                "Box Plot",
                "Violin Plot",
                "Time Series",
                "Rolling Statistics",
            ),
        )

        # Histogram with KDE
        hist_data = go.Histogram(
            x=self.clean_data_series.values,
            nbinsx=50,
            name="Histogram",
            histnorm="probability density",
        )
        fig.add_trace(hist_data, row=1, col=1)

        # Add KDE
        kde_x = np.linspace(
            self.clean_data_series.min(), self.clean_data_series.max(), 100
        )
        kde = stats.gaussian_kde(self.clean_data_series.values)
        kde_y = kde(kde_x)
        fig.add_trace(
            go.Scatter(x=kde_x, y=kde_y, mode="lines", name="KDE"), row=1, col=1
        )

        # Q-Q plot
        theoretical_quantiles = stats.norm.ppf(
            np.linspace(0.01, 0.99, len(self.clean_data_series))
        )
        sample_quantiles = np.sort(self.clean_data_series.values)
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
        # Add reference line
        fig.add_trace(
            go.Scatter(
                x=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                y=[theoretical_quantiles.min(), theoretical_quantiles.max()],
                mode="lines",
                line=dict(color="red", dash="dash"),
                name="Normal",
            ),
            row=1,
            col=2,
        )

        # Box plot
        fig.add_trace(
            go.Box(y=self.clean_data_series.values, name="Box Plot"), row=1, col=3
        )

        # Violin plot
        fig.add_trace(
            go.Violin(
                y=self.clean_data_series.values,
                name="Violin Plot",
                box_visible=True,
                meanline_visible=True,
            ),
            row=2,
            col=1,
        )

        # Time series
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.clean_data_series)),
                y=self.clean_data_series.values,
                mode="lines",
                name="Time Series",
                line=dict(width=0.5),
            ),
            row=2,
            col=2,
        )

        # Rolling statistics
        window = min(
            168, len(self.clean_data_series) // 10
        )  # Weekly window or 10% of data
        rolling_mean = (
            pd.Series(self.clean_data_series.values).rolling(window=window).mean()
        )
        rolling_std = (
            pd.Series(self.clean_data_series.values).rolling(window=window).std()
        )

        fig.add_trace(
            go.Scatter(
                x=np.arange(len(rolling_mean)),
                y=rolling_mean,
                mode="lines",
                name="Rolling Mean",
            ),
            row=2,
            col=3,
        )
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(rolling_std)),
                y=rolling_std,
                mode="lines",
                name="Rolling Std",
            ),
            row=2,
            col=3,
        )

        # Add statistics text
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

        _ = fig.write_html(
            PLOTS_FOLDER / f"{self.dimension_name}_distribution_analysis.html"
        )
        print("Distribution analysis plots created")

    def fit_sarima_model(self, seasonal_period=24):
        """Fit SARIMA model with automatic parameter selection"""
        print(f"Fitting SARIMA model for {self.friendly_name}...")

        try:
            # Use auto_arima for parameter selection
            self.model = pm.auto_arima(
                self.clean_data_series,
                seasonal=True,
                m=seasonal_period,  # Seasonal period (24 for hourly data with daily seasonality)
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

            # Get model parameters
            order = self.model.order
            seasonal_order = self.model.seasonal_order

            self.results["sarima_params"] = {
                "order": order,
                "seasonal_order": seasonal_order,
                "aic": self.model.aic(),
                "bic": self.model.bic(),
                "model_summary": str(self.model.summary()),
            }

            # Save model
            model_path = MODELS_FOLDER / f"{self.dimension_name}_sarima_model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(self.model, f)

            # Save model parameters
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
            _ = params_df.to_csv(
                TABLES_FOLDER / f"{self.dimension_name}_sarima_params.csv", index=False
            )

            print(f"SARIMA{order}x{seasonal_order} fitted successfully")
            print(f"AIC: {self.model.aic():.2f}, BIC: {self.model.bic():.2f}")

            # Perform residual diagnostics
            self._analyze_residuals()

            return self.model

        except Exception as e:
            print(f"Error fitting SARIMA model: {e}")
            self.results["sarima_params"] = {"error": str(e)}
            return None

    def _analyze_residuals(self):
        """Analyze model residuals"""
        if self.model is None:
            return

        residuals = self.model.resid()

        # Ljung-Box test for residual autocorrelation
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)

        # Create residual plots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Residuals Over Time",
                "Residual Distribution",
                "Residual ACF",
                "Residual Q-Q Plot",
            ),
        )

        # Residuals over time
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(residuals)), y=residuals, mode="lines", name="Residuals"
            ),
            row=1,
            col=1,
        )

        # Residual histogram
        fig.add_trace(
            go.Histogram(x=residuals, nbinsx=30, name="Distribution"), row=1, col=2
        )

        # Residual ACF
        acf_values = acf(residuals, nlags=min(40, len(residuals) // 4))
        fig.add_trace(
            go.Bar(x=np.arange(len(acf_values)), y=acf_values, name="ACF"), row=2, col=1
        )

        # Residual Q-Q plot
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles, y=sample_quantiles, mode="markers", name="Q-Q"
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            title=f"Residual Analysis - {self.friendly_name}",
            height=800,
            showlegend=False,
        )

        _ = fig.write_html(
            PLOTS_FOLDER / f"{self.dimension_name}_residual_analysis.html"
        )

        # Save Ljung-Box test results
        _ = lb_test.to_csv(TABLES_FOLDER / f"{self.dimension_name}_ljungbox_test.csv")

        print("Residual analysis completed")

    def create_forecast_plot(self, n_periods=168):
        """Create forecast plot"""
        if self.model is None:
            return

        # Generate forecast
        forecast = self.model.predict(n_periods=n_periods)
        conf_int = self.model.predict(n_periods=n_periods, return_conf_int=True)[1]

        # Create plot
        fig = go.Figure()

        # Historical data
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(self.clean_data_series)),
                y=self.clean_data_series.values,
                mode="lines",
                name="Historical",
                line=dict(color="blue", width=0.5),
            )
        )

        # Forecast
        forecast_x = np.arange(
            len(self.clean_data_series), len(self.clean_data_series) + n_periods
        )
        fig.add_trace(
            go.Scatter(
                x=forecast_x,
                y=forecast,
                mode="lines",
                name="Forecast",
                line=dict(color="red", width=2),
            )
        )

        # Confidence intervals
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([forecast_x, forecast_x[::-1]]),
                y=np.concatenate([conf_int[:, 0], conf_int[:, 1][::-1]]),
                fill="toself",
                fillcolor="rgba(255,0,0,0.2)",
                line=dict(color="rgba(255,0,0,0)"),
                name="95% CI",
            )
        )

        fig.update_layout(
            title=f"SARIMA Forecast - {self.friendly_name}",
            xaxis_title="Time Index",
            yaxis_title="Value",
            height=600,
        )

        _ = fig.write_html(PLOTS_FOLDER / f"{self.dimension_name}_forecast.html")
        print(f"Forecast plot created ({n_periods} periods)")

    def run_complete_analysis(self):
        """Run the complete SARIMA analysis pipeline"""
        print(f"\n{'='*60}")
        print(f"Starting SARIMA Analysis for: {self.friendly_name}")
        print(f"{'='*60}")

        try:
            # 1. Clean data
            self.clean_data()

            # 2. Calculate statistics
            self.calculate_statistics()

            # 3. Perform ADF test
            self.perform_adf_test()

            # 4. Detect frequency components
            self.detect_frequency_components()

            # 5. Detect multimodal distribution
            self.detect_multimodal_distribution()

            # 6. Create ACF/PACF plots
            self.plot_acf_pacf()

            # 7. Create distribution analysis plots
            self.plot_distribution_analysis()

            # 8. Fit SARIMA model
            self.fit_sarima_model()

            # 9. Create forecast plot
            if self.model is not None:
                self.create_forecast_plot()

            print(f"Analysis completed successfully for {self.friendly_name}")
            return self.results

        except Exception as e:
            print(f"Error in analysis for {self.dimension_name}: {e}")
            import traceback

            traceback.print_exc()
            return {"error": str(e)}


def create_summary_report(all_results):
    """Create a comprehensive summary report"""
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
                    "Is Stationary": results.get("adf", {}).get("is_stationary", False),
                    "ADF p-value": results.get("adf", {}).get("p_value", np.nan),
                    "Is Multimodal": results.get("multimodal_info", {}).get(
                        "is_multimodal", False
                    ),
                    "N Modes": results.get("multimodal_info", {}).get("n_modes", 1),
                    "Low Freq Components": results.get("frequency_components", {}).get(
                        "low_frequency_count", 0
                    ),
                    "High Freq Components": results.get("frequency_components", {}).get(
                        "high_frequency_count", 0
                    ),
                    "SARIMA Order": str(
                        results.get("sarima_params", {}).get("order", "N/A")
                    ),
                    "Seasonal Order": str(
                        results.get("sarima_params", {}).get("seasonal_order", "N/A")
                    ),
                    "AIC": results.get("sarima_params", {}).get("aic", np.nan),
                    "BIC": results.get("sarima_params", {}).get("bic", np.nan),
                }
            )

    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        _ = summary_df.to_csv(
            TABLES_FOLDER / "complete_analysis_summary.csv", index=False
        )
        _ = summary_df.to_excel(
            TABLES_FOLDER / "complete_analysis_summary.xlsx", index=False
        )
        print("\nSummary report saved to tables folder")
        return summary_df

    return None


def main():
    """Main execution function - use this with your load_df function"""
    print("Starting SARIMA Time Series Analysis")
    print("=" * 60)

    # Import your load_df function
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from utils.load_df import load_df

    # Load data
    print("Loading data...")
    df = load_df()

    if df.empty:
        print("Error: Empty dataframe")
        return

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Store all results
    all_results = {}

    # Analyze each dimension
    for dimension in DIMS:
        if dimension not in df.columns:
            print(f"\nWarning: Dimension '{dimension}' not found in dataframe")
            continue

        try:
            # Get data for this dimension
            data = df[dimension]
            friendly_name = DIMENSION_NAMES.get(dimension, dimension)

            # Create analyzer
            analyzer = SARIMAAnalyzer(data, dimension, friendly_name)

            # Run complete analysis
            results = analyzer.run_complete_analysis()

            # Store results
            all_results[dimension] = results

        except Exception as e:
            print(f"\nError analyzing {dimension}: {e}")
            all_results[dimension] = {"error": str(e)}

    # Create summary report
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
    print(f"  - Models: {MODELS_FOLDER}")
    print(f"  - Plots: {PLOTS_FOLDER}")
    print(f"  - Tables: {TABLES_FOLDER}")
    print(f"  - Multimodal Analysis: {MULTIMODAL_FOLDER}")


if __name__ == "__main__":
    main()
