from utils.load_df import load_df
from os import makedirs
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pywt
from scipy import signal
from scipy.stats import kurtosis, skew

# All dimensions to analyze
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

# Friendly names for better plot titles
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

# Base folder for saving plots
HTML_WAVELET_FOLDER = Path(
    "/home/dolores/Documents/matheus-ferreira/TCC-IA/html/wavelet_analysis"
)

# Wavelets for different frequency ranges
LOW_FREQ_WAVELET = "morl"  # Morlet wavelet - excellent for low frequencies
HIGH_FREQ_WAVELET = "cmor1.5-1.0"  # Complex Morlet - good for high frequencies
GENERAL_WAVELET = "mexh"  # Mexican hat - good general purpose


def create_dimension_folders(dimension: str) -> tuple:
    """Create subfolders for each dimension and frequency type."""
    low_freq_folder = HTML_WAVELET_FOLDER / "low_frequency" / dimension
    high_freq_folder = HTML_WAVELET_FOLDER / "high_frequency" / dimension
    general_folder = HTML_WAVELET_FOLDER / "general" / dimension

    makedirs(low_freq_folder, exist_ok=True)
    makedirs(high_freq_folder, exist_ok=True)
    makedirs(general_folder, exist_ok=True)

    return low_freq_folder, high_freq_folder, general_folder


def compute_cwt_scalogram(
    signal_clean: np.ndarray, wavelet: str, scales: np.ndarray, dt: float = 1.0
):
    """Compute Continuous Wavelet Transform and return coefficients."""
    # Compute CWT
    coeffs, freqs = pywt.cwt(signal_clean, scales, wavelet, sampling_period=dt)

    # Convert to power (magnitude squared)
    power = np.abs(coeffs) ** 2

    # Convert frequencies to periods (hours)
    periods = 1 / freqs

    return coeffs, power, freqs, periods


def create_scalogram_plot(time_axis, periods, power, title, colorscale="Viridis"):
    """Create a scalogram (wavelet power spectrum) plot."""
    fig = go.Figure(
        data=go.Heatmap(
            z=np.log10(power + 1e-12),  # Log scale for better visualization
            x=time_axis,
            y=periods,
            colorscale=colorscale,
            colorbar=dict(title="Log10(Power)"),
            hovertemplate="Time: %{x}<br>Period: %{y:.2f} hours<br>Log Power: %{z:.2f}<extra></extra>",
        )
    )

    # Add period markers
    fig.add_hline(
        y=24,
        line_dash="dash",
        line_color="white",
        opacity=0.5,
        annotation_text="Daily (24h)",
    )
    fig.add_hline(
        y=12,
        line_dash="dash",
        line_color="yellow",
        opacity=0.5,
        annotation_text="Semi-daily (12h)",
    )
    fig.add_hline(
        y=168,
        line_dash="dash",
        line_color="cyan",
        opacity=0.5,
        annotation_text="Weekly (168h)",
    )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Period [hours]",
        yaxis=dict(type="log", range=[np.log10(2), np.log10(max(periods))]),
        template="plotly_dark",
        height=600,
    )

    return fig


def compute_wavelet_variance(coeffs, scales):
    """Compute wavelet variance (energy) at each scale."""
    variance = np.mean(np.abs(coeffs) ** 2, axis=1)
    return variance


def analyze_dimension_wavelet(df: pd.DataFrame, dimension: str) -> dict:
    """
    Analyze a single dimension using Wavelet Transform.

    Returns:
        dict: Dictionary containing analysis results and statistics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing dimension with wavelets: {dimension}")
    print(f"{'='*60}")

    # Create folders for this dimension
    low_folder, high_folder, general_folder = create_dimension_folders(dimension)

    # Check if dimension exists
    if dimension not in df.columns:
        print(f"Warning: Dimension '{dimension}' not found in dataframe")
        return {"status": "not_found", "dimension": dimension}

    # Extract signal
    signal = df[dimension].values

    # Basic statistics
    print(f"Original signal shape: {signal.shape}")
    print(f"Signal range: {np.nanmin(signal):.4f} to {np.nanmax(signal):.4f}")
    print(
        f"NaN count: {np.isnan(signal).sum()} ({np.isnan(signal).sum()/len(signal)*100:.2f}%)"
    )

    # Handle missing values
    valid_mask = ~np.isnan(signal)
    signal_clean = signal[valid_mask]

    if len(signal_clean) == 0:
        print(f"Error: No valid data after removing NaN values for {dimension}")
        return {"status": "no_valid_data", "dimension": dimension}

    # Normalize signal for better wavelet analysis
    signal_mean = np.mean(signal_clean)
    signal_std = np.std(signal_clean)
    signal_normalized = (signal_clean - signal_mean) / (signal_std + 1e-10)

    N = len(signal_normalized)
    dt = 1.0  # sampling interval in hours

    # Create time axis
    if "datetime" in df.columns:
        valid_dates = df.loc[valid_mask, "datetime"]
        time_axis = pd.to_datetime(valid_dates).values
    else:
        time_axis = np.arange(N) * dt

    # Get friendly name
    friendly_name = DIMENSION_NAMES.get(dimension, dimension)

    # Define scales for different frequency ranges
    # Low frequency: periods from 12 hours to 30 days
    scales_low = np.logspace(np.log10(12), np.log10(30 * 24), num=100)
    # High frequency: periods from 1 hour to 24 hours
    scales_high = np.logspace(np.log10(1), np.log10(24), num=100)
    # General: full range
    scales_general = np.logspace(np.log10(1), np.log10(30 * 24), num=150)

    results_dict = {}

    # 1. LOW FREQUENCY ANALYSIS (Morlet wavelet)
    print(f"\nPerforming low-frequency wavelet analysis ({LOW_FREQ_WAVELET})...")
    coeffs_low, power_low, freqs_low, periods_low = compute_cwt_scalogram(
        signal_normalized, LOW_FREQ_WAVELET, scales_low, dt
    )

    # Create low-frequency scalogram
    fig_low = create_scalogram_plot(
        time_axis,
        periods_low,
        power_low,
        f"Low-Frequency Scalogram ({LOW_FREQ_WAVELET}) - {friendly_name}",
        "Blues",
    )
    fig_low.write_html(low_folder / "scalogram.html")

    # Wavelet variance plot for low frequencies
    variance_low = compute_wavelet_variance(coeffs_low, scales_low)
    fig_var_low = go.Figure()
    fig_var_low.add_trace(
        go.Scatter(
            x=periods_low,
            y=variance_low,
            mode="lines+markers",
            name="Wavelet Variance",
            line=dict(width=2, color="blue"),
        )
    )
    fig_var_low.update_layout(
        title=f"Low-Frequency Wavelet Variance - {friendly_name}",
        xaxis_title="Period [hours]",
        yaxis_title="Variance (Energy)",
        xaxis=dict(type="log"),
        yaxis=dict(type="log"),
        template="plotly_white",
        height=500,
    )
    fig_var_low.write_html(low_folder / "variance.html")

    # 2. HIGH FREQUENCY ANALYSIS (Complex Morlet wavelet)
    print(f"Performing high-frequency wavelet analysis ({HIGH_FREQ_WAVELET})...")
    coeffs_high, power_high, freqs_high, periods_high = compute_cwt_scalogram(
        signal_normalized, HIGH_FREQ_WAVELET, scales_high, dt
    )

    # Create high-frequency scalogram
    fig_high = create_scalogram_plot(
        time_axis,
        periods_high,
        power_high,
        f"High-Frequency Scalogram ({HIGH_FREQ_WAVELET}) - {friendly_name}",
        "Reds",
    )
    fig_high.write_html(high_folder / "scalogram.html")

    # Wavelet variance plot for high frequencies
    variance_high = compute_wavelet_variance(coeffs_high, scales_high)
    fig_var_high = go.Figure()
    fig_var_high.add_trace(
        go.Scatter(
            x=periods_high,
            y=variance_high,
            mode="lines+markers",
            name="Wavelet Variance",
            line=dict(width=2, color="red"),
        )
    )
    fig_var_high.update_layout(
        title=f"High-Frequency Wavelet Variance - {friendly_name}",
        xaxis_title="Period [hours]",
        yaxis_title="Variance (Energy)",
        xaxis=dict(type="log"),
        yaxis=dict(type="log"),
        template="plotly_white",
        height=500,
    )
    fig_var_high.write_html(high_folder / "variance.html")

    # 3. GENERAL ANALYSIS (Mexican Hat wavelet)
    print(f"Performing general wavelet analysis ({GENERAL_WAVELET})...")
    coeffs_general, power_general, freqs_general, periods_general = (
        compute_cwt_scalogram(signal_normalized, GENERAL_WAVELET, scales_general, dt)
    )

    # Create general scalogram
    fig_general = create_scalogram_plot(
        time_axis,
        periods_general,
        power_general,
        f"General Scalogram ({GENERAL_WAVELET}) - {friendly_name}",
        "Viridis",
    )
    fig_general.write_html(general_folder / "scalogram.html")

    # 4. COMBINED OVERVIEW
    fig_overview = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            "Original Signal",
            "Wavelet Variance Comparison",
            "Low-Freq Dominant Scales",
            "High-Freq Dominant Scales",
            "Global Wavelet Spectrum",
            "Signal Statistics",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "table"}],
        ],
        vertical_spacing=0.12,
    )

    # Original signal
    plot_length = min(2000, len(signal_normalized))
    fig_overview.add_trace(
        go.Scatter(
            x=(
                time_axis[:plot_length]
                if hasattr(time_axis[0], "year")
                else np.arange(plot_length)
            ),
            y=signal_normalized[:plot_length],
            mode="lines",
            name="Signal",
            line=dict(width=1, color="blue"),
        ),
        row=1,
        col=1,
    )

    # Wavelet variance comparison
    fig_overview.add_trace(
        go.Scatter(
            x=periods_low,
            y=variance_low,
            mode="lines",
            name="Low-Freq",
            line=dict(width=2, color="blue"),
        ),
        row=1,
        col=2,
    )
    fig_overview.add_trace(
        go.Scatter(
            x=periods_high,
            y=variance_high,
            mode="lines",
            name="High-Freq",
            line=dict(width=2, color="red"),
        ),
        row=1,
        col=2,
    )

    # Top dominant scales for low frequencies
    top_n = 5
    top_low_idx = np.argsort(variance_low)[-top_n:][::-1]
    fig_overview.add_trace(
        go.Bar(
            x=periods_low[top_low_idx],
            y=variance_low[top_low_idx],
            name="Top Low-Freq",
            marker_color="blue",
            text=[f"{p:.1f}h" for p in periods_low[top_low_idx]],
            textposition="auto",
        ),
        row=2,
        col=1,
    )

    # Top dominant scales for high frequencies
    top_high_idx = np.argsort(variance_high)[-top_n:][::-1]
    fig_overview.add_trace(
        go.Bar(
            x=periods_high[top_high_idx],
            y=variance_high[top_high_idx],
            name="Top High-Freq",
            marker_color="red",
            text=[f"{p:.1f}h" for p in periods_high[top_high_idx]],
            textposition="auto",
        ),
        row=2,
        col=2,
    )

    # Global wavelet spectrum (averaged over time)
    global_spectrum = np.mean(power_general, axis=1)
    fig_overview.add_trace(
        go.Scatter(
            x=periods_general,
            y=global_spectrum,
            mode="lines",
            name="Global Spectrum",
            line=dict(width=2, color="green"),
        ),
        row=3,
        col=1,
    )

    # Statistics table
    stats_data = {
        "Metric": [
            "Mean",
            "Std Dev",
            "Skewness",
            "Kurtosis",
            "Min",
            "Max",
            "Dominant Period (Low)",
            "Dominant Period (High)",
        ],
        "Value": [
            f"{signal_mean:.4f}",
            f"{signal_std:.4f}",
            f"{skew(signal_clean):.4f}",
            f"{kurtosis(signal_clean):.4f}",
            f"{np.min(signal_clean):.4f}",
            f"{np.max(signal_clean):.4f}",
            f"{periods_low[np.argmax(variance_low)]:.1f} hours",
            f"{periods_high[np.argmax(variance_high)]:.1f} hours",
        ],
    }

    fig_overview.add_trace(
        go.Table(
            header=dict(values=["Metric", "Value"], fill_color="paleturquoise"),
            cells=dict(
                values=[stats_data["Metric"], stats_data["Value"]],
                fill_color="lavender",
            ),
        ),
        row=3,
        col=2,
    )

    # Update layout
    fig_overview.update_xaxes(title_text="Time/Sample", row=1, col=1)
    fig_overview.update_xaxes(title_text="Period [hours]", type="log", row=1, col=2)
    fig_overview.update_xaxes(title_text="Period [hours]", row=2, col=1)
    fig_overview.update_xaxes(title_text="Period [hours]", row=2, col=2)
    fig_overview.update_xaxes(title_text="Period [hours]", type="log", row=3, col=1)

    fig_overview.update_yaxes(title_text="Normalized Amplitude", row=1, col=1)
    fig_overview.update_yaxes(title_text="Variance", type="log", row=1, col=2)
    fig_overview.update_yaxes(title_text="Variance", row=2, col=1)
    fig_overview.update_yaxes(title_text="Variance", row=2, col=2)
    fig_overview.update_yaxes(title_text="Power", type="log", row=3, col=1)

    fig_overview.update_layout(
        title_text=f"Wavelet Analysis Overview - {friendly_name}",
        height=1200,
        showlegend=True,
        template="plotly_white",
    )
    fig_overview.write_html(general_folder / "overview.html")

    # 5. Time-averaged power spectrum for all wavelets
    fig_comparison = go.Figure()

    # Average power over time for each wavelet
    avg_power_low = np.mean(power_low, axis=1)
    avg_power_high = np.mean(power_high, axis=1)
    avg_power_general = np.mean(power_general, axis=1)

    fig_comparison.add_trace(
        go.Scatter(
            x=periods_low,
            y=avg_power_low,
            mode="lines",
            name=f"Low-Freq ({LOW_FREQ_WAVELET})",
            line=dict(width=2, color="blue"),
        )
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=periods_high,
            y=avg_power_high,
            mode="lines",
            name=f"High-Freq ({HIGH_FREQ_WAVELET})",
            line=dict(width=2, color="red"),
        )
    )

    fig_comparison.add_trace(
        go.Scatter(
            x=periods_general,
            y=avg_power_general,
            mode="lines",
            name=f"General ({GENERAL_WAVELET})",
            line=dict(width=2, color="green"),
        )
    )

    fig_comparison.update_layout(
        title=f"Wavelet Power Comparison - {friendly_name}",
        xaxis_title="Period [hours]",
        yaxis_title="Average Power",
        xaxis=dict(type="log"),
        yaxis=dict(type="log"),
        template="plotly_white",
        height=600,
    )
    fig_comparison.write_html(
        HTML_WAVELET_FOLDER / f"{dimension}_wavelet_comparison.html"
    )

    # Calculate dominant periods
    dominant_period_low = periods_low[np.argmax(variance_low)]
    dominant_period_high = periods_high[np.argmax(variance_high)]
    dominant_period_general = periods_general[np.argmax(np.mean(power_general, axis=1))]

    results = {
        "status": "success",
        "dimension": dimension,
        "friendly_name": friendly_name,
        "signal_length": N,
        "dominant_period_low": dominant_period_low,
        "dominant_period_high": dominant_period_high,
        "dominant_period_general": dominant_period_general,
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "signal_skewness": skew(signal_clean),
        "signal_kurtosis": kurtosis(signal_clean),
        "max_variance_low": np.max(variance_low),
        "max_variance_high": np.max(variance_high),
    }

    print(f"\nWavelet Analysis Summary for {friendly_name}:")
    print(f"  Signal length: {N} samples")
    print(f"  Signal statistics:")
    print(f"    Mean: {signal_mean:.4f}")
    print(f"    Std Dev: {signal_std:.4f}")
    print(f"    Skewness: {results['signal_skewness']:.4f}")
    print(f"    Kurtosis: {results['signal_kurtosis']:.4f}")
    print(f"  Dominant periods:")
    print(
        f"    Low-frequency: {dominant_period_low:.2f} hours ({dominant_period_low/24:.2f} days)"
    )
    print(f"    High-frequency: {dominant_period_high:.2f} hours")
    print(
        f"    General: {dominant_period_general:.2f} hours ({dominant_period_general/24:.2f} days)"
    )
    print(f"  Plots saved to respective folders in: {HTML_WAVELET_FOLDER}")

    return results


def create_summary_report(results_list: list) -> None:
    """Create a summary HTML report for all dimensions."""
    print(f"\n{'='*60}")
    print("Creating wavelet analysis summary report...")
    print(f"{'='*60}")

    # Filter successful analyses
    successful = [r for r in results_list if r.get("status") == "success"]

    if not successful:
        print("No successful analyses to summarize")
        return

    # Create summary table
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Dimension",
                        "Low-Freq Period (days)",
                        "High-Freq Period (hours)",
                        "Mean",
                        "Std Dev",
                        "Skewness",
                        "Kurtosis",
                    ],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        [r["friendly_name"] for r in successful],
                        [f"{r['dominant_period_low']/24:.2f}" for r in successful],
                        [f"{r['dominant_period_high']:.2f}" for r in successful],
                        [f"{r['signal_mean']:.4f}" for r in successful],
                        [f"{r['signal_std']:.4f}" for r in successful],
                        [f"{r['signal_skewness']:.4f}" for r in successful],
                        [f"{r['signal_kurtosis']:.4f}" for r in successful],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Wavelet Analysis Summary - All Dimensions",
        height=600,
        template="plotly_white",
    )

    fig.write_html(HTML_WAVELET_FOLDER / "wavelet_summary_report.html")
    print(
        f"Summary report saved to: {HTML_WAVELET_FOLDER / 'wavelet_summary_report.html'}"
    )


def main():
    """Main function to load data and perform wavelet analysis on all dimensions."""
    try:
        # Create main output folders
        makedirs(HTML_WAVELET_FOLDER, exist_ok=True)
        makedirs(HTML_WAVELET_FOLDER / "low_frequency", exist_ok=True)
        makedirs(HTML_WAVELET_FOLDER / "high_frequency", exist_ok=True)
        makedirs(HTML_WAVELET_FOLDER / "general", exist_ok=True)

        # Load data
        print("Loading data...")
        df = load_df()

        if df.empty:
            print("Error: Loaded dataframe is empty")
            return

        print(f"Loaded dataframe with shape: {df.shape}")
        print(f"Available columns: {list(df.columns)}")

        # Analyze each dimension
        results = []
        for dimension in DIMS:
            try:
                result = analyze_dimension_wavelet(df, dimension)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {dimension}: {e}")
                results.append(
                    {"status": "error", "dimension": dimension, "error": str(e)}
                )
                import traceback

                traceback.print_exc()

        # Create summary report
        create_summary_report(results)

        # Final summary
        print(f"\n{'='*60}")
        print("WAVELET ANALYSIS COMPLETE")
        print(f"{'='*60}")
        successful = sum(1 for r in results if r.get("status") == "success")
        print(f"Successfully analyzed: {successful}/{len(DIMS)} dimensions")
        print(f"All plots saved to: {HTML_WAVELET_FOLDER}")
        print(f"\nFolder structure:")
        print(f"  {HTML_WAVELET_FOLDER}/")
        print(
            f"    ├── low_frequency/    (Low-frequency analysis using {LOW_FREQ_WAVELET})"
        )
        print(
            f"    ├── high_frequency/   (High-frequency analysis using {HIGH_FREQ_WAVELET})"
        )
        print(f"    ├── general/          (General analysis using {GENERAL_WAVELET})")
        print(f"    └── wavelet_summary_report.html")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
