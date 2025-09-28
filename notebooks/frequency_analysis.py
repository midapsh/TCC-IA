from utils.load_df import load_df
from os import makedirs
from pathlib import Path
from scipy.fft import fft, fftfreq
from scipy.signal import spectrogram, welch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
HTML_FREQ_FOLDER = Path(
    "/home/dolores/Documents/matheus-ferreira/TCC-IA/html/frequency_analysis"
)


def create_dimension_folder(dimension: str, /) -> Path:
    """Create a subfolder for each dimension."""
    folder = HTML_FREQ_FOLDER / dimension
    makedirs(folder, exist_ok=True)
    return folder


def analyze_dimension_fft(df: pd.DataFrame, dimension: str) -> dict:
    """
    Analyze a single dimension using FFT and create frequency domain plots.

    Returns:
        dict: Dictionary containing analysis results and statistics
    """
    print(f"\n{'='*60}")
    print(f"Analyzing dimension: {dimension}")
    print(f"{'='*60}")

    # Create folder for this dimension
    # output_folder = create_dimension_folder(dimension)
    output_folder = HTML_FREQ_FOLDER

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
    print(f"Non-zero count: {np.count_nonzero(~np.isnan(signal))}")

    # Handle missing values
    valid_mask = ~np.isnan(signal)
    signal_clean = signal[valid_mask]

    if len(signal_clean) == 0:
        print(f"Error: No valid data after removing NaN values for {dimension}")
        return {"status": "no_valid_data", "dimension": dimension}

    # Remove DC component (mean) for better frequency analysis
    signal_mean = np.mean(signal_clean)
    signal_clean = signal_clean - signal_mean

    N = len(signal_clean)
    dt = 1.0  # sampling interval in hours

    # FFT calculation
    fft_vals = fft(signal_clean)
    freqs = fftfreq(N, d=dt)

    # Calculate magnitude and phase
    magnitude = np.abs(fft_vals) / N  # Normalized magnitude
    phase = np.angle(fft_vals)

    # Power spectral density using Welch's method for comparison
    f_welch, psd = welch(signal_clean, fs=1 / dt, nperseg=min(256, N // 4))

    # Convert to cycles/day
    freqs_per_day = freqs * 24
    f_welch_per_day = f_welch * 24

    # Filter for positive frequencies only
    positive_mask = freqs > 0

    if np.sum(positive_mask) == 0:
        print(f"Warning: No positive frequencies found for {dimension}")
        return {"status": "no_positive_freq", "dimension": dimension}

    # Get friendly name
    friendly_name = DIMENSION_NAMES.get(dimension, dimension)

    # 1. Create magnitude spectrum plot
    fig_mag = go.Figure()
    fig_mag.add_trace(
        go.Scatter(
            x=freqs_per_day[positive_mask],
            y=magnitude[positive_mask],
            mode="lines",
            name="FFT Magnitude",
            line=dict(width=1.5, color="blue"),
            hovertemplate="Freq: %{x:.3f} cycles/day<br>Magnitude: %{y:.6f}<extra></extra>",
        )
    )

    # Add important frequency markers
    fig_mag.add_vline(
        x=1,
        line_dash="dash",
        line_color="red",
        opacity=0.5,
        annotation_text="Daily (24h)",
    )
    fig_mag.add_vline(
        x=2,
        line_dash="dash",
        line_color="orange",
        opacity=0.5,
        annotation_text="Semi-daily (12h)",
    )
    fig_mag.add_vline(
        x=7, line_dash="dash", line_color="green", opacity=0.5, annotation_text="Weekly"
    )

    fig_mag.update_layout(
        title=f"Frequency Spectrum (Magnitude) - {friendly_name}",
        xaxis_title="Frequency [cycles/day]",
        yaxis_title="Normalized Magnitude",
        template="plotly_white",
        xaxis=dict(type="log", range=[-2, 2]),  # Log scale for better visualization
        yaxis=dict(type="log"),
        height=600,
    )
    _ = fig_mag.write_html(output_folder / f"{dimension}_magnitude.html")

    # 2. Create phase spectrum plot
    fig_phase = go.Figure()
    fig_phase.add_trace(
        go.Scatter(
            x=freqs_per_day[positive_mask],
            y=phase[positive_mask],
            mode="markers",
            name="Phase",
            marker=dict(size=2, color="purple", opacity=0.6),
            hovertemplate="Freq: %{x:.3f} cycles/day<br>Phase: %{y:.3f} rad<extra></extra>",
        )
    )
    fig_phase.update_layout(
        title=f"Phase Spectrum - {friendly_name}",
        xaxis_title="Frequency [cycles/day]",
        yaxis_title="Phase [radians]",
        template="plotly_white",
        xaxis=dict(type="log", range=[-2, 2]),
        yaxis=dict(range=[-np.pi, np.pi]),
        height=600,
    )
    _ = fig_phase.write_html(output_folder / f"{dimension}_phase.html")

    # 3. Create Power Spectral Density plot (using Welch's method)
    fig_psd = go.Figure()
    fig_psd.add_trace(
        go.Scatter(
            x=f_welch_per_day,
            y=psd,
            mode="lines",
            name="Power Spectral Density",
            line=dict(width=1.5, color="darkgreen"),
            hovertemplate="Freq: %{x:.3f} cycles/day<br>PSD: %{y:.6e}<extra></extra>",
        )
    )

    # Add frequency markers
    fig_psd.add_vline(
        x=1, line_dash="dash", line_color="red", opacity=0.5, annotation_text="Daily"
    )
    fig_psd.add_vline(
        x=2,
        line_dash="dash",
        line_color="orange",
        opacity=0.5,
        annotation_text="Semi-daily",
    )

    fig_psd.update_layout(
        title=f"Power Spectral Density (Welch) - {friendly_name}",
        xaxis_title="Frequency [cycles/day]",
        yaxis_title="PSD [(signal unit)²/Hz]",
        template="plotly_white",
        xaxis=dict(type="log", range=[-2, 2]),
        yaxis=dict(type="log"),
        height=600,
    )
    _ = fig_psd.write_html(output_folder / f"{dimension}_psd.html")

    # 4. Create spectrogram
    try:
        # Use appropriate window size
        nperseg = min(256, N // 4)
        noverlap = nperseg // 2

        f, t, Sxx = spectrogram(
            signal_clean, fs=1 / dt, nperseg=nperseg, noverlap=noverlap
        )

        # Convert frequencies to cycles/day
        f_per_day = f * 24

        # Convert time axis to datetime if possible
        if "datetime" in df.columns:
            start_time = df["datetime"].iloc[0]
            t_datetime = pd.to_datetime(start_time) + pd.to_timedelta(t, unit="h")
            x_axis = t_datetime
            x_title = "Datetime"
        else:
            x_axis = t
            x_title = "Time [hours]"

        # Create spectrogram plot
        fig_spec = go.Figure(
            data=go.Heatmap(
                z=10 * np.log10(Sxx + 1e-12),  # Power in dB
                x=x_axis,
                y=f_per_day,
                colorscale="Viridis",
                colorbar=dict(title="Power [dB]"),
                hovertemplate="Time: %{x}<br>Frequency: %{y:.3f} cycles/day<br>Power: %{z:.2f} dB<extra></extra>",
            )
        )
        fig_spec.update_layout(
            title=f"Spectrogram - {friendly_name}",
            xaxis_title=x_title,
            yaxis_title="Frequency [cycles/day]",
            template="plotly_white",
            yaxis=dict(
                range=[0, min(12, max(f_per_day))]
            ),  # Focus on relevant frequencies
            height=600,
        )
        _ = fig_spec.write_html(output_folder / f"{dimension}_spectrogram.html")

    except Exception as e:
        print(f"Warning: Could not create spectrogram for {dimension}: {e}")

    # 5. Create combined overview plot
    fig_overview = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Time Series (first 1000 points)",
            "FFT Magnitude Spectrum",
            "Power Spectral Density",
            "Top 10 Dominant Frequencies",
        ),
        specs=[
            [{"type": "scatter"}, {"type": "scatter"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # Time series subplot
    plot_length = min(1000, len(signal_clean))
    fig_overview.add_trace(
        go.Scatter(
            x=np.arange(plot_length),
            y=signal_clean[:plot_length],
            mode="lines",
            name="Signal",
            line=dict(width=1),
        ),
        row=1,
        col=1,
    )

    # Magnitude spectrum subplot
    fig_overview.add_trace(
        go.Scatter(
            x=freqs_per_day[positive_mask][:1000],
            y=magnitude[positive_mask][:1000],
            mode="lines",
            name="Magnitude",
            line=dict(width=1),
        ),
        row=1,
        col=2,
    )

    # PSD subplot
    fig_overview.add_trace(
        go.Scatter(
            x=f_welch_per_day, y=psd, mode="lines", name="PSD", line=dict(width=1)
        ),
        row=2,
        col=1,
    )

    # Top frequencies bar chart
    top_n = 10
    top_indices = np.argsort(magnitude[positive_mask])[-top_n:][::-1]
    top_freqs = freqs_per_day[positive_mask][top_indices]
    top_mags = magnitude[positive_mask][top_indices]

    fig_overview.add_trace(
        go.Bar(
            x=top_freqs,
            y=top_mags,
            name="Top Frequencies",
            text=[f"{f:.3f}" for f in top_freqs],
            textposition="auto",
        ),
        row=2,
        col=2,
    )

    # Update layout
    fig_overview.update_xaxes(title_text="Sample", row=1, col=1)
    fig_overview.update_xaxes(
        title_text="Frequency [cycles/day]", type="log", row=1, col=2
    )
    fig_overview.update_xaxes(
        title_text="Frequency [cycles/day]", type="log", row=2, col=1
    )
    fig_overview.update_xaxes(title_text="Frequency [cycles/day]", row=2, col=2)

    fig_overview.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig_overview.update_yaxes(title_text="Magnitude", type="log", row=1, col=2)
    fig_overview.update_yaxes(title_text="PSD", type="log", row=2, col=1)
    fig_overview.update_yaxes(title_text="Magnitude", row=2, col=2)

    fig_overview.update_layout(
        title_text=f"Spectral Analysis Overview - {friendly_name}",
        height=900,
        showlegend=False,
        template="plotly_white",
    )
    _ = fig_overview.write_html(output_folder / f"{dimension}_overview.html")

    # Calculate and print summary statistics
    max_freq_idx = np.argmax(magnitude[positive_mask])
    dominant_freq = freqs_per_day[positive_mask][max_freq_idx]
    max_magnitude = magnitude[positive_mask][max_freq_idx]

    # Find significant peaks (above 10% of max magnitude)
    threshold = 0.1 * max_magnitude
    significant_peaks = freqs_per_day[positive_mask][
        magnitude[positive_mask] > threshold
    ]

    results = {
        "status": "success",
        "dimension": dimension,
        "friendly_name": friendly_name,
        "signal_length": N,
        "dominant_frequency": dominant_freq,
        "dominant_period_days": 1 / dominant_freq if dominant_freq > 0 else np.inf,
        "max_magnitude": max_magnitude,
        "dc_component": signal_mean,
        "num_significant_peaks": len(significant_peaks),
        "output_folder": str(output_folder),
    }

    print(f"\nAnalysis Summary for {friendly_name}:")
    print(f"  Signal length: {N} samples")
    print(f"  DC component (mean): {signal_mean:.4f}")
    print(f"  Dominant frequency: {dominant_freq:.4f} cycles/day")
    print(f"  Dominant period: {results['dominant_period_days']:.2f} days")
    print(f"  Maximum magnitude: {max_magnitude:.6f}")
    print(f"  Number of significant peaks: {len(significant_peaks)}")
    print(f"  Plots saved to: {output_folder}")

    return results


def create_summary_report(results_list: list) -> None:
    """Create a summary HTML report for all dimensions."""
    print(f"\n{'='*60}")
    print("Creating summary report...")
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
                        "Dominant Freq (cycles/day)",
                        "Period (days)",
                        "Max Magnitude",
                        "Significant Peaks",
                    ],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        [r["friendly_name"] for r in successful],
                        [f"{r['dominant_frequency']:.4f}" for r in successful],
                        [f"{r['dominant_period_days']:.2f}" for r in successful],
                        [f"{r['max_magnitude']:.6f}" for r in successful],
                        [r["num_significant_peaks"] for r in successful],
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )

    fig.update_layout(
        title="Spectral Analysis Summary - All Dimensions",
        height=600,
        template="plotly_white",
    )

    _ = fig.write_html(HTML_FREQ_FOLDER / "summary_report.html")
    print(f"Summary report saved to: {HTML_FREQ_FOLDER / 'summary_report.html'}")


def main():
    """Main function to load data and perform FFT analysis on all dimensions."""
    try:
        # Create main output folder
        makedirs(HTML_FREQ_FOLDER, exist_ok=True)

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
                result = analyze_dimension_fft(df, dimension)
                results.append(result)
            except Exception as e:
                print(f"Error analyzing {dimension}: {e}")
                results.append(
                    {"status": "error", "dimension": dimension, "error": str(e)}
                )

        # Create summary report
        create_summary_report(results)

        # Final summary
        print(f"\n{'='*60}")
        print("ANALYSIS COMPLETE")
        print(f"{'='*60}")
        successful = sum(1 for r in results if r.get("status") == "success")
        print(f"Successfully analyzed: {successful}/{len(DIMS)} dimensions")
        print(f"All plots saved to: {HTML_FREQ_FOLDER}")

    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
