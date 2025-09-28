"""
# Notas 1
Radiacao Global
- Dividir por dias que se fez muito sol ou pouco sol
- Verificar se nos dias que teve mais ou menos radiacao rolou algo diferente
Precipitacao total horario
- Serve pra medida mensal
- Talvez aplicar um iFFT para ver se acha alguma relacao inversa
Pressao atmosferica ao nivel da estacao
- Meio estranho porque nao se repete bem a cada mes
Temperatura do ar bulbo seco
- Verificar se tem alguma relacao com a radiacao

# Notas 2
- Engracado como a `temperatura do ar bulbo seco` vs a `radiacao global` em setembro 2006/2007 (abaixou a temperatura)
possui em comportamento que em julho 2006/2007 (aumentou a temperatura). Mas, uma coisa que se manteve, é a variacao
de amplitude; verificar se há uma relacao de variacao entre `temperatura do ar bulbo seco` e a `radiacao global`. Usar
min e max aqui, pq da uma ideia de quantil ao inves de media
    - A `umidade relativa do ar` tinha parado de ser medida em 2007. Talvez tenha dado pau em todos os roles de medicao
- Temperatura do ponto de orvalho é uma porra meio inutil a primeira vista
- Vento parece bem inutil

"""

from os import makedirs
from pathlib import Path
from typing import Literal, Union

from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils.load_df import load_df


# IMAGES_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/images")
# DATA_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/data")
# DATABASE_URI = str(DATA_FOLDER / "database.db")
# In-memory db
# DATABASE_URI = "/mnt/ramdisk/database.db"
# HTML_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/html")
HTML_MONTHLY_FOLDER = Path(
    "/home/dolores/Documents/matheus-ferreira/TCC-IA/html/monthly"
)
makedirs(HTML_MONTHLY_FOLDER, exist_ok=True)


# lista de dimensões passadas na sua mensagem
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


def create_monthly_timeseries_plots(
    df: pd.DataFrame, column_to_plot: str, /, date_column: str = "datetime"
):
    """
    Create twelve time series plots (one for each month) showing multiple years as different lines.

    Parameters:
    df: pandas DataFrame with datetime index or date column
    column_to_plot: string, name of the column to plot
    date_column: string, name of the date column (if not using datetime index)
    """

    # Extract month, year, day, and hour for grouping
    df["year"] = df["datetime"].dt.year
    df["month"] = df["datetime"].dt.month
    df["day"] = df["datetime"].dt.day
    df["hour"] = df["datetime"].dt.hour

    # Month names for titles
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Create subplots (3x4 grid for 12 months)
    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=month_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Color palette for different years
    colors = px.colors.qualitative.Set1

    # Get unique years
    years = sorted(df["year"].unique())

    # Create a plot for each month
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1

        month_data = df[df["month"] == month].copy()

        if month_data.empty:
            continue

        # For each year, create a line
        for i, year in enumerate(years):
            year_data = month_data[month_data["year"] == year].copy()

            if year_data.empty:
                continue

            # Sort by day and hour for proper line connection
            year_data = year_data.sort_values(["day", "hour"])

            # Create x-axis values (day-hour combination)
            x_values = year_data["day"] + year_data["hour"] / 24

            _ = fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=year_data[column_to_plot],
                    mode="lines",
                    name=f"{year}",
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=(
                        True if month == 1 else False
                    ),  # Show legend only for first subplot
                    legendgroup=f"{year}",  # Group legends by year
                ),
                row=row,
                col=col,
            )

        # Update x-axis for this subplot
        _ = fig.update_xaxes(title_text="Day of Month", row=row, col=col)

        # Update y-axis for this subplot
        _ = fig.update_yaxes(
            title_text=column_to_plot.replace("_", " ").title(), row=row, col=col
        )

    # Update layout
    _ = fig.update_layout(
        title=f'Monthly Time Series: {column_to_plot.replace("_", " ").title()}',
        hovermode="x",
        height=900,
        width=2300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def create_daily_timeseries_plots_sampled(
    df: pd.DataFrame,
    column_to_plot: str,
    /,
    date_column: str = "datetime",
    sample_type: Union[
        Literal["mean"],
        Literal["median"],
        Literal["min"],
        Literal["max"],
        Literal["fixed_hour"],
    ] = "mean",
    fixed_hour: int = 12,  # <-- choose which hour to sample each day
):
    """
    Create twelve time series plots (one for each month) showing daily aggregated values.
    Multiple years appear as different lines.

    Parameters:
    df: pandas DataFrame with datetime index or date column
    column_to_plot: string, name of the column to plot
    date_column: string, name of the date column (if not using datetime index)
    """

    # Extract year, month, day
    df["year"] = df[date_column].dt.year
    df["month"] = df[date_column].dt.month
    df["day"] = df[date_column].dt.day

    daily_df = None
    match sample_type:
        case "mean":
            daily_df = (
                df.groupby(["year", "month", "day"], as_index=False)[column_to_plot]
                .mean()
                .sort_values(["year", "month", "day"])
            )  # type: ignore
        case "median":
            daily_df = (
                df.groupby(["year", "month", "day"], as_index=False)[column_to_plot]
                .median()
                .sort_values(["year", "month", "day"])
            )  # type: ignore
        case "min":
            daily_df = (
                df.groupby(["year", "month", "day"], as_index=False)[column_to_plot]
                .min()
                .sort_values(["year", "month", "day"])
            )  # type: ignore
        case "max":
            daily_df = (
                df.groupby(["year", "month", "day"], as_index=False)[column_to_plot]
                .max()
                .sort_values(["year", "month", "day"])
            )  # type: ignore
        case "fixed_hour":
            df["hour"] = df[date_column].dt.hour
            daily_df = (
                df[df["hour"] == fixed_hour]
                .sort_values(["year", "month", "day"])
                .copy()
            )

    # Month names for titles
    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    # Create subplots (3x4 grid for 12 months)
    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=month_names,
        vertical_spacing=0.08,
        horizontal_spacing=0.05,
    )

    # Color palette for different years
    colors = px.colors.qualitative.Set1

    # Get unique years
    years = sorted(daily_df["year"].unique())

    # Create a plot for each month
    for month in range(1, 13):
        row = (month - 1) // 4 + 1
        col = (month - 1) % 4 + 1

        month_data = daily_df[daily_df["month"] == month].copy()

        if month_data.empty:
            continue

        # For each year, create a line
        for i, year in enumerate(years):
            year_data = month_data[month_data["year"] == year].copy()

            if year_data.empty:
                continue

            # x-axis = day of month
            x_values = year_data["day"]

            _ = fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=year_data[column_to_plot],
                    mode="lines",
                    name=f"{year}",
                    line=dict(color=colors[i % len(colors)]),
                    showlegend=True if month == 1 else False,
                    legendgroup=f"{year}",
                ),
                row=row,
                col=col,
            )

        # Update x-axis for this subplot
        _ = fig.update_xaxes(title_text="Day of Month", row=row, col=col)

        # Update y-axis for this subplot
        _ = fig.update_yaxes(
            title_text=column_to_plot.replace("_", " ").title(), row=row, col=col
        )

    # Update layout
    _ = fig.update_layout(
        title=f'Daily Time Series: {column_to_plot.replace("_", " ").title()}',
        hovermode="x",
        height=900,
        width=2300,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    df = load_df()
    # Choose a dimension to plot
    # column_to_analyze = (
    #     "temperatura_do_ar_bulbo_seco_horaria"  # Change this to any column from dims
    # )
    for column_to_analyze in DIMS:
        # Create a single figure with 12 subplots
        try:
            fig_combined = create_monthly_timeseries_plots(df, column_to_analyze)

            # Save as HTML
            _ = fig_combined.write_html(
                str(
                    HTML_MONTHLY_FOLDER / f"monthly_timeseries_{column_to_analyze}.html"
                )
            )
            print(f"Combined plot saved as monthly_timeseries_{column_to_analyze}.html")

        except Exception as e:
            print(f"Error creating combined plot: {e}")

        # Create a single figure with 12 subplots sampled
        sample_types = [
            "mean",
            "median",
            "min",
            "max",
            "fixed_hour",
        ]
        for sample_type in sample_types:
            try:
                fig_combined = create_daily_timeseries_plots_sampled(
                    df,
                    column_to_analyze,
                    sample_type=sample_type,  # type: ignore
                )

                # Save as HTML
                _ = fig_combined.write_html(
                    str(
                        HTML_MONTHLY_FOLDER
                        / f"monthly_timeseries_sampled_{sample_type}_{column_to_analyze}.html"
                    )
                )
                print(
                    f"Combined plot saved as monthly_timeseries_sampled_{sample_type}_{column_to_analyze}.html"
                )

            except Exception as e:
                print(f"Error creating combined plot: {e}")


# Example usage - uncomment and modify as needed:
if __name__ == "__main__":
    main()
