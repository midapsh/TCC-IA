import sqlite3

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from configs import CONFIGS


# Create DataFrame
def load_metadata() -> pd.DataFrame:
    STMT = """
    SELECT
        sm.name,
        sm.latitude,
        sm.longitude,
        sm.altitude
    FROM station_metadata AS sm
    JOIN (
        SELECT id_code, MAX(year) AS latest_year
        FROM station_metadata
        GROUP BY id_code
    ) AS latest
    ON sm.id_code = latest.id_code
    AND sm.year = latest.latest_year;
    """
    columns = ["name", "latitude", "longitude", "altitude"]
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        cursor = conn.execute(STMT)
        rows = cursor.fetchall()
        data = [dict(zip(columns, row)) for row in rows]
        df = pd.DataFrame(data)

    return df


def show(df: pd.DataFrame, /) -> None:
    print(f"Total stations: {len(df)}")
    print(f"\nLatitude range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    print(
        f"Longitude range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}"
    )
    return


def graphs(df: pd.DataFrame, /) -> None:
    folder_maps = CONFIGS.DATA_HTML_FOLDER / "brazil_map"
    folder_maps.mkdir(exist_ok=True)

    # Create interactive map using scatter_mapbox
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        hover_name="name",
        hover_data={"latitude": ":.4f", "longitude": ":.4f", "altitude": True},
        zoom=3.5,
        height=700,
        title="Weather Stations in Brazil",
    )

    # Use OpenStreetMap style (free, no token required)
    _ = fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=dict(lat=-15, lon=-52)),  # Center on Brazil
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
    )

    # Customize markers
    _ = fig.update_traces(
        marker=dict(size=8, color="red", opacity=0.7),
        selector=dict(mode="markers"),
    )

    _ = fig.write_html(folder_maps / "weather-stations-in-brazil.html")


def main() -> None:
    df = load_metadata()
    show(df)
    graphs(df)


if __name__ == "__main__":
    main()
