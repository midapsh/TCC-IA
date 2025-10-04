from datetime import datetime
from pathlib import Path
from typing import NamedTuple, TypedDict
import logging

from configs import CONFIGS

logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("csv_stuff.log"),
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)16d] [TID:%(thread)20d] "
        "%(module)s-%(lineno)d-%(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
LOGGER = logging.getLogger(__file__)


class StationMetadata(NamedTuple):
    region: str
    state: str
    station: str
    code: str
    latitude: str
    longitude: str
    altitude: str
    foundation_date: str


class StationMetadataTemp(TypedDict, total=False):
    region: str
    state: str
    station: str
    code: str
    latitude: str
    longitude: str
    altitude: str
    foundation_date: str


def parse_header(file_path: Path, /) -> StationMetadata:
    metadata: StationMetadataTemp = {}
    with file_path.open() as f:
        for _ in range(8):
            line = f.readline().strip()[1:].strip()
            if ":;" not in line:
                continue

            key, value = line.split(":;", 1)
            key = key.lower()
            value = value.strip()

            if "regiao" in key:
                metadata["region"] = value
            elif "uf" in key:
                metadata["state"] = value
            elif "estacao" in key:
                metadata["station"] = value
            elif "codigo" in key:
                metadata["code"] = value
            elif "latitude" in key:
                metadata["latitude"] = value.replace(",", ".")
            elif "longitude" in key:
                metadata["longitude"] = value.replace(",", ".")
            elif "altitude" in key:
                metadata["altitude"] = value.replace(",", ".")
            elif "fundacao" in key:
                try:
                    metadata["foundation_date"] = (
                        datetime.strptime(value, "%Y-%m-%d").date().isoformat()
                    )
                except ValueError:
                    metadata["foundation_date"] = (
                        datetime.strptime(value, "%d/%m/%y").date().isoformat()
                    )

    return StationMetadata(**metadata)


##############

from datetime import datetime
from typing import NamedTuple


def get_all_metadata() -> list[StationMetadata]:
    data = []
    for folder_year in CONFIGS.DATA_CSV_FOLDER.iterdir():
        if not folder_year.is_dir():
            continue
        for filepath in folder_year.glob("*.csv", case_sensitive=False):
            station_metadata = parse_header(filepath)
            data.append(station_metadata)
    return data


##############


def filter_unique_inplace(data: list[StationMetadata], /) -> None:
    LOGGER.info("Starting uniqueness filter. Total stations to map: %d", len(data))

    seen = set()
    unique_data = []

    for station in data:
        if station.code in seen:
            LOGGER.warning(
                "Duplicate station code found: %s - %s", station.code, station.station
            )
            continue
        seen.add(station.code)
        unique_data.append(station)

    duplicates_count = len(data) - len(unique_data)
    if duplicates_count > 0:
        LOGGER.warning("Removed %d duplicate stations", duplicates_count)

    data.clear()
    data.extend(unique_data)

    LOGGER.info("Uniqueness filter complete. Unique stations: %d", len(data))
    return


##############


import sqlite3


def setup_database() -> None:
    STMT = """
        CREATE TABLE station_metadata (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            state TEXT,
            station TEXT,
            code TEXT,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            foundation_date DATE
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.execute(STMT)
        conn.execute(
            "CREATE INDEX idx_station_metadata_region ON station_metadata(region)"
        )
        conn.execute(
            "CREATE INDEX idx_station_metadata_state ON station_metadata(state)"
        )
        conn.execute(
            "CREATE INDEX idx_station_metadata_station ON station_metadata(station)"
        )
        conn.execute("CREATE INDEX idx_station_metadata_code ON station_metadata(code)")
        conn.execute(
            "CREATE INDEX idx_station_metadata_latitude ON station_metadata(latitude)"
        )
        conn.execute(
            "CREATE INDEX idx_station_metadata_longitude ON station_metadata(longitude)"
        )
        conn.execute(
            "CREATE INDEX idx_station_metadata_altitude ON station_metadata(altitude)"
        )
        conn.execute(
            "CREATE INDEX idx_station_metadata_foundation_date ON station_metadata(foundation_date)"
        )
    return


##############


def save_metadata(data: list[StationMetadata], /) -> None:
    STMT = """
    INSERT INTO station_metadata (
    region, state, station, code, latitude, longitude, altitude, foundation_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.executemany(STMT, data)
    return


##############


def main() -> None:
    data = get_all_metadata()
    filter_unique_inplace(data)
    save_metadata(data)


if __name__ == "__main__":
    main()
