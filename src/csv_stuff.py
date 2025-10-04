from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import NamedTuple, TypedDict
import logging
import sqlite3

from configs import CONFIGS
from utils.slugify import slugify


logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("csv_stuff.log"),
    level=logging.DEBUG,
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


def _parse_header(file_path: Path, /) -> StationMetadata:
    metadata: StationMetadataTemp = {}
    try:
        with file_path.open() as f:
            for _ in range(8):
                line = f.readline()
                line = line.strip()
                if ":;" not in line:
                    continue

                key, value = line.split(":;", 1)
                key = slugify(key)
                value = value.strip()

                if "regiao" in key or "regio" in key:
                    metadata["region"] = value
                elif "uf" in key:
                    metadata["state"] = value
                elif "estacao" in key or "estaco" in key:
                    metadata["station"] = value
                elif "codigo" in key:
                    metadata["code"] = value
                elif "latitude" in key:
                    metadata["latitude"] = value.replace(",", ".")
                elif "longitude" in key:
                    metadata["longitude"] = value.replace(",", ".")
                elif "altitude" in key:
                    metadata["altitude"] = value.replace(",", ".")
                elif "fundacao" in key or "fundaco" in key:
                    try:
                        metadata["foundation_date"] = (
                            datetime.strptime(value, "%Y-%m-%d").date().isoformat()
                        )
                    except ValueError:
                        metadata["foundation_date"] = (
                            datetime.strptime(value, "%d/%m/%y").date().isoformat()
                        )

        return StationMetadata(**metadata)
    except Exception as e:
        LOGGER.error(
            "Error parsing file: %s\nError: %s\nMetadata: %s",
            file_path,
            str(e),
            metadata,
        )
        raise


def get_all_metadata() -> list[StationMetadata]:
    LOGGER.info("Get all metadata")
    files = [
        filepath
        for folder_year in CONFIGS.DATA_CSV_FOLDER.iterdir()
        if folder_year.is_dir()
        for filepath in folder_year.glob("*.csv", case_sensitive=False)
    ]

    total_files = len(files)
    LOGGER.info("Total files to parse: %d", total_files)

    processes = max((cpu_count() or 4) - 1, 2)
    with Pool(processes=processes) as pool:
        data = pool.map(_parse_header, files)

    LOGGER.info("Finished parsing all files")
    return data


def filter_unique_inplace(data: list[StationMetadata], /) -> None:
    LOGGER.info("Starting uniqueness filter. Total stations to map: %d", len(data))

    seen = set()
    unique_data = []

    for station in data:
        if station in seen:
            continue
        seen.add(station)
        unique_data.append(station)

    duplicates_count = len(data) - len(unique_data)
    if duplicates_count > 0:
        LOGGER.warning("Removed %d duplicate stations", duplicates_count)

    data.clear()
    data.extend(unique_data)

    LOGGER.info("Uniqueness filter complete. Unique stations: %d", len(data))
    return


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
        )
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.execute("DROP TABLE IF EXISTS station_metadata")
        conn.execute(STMT)

        for col in [
            "region",
            "state",
            "station",
            "code",
            "latitude",
            "longitude",
            "altitude",
            "foundation_date",
        ]:
            conn.execute(
                f"CREATE INDEX idx_station_metadata_{col} ON station_metadata({col})"
            )

    return


def save_metadata(data: list[StationMetadata], /) -> None:
    LOGGER.info("Save metadata")
    STMT = """
    INSERT INTO station_metadata (
        region, state, station, code, latitude, longitude, altitude, foundation_date
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.executemany(STMT, data)

    return


def main() -> None:
    setup_database()
    data = get_all_metadata()
    filter_unique_inplace(data)
    save_metadata(data)
    return


if __name__ == "__main__":
    main()
