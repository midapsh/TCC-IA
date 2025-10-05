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
LOGGER = logging.getLogger(__name__)


class Station(NamedTuple):
    code: str
    id_code: int


class StationTemp(TypedDict, total=False):
    code: str
    id_code: int


class StationMetadata(NamedTuple):
    id_code: int
    region: str
    state: str
    name: str
    latitude: str
    longitude: str
    altitude: str | None
    foundation_date: str
    year: str
    filename: str


class StationMetadataTemp(TypedDict, total=False):
    id_code: int
    region: str
    state: str
    name: str
    latitude: str
    longitude: str
    altitude: str | None
    foundation_date: str
    year: str
    filename: str


class FileMetadata(NamedTuple):
    station: Station
    station_metadata: StationMetadata


def string_to_number(text: str, /) -> int:
    return int("".join("{}".format(ord(c)) for c in text))


# -------------------- PARSE HEADER --------------------
def _parse_header(file_path: Path, /) -> FileMetadata:
    station_temp: StationTemp = {}
    station_metadata_temp: StationMetadataTemp = {}
    try:
        with file_path.open(mode="r") as f:
            for _ in range(8):
                line = f.readline().strip()
                if ":;" not in line:
                    continue

                key, value = line.split(":;", 1)
                key = slugify(key)
                value = value.strip()

                if "regiao" in key or "regio" in key:
                    station_metadata_temp["region"] = value
                    continue
                elif "uf" in key:
                    station_metadata_temp["state"] = value
                    continue
                elif "estacao" in key or "estaco" in key:
                    station_metadata_temp["name"] = value
                    continue
                elif "codigo" in key:
                    id_code = string_to_number(value)
                    station_temp["code"] = value
                    station_temp["id_code"] = id_code
                    station_metadata_temp["id_code"] = id_code
                    continue
                elif "latitude" in key:
                    station_metadata_temp["latitude"] = value.replace(",", ".")
                    continue
                elif "longitude" in key:
                    station_metadata_temp["longitude"] = value.replace(",", ".")
                    continue
                elif "altitude" in key:
                    if value.strip().upper() == "F":
                        station_metadata_temp["altitude"] = None
                        continue
                    else:
                        station_metadata_temp["altitude"] = value.replace(",", ".")
                        continue
                elif "fundacao" in key or "fundaco" in key:
                    try:
                        station_metadata_temp["foundation_date"] = (
                            datetime.strptime(value, "%Y-%m-%d").date().isoformat()
                        )
                        continue
                    except ValueError:
                        station_metadata_temp["foundation_date"] = (
                            datetime.strptime(value, "%d/%m/%y").date().isoformat()
                        )
                        continue

        station = Station(**station_temp)

        station_metadata_temp["year"] = file_path.parent.name
        station_metadata_temp["filename"] = file_path.name
        station_metadata = StationMetadata(**station_metadata_temp)

        file_metadata = FileMetadata(
            station=station,
            station_metadata=station_metadata,
        )
        return file_metadata
    except Exception as e:
        LOGGER.error(
            "Error parsing file: %s\nError: %s\nStation: %s\nMetadata: %s",
            file_path,
            str(e),
            station_temp,
            station_metadata_temp,
        )
        raise


# -------------------- LOAD ALL FileMetadata --------------------
def get_all_metadata() -> list[FileMetadata]:
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


class AggFileMetadata(NamedTuple):
    station: Station
    stations_metadata: list[StationMetadata]


def agg_unique_consume(data: list[FileMetadata], /) -> list[AggFileMetadata]:
    LOGGER.info("Starting uniqueness filter. Total stations to map: %d", len(data))

    aggregator: dict[str, AggFileMetadata] = {}

    for file_metadata in data:
        key = file_metadata.station.code
        if key not in aggregator:
            aggregator[key] = AggFileMetadata(
                station=file_metadata.station,
                stations_metadata=[file_metadata.station_metadata],
            )
        else:
            aggregator[key].stations_metadata.append(file_metadata.station_metadata)

    duplicates_count = len(data) - len(aggregator)
    if duplicates_count > 0:
        LOGGER.warning("Removed %d duplicate stations", duplicates_count)

    data.clear()

    LOGGER.info("Uniqueness filter complete. Unique stations: %d", len(data))

    temp = [agg for agg in aggregator.values()]
    aggregator.clear()
    return temp


def setup_database() -> None:
    STMT_STATION = """
CREATE TABLE station (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    code TEXT NOT NULL,
    id_code INTEGER NOT NULL
)
    """

    STMT_STATION_METADATA = """
CREATE TABLE station_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_code INTEGER NOT NULL,
    region TEXT NOT NULL,
    state TEXT NOT NULL,
    name TEXT NOT NULL,
    year INTEGER NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    altitude REAL NULL,
    filename TEXT NOT NULL,
    foundation_date DATE NOT NULL,
    FOREIGN KEY (id_code) REFERENCES station(id_code)
)
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.execute("DROP TABLE IF EXISTS station")
        conn.execute("DROP TABLE IF EXISTS station_metadata")
        conn.execute(STMT_STATION)
        conn.execute(STMT_STATION_METADATA)

        for col in [
            "code",
            "id_code",
        ]:
            conn.execute(f"CREATE INDEX idx_station_{col} ON station({col})")

        for col in [
            "id_code",
            "region",
            "state",
            "name",
            "year",
            "latitude",
            "longitude",
            "altitude",
            "filename",
            "foundation_date",
        ]:
            conn.execute(
                f"CREATE INDEX idx_station_metadata_{col} ON station_metadata({col})"
            )
    return


def save_metadata_consume(data: list[AggFileMetadata], /) -> None:
    LOGGER.info("Save metadata")

    list_station = [d.station for d in data]
    list_station_metadata = []
    for d in data:
        list_station_metadata.extend(d.stations_metadata)
    data.clear()

    STMT_INSERT_STATION_METADATA = """
    INSERT INTO station (
        code, id_code
    ) VALUES (?, ?);
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.executemany(STMT_INSERT_STATION_METADATA, list_station)

    STMT_RAW_LOCATION_DATA = """
    INSERT INTO station_metadata (
        id_code, region, state, name, latitude, longitude, altitude, foundation_date, year, filename
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.executemany(STMT_RAW_LOCATION_DATA, list_station_metadata)

    return


def main() -> None:
    setup_database()
    data = get_all_metadata()
    agg = agg_unique_consume(data)
    save_metadata_consume(agg)
    return


if __name__ == "__main__":
    main()
