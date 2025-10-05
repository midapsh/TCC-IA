from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Any, NamedTuple
import csv
import logging
import sqlite3

from configs import CONFIGS


logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("csv_data_into_db.log"),
    level=logging.DEBUG,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)6d] [TID:%(thread)6d] "
        "%(module)s-%(lineno)d-%(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
LOGGER = logging.getLogger(__name__)


class Mapper(NamedTuple):
    id_code: int
    filepath: Path


def get_all_files() -> list[Mapper]:
    STMT = "SELECT id_code, year, filename FROM station_metadata;"
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        cursor = conn.execute(STMT)
        rows = cursor.fetchall()
        data = [
            Mapper(row[0], CONFIGS.DATA_CSV_FOLDER / str(row[1]) / row[2])
            for row in rows
        ]
    return data


class StationTimeseries(NamedTuple):
    id_code: int
    timestamp: int
    year: int
    data: str
    horario: str
    precipitacao_total_horario: float | None
    pressao_atmosferica_ao_nivel_da_estacao_horaria: float | None
    pressao_atmosferica_max_na_hora_ant: float | None
    pressao_atmosferica_min_na_hora_ant: float | None
    radiacao_global: float | None
    temperatura_do_ar_bulbo_seco_horaria: float | None
    temperatura_do_ponto_de_orvalho: float | None
    temperatura_maxima_na_hora_ant: float | None
    temperatura_minima_na_hora_ant: float | None
    temperatura_orvalho_max_na_hora_ant: float | None
    temperatura_orvalho_min_na_hora_ant: float | None
    umidade_relativa_max_na_hora_ant: float | None
    umidade_relativa_min_na_hora_ant: float | None
    umidade_relativa_do_ar_horaria: float | None
    vento_direcao_horaria: float | None
    vento_rajada_maxima: float | None
    vento_velocidade_horaria: float | None


def safe_float(value: str, /) -> float | None:
    """Convert string to float, handling empty values, commas, and -9999 sentinel values."""
    cleaned = value.strip().replace(",", ".")
    if not cleaned:
        return None

    float_val = float(cleaned)
    if float_val == -9999 or float_val == -9999.0:
        return None
    return float_val


def standard_format(datetime_hour: str, /, *, format_="%Y/%m/%d%H%M") -> datetime:
    datetime_hour = datetime_hour.replace("-", "/")
    datetime_hour = str(datetime_hour).replace("UTC", "").strip()
    if ":" in datetime_hour:
        return datetime.strptime(datetime_hour.replace(":", ""), format_)
    return datetime.strptime(datetime_hour, format_)


def process_single_file(mapper: Mapper, /) -> list[StationTimeseries]:
    """Process a single CSV file and return timeseries data."""
    csv_columns = [
        "data",
        "horario",
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

    csv_float_columns = csv_columns[2:]

    data: list[StationTimeseries] = []

    try:
        with mapper.filepath.open(mode="r") as f:
            # Skip metadata (+8 lines) + header (+1 line)
            for _ in range(9):
                _ = f.readline()

            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if not row:
                    continue
                if len(row) < len(csv_columns):
                    raise ValueError(f"Bad size: {row}")

                dict_row: dict[str, Any] = dict(zip(csv_columns, row))

                try:
                    datetime_ = standard_format(dict_row["data"] + dict_row["horario"])

                    for col in csv_float_columns:
                        dict_row[col] = safe_float(dict_row[col])

                    dict_row["id_code"] = mapper.id_code
                    dict_row["timestamp"] = int(datetime_.timestamp())
                    dict_row["data"] = datetime_.date().isoformat()
                    dict_row["horario"] = datetime_.time().isoformat()
                    dict_row["year"] = datetime_.year

                    temp = StationTimeseries(**dict_row)
                    data.append(temp)
                    continue
                except Exception:
                    LOGGER.error("Error processing row: '%s'", row, exc_info=True)
                    raise

    except Exception:
        LOGGER.error("Error processing file: '%s'", mapper.filepath, exc_info=True)
        raise

    return data


def extract_data_parallel(list_mapper: list[Mapper], /) -> list[StationTimeseries]:
    """Extract data from multiple files using multiprocessing."""
    processes = max((cpu_count() or 4) - 1, 2)
    LOGGER.info("Processing %d files with %d processes", len(list_mapper), processes)

    with Pool(processes=processes) as pool:
        results = pool.map(process_single_file, list_mapper)

    data = [item for sublist in results for item in sublist]
    LOGGER.info("Extracted %d total records", len(data))
    return data


def setup_table():
    STMT_STATION_TIMESERIE = """
CREATE TABLE IF NOT EXISTS station_timeserie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_code INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    year INTEGER NOT NULL,
    data TEXT NOT NULL,
    horario TEXT NOT NULL,
    precipitacao_total_horario REAL,
    pressao_atmosferica_ao_nivel_da_estacao_horaria REAL,
    pressao_atmosferica_max_na_hora_ant REAL,
    pressao_atmosferica_min_na_hora_ant REAL,
    radiacao_global REAL,
    temperatura_do_ar_bulbo_seco_horaria REAL,
    temperatura_do_ponto_de_orvalho REAL,
    temperatura_maxima_na_hora_ant REAL,
    temperatura_minima_na_hora_ant REAL,
    temperatura_orvalho_max_na_hora_ant REAL,
    temperatura_orvalho_min_na_hora_ant REAL,
    umidade_relativa_max_na_hora_ant REAL,
    umidade_relativa_min_na_hora_ant REAL,
    umidade_relativa_do_ar_horaria REAL,
    vento_direcao_horaria REAL,
    vento_rajada_maxima REAL,
    vento_velocidade_horaria REAL,
    FOREIGN KEY(id_code) REFERENCES station(id_code)
)
"""
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        conn.execute("DROP TABLE IF EXISTS station_timeserie")
        conn.execute(STMT_STATION_TIMESERIE)

        for col in ["id_code", "timestamp", "year"]:
            conn.execute(f"DROP INDEX IF EXISTS idx_station_timeserie_{col}")
            conn.execute(
                f"CREATE INDEX idx_station_timeserie_{col} ON station_timeserie({col})"
            )
    LOGGER.info("Table setup complete")


def save_data_batch(
    list_station_timeseries: list[StationTimeseries], /, *, batch_size: int = 10_000
) -> None:
    """Save data in batches to avoid memory issues."""
    STMT = """
    INSERT INTO station_timeserie (
        id_code, timestamp, year, data, horario, precipitacao_total_horario,
        pressao_atmosferica_ao_nivel_da_estacao_horaria,
        pressao_atmosferica_max_na_hora_ant,
        pressao_atmosferica_min_na_hora_ant, radiacao_global,
        temperatura_do_ar_bulbo_seco_horaria, temperatura_do_ponto_de_orvalho,
        temperatura_maxima_na_hora_ant, temperatura_minima_na_hora_ant,
        temperatura_orvalho_max_na_hora_ant, temperatura_orvalho_min_na_hora_ant,
        umidade_relativa_max_na_hora_ant, umidade_relativa_min_na_hora_ant,
        umidade_relativa_do_ar_horaria, vento_direcao_horaria,
        vento_rajada_maxima, vento_velocidade_horaria
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """

    total = len(list_station_timeseries)
    LOGGER.info("Saving %d records in batches of %d", total, batch_size)

    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        for i in range(0, total, batch_size):
            batch = list_station_timeseries[i : i + batch_size]
            conn.executemany(STMT, batch)
            conn.commit()
            LOGGER.info(
                "Saved batch %d (%d/%d records)",
                (i // batch_size + 1),
                (i + len(batch)),
                (total),
            )


def main() -> None:
    setup_table()
    stuff = get_all_files()
    list_station_timeseries = extract_data_parallel(stuff)
    save_data_batch(list_station_timeseries)
    LOGGER.info("Processing complete")


if __name__ == "__main__":
    main()
