from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Any, NamedTuple
import csv
import gc  # para limpeza de memória
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


class StationTimeseries(NamedTuple):
    id_code: int
    timestamp: int
    year: int
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


def get_all_files() -> list[Mapper]:
    STMT = "SELECT id_code, year, filename FROM station_metadata;"
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        cursor = conn.execute(STMT)
        rows = cursor.fetchall()
    return [Mapper(r[0], CONFIGS.DATA_CSV_FOLDER / str(r[1]) / r[2]) for r in rows]


def safe_float(value: str, /) -> float | None:
    cleaned = value.strip().replace(",", ".")
    if not cleaned:
        return None
    f = float(cleaned)
    return None if f in (-9999, -9999.0) else f


def standard_format(datetime_hour: str, /, *, format_="%Y/%m/%d%H%M") -> datetime:
    datetime_hour = datetime_hour.replace("-", "/").replace("UTC", "").strip()
    if ":" in datetime_hour:
        datetime_hour = datetime_hour.replace(":", "")
    return datetime.strptime(datetime_hour, format_)


def process_single_file(mapper: Mapper, /) -> list[StationTimeseries]:
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
        with mapper.filepath.open("r") as f:
            for _ in range(9):
                f.readline()
            reader = csv.reader(f, delimiter=";")
            for row in reader:
                if not row or len(row) < len(csv_columns):
                    continue
                dict_row: dict[str, Any] = dict(zip(csv_columns, row))
                datetime_ = standard_format(
                    dict_row.pop("data") + dict_row.pop("horario")
                )
                for col in csv_float_columns:
                    dict_row[col] = safe_float(dict_row[col])
                dict_row["id_code"] = mapper.id_code
                dict_row["timestamp"] = int(datetime_.timestamp())
                dict_row["year"] = datetime_.year
                data.append(StationTimeseries(**dict_row))
    except Exception:
        LOGGER.error("Error processing file: %s", mapper.filepath, exc_info=True)
    return data


def save_data_batch(
    list_station_timeseries: list[StationTimeseries], batch_size: int = 100_000
):
    STMT = """
    INSERT INTO station_timeserie (
        id_code, timestamp, year,
        precipitacao_total_horario,
        pressao_atmosferica_ao_nivel_da_estacao_horaria,
        pressao_atmosferica_max_na_hora_ant,
        pressao_atmosferica_min_na_hora_ant,
        radiacao_global,
        temperatura_do_ar_bulbo_seco_horaria,
        temperatura_do_ponto_de_orvalho,
        temperatura_maxima_na_hora_ant,
        temperatura_minima_na_hora_ant,
        temperatura_orvalho_max_na_hora_ant,
        temperatura_orvalho_min_na_hora_ant,
        umidade_relativa_max_na_hora_ant,
        umidade_relativa_min_na_hora_ant,
        umidade_relativa_do_ar_horaria,
        vento_direcao_horaria,
        vento_rajada_maxima,
        vento_velocidade_horaria
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    """
    TEN_GBS = 1024 * 1024 * 10
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        # Performance-oriented pragmas (tune as needed)
        # WAL tends to speed up writes, and NORMAL is a good balance of safety/speed.
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        # Optional (tune or remove):
        # Use memory for temp structures and enlarge cache.
        conn.execute("PRAGMA temp_store = MEMORY;")
        # Set cache size in KiB when negative. Example: 256 MiB.
        conn.execute(f"PRAGMA cache_size = -{TEN_GBS}")

        # One transaction for the entire load
        conn.execute("BEGIN IMMEDIATE;")
        for i in range(0, len(list_station_timeseries), batch_size):
            batch = list_station_timeseries[i : i + batch_size]
            conn.executemany(STMT, batch)
            LOGGER.info(
                "Saved batch (%d/%d records)",
                i + len(batch),
                len(list_station_timeseries),
            )
        conn.commit()


def extract_and_save_in_batches(files: list[Mapper], /):
    total_files = len(files)
    processes = max((cpu_count() or 4) - 1, 2)
    batch_size_files: int = processes * 25
    LOGGER.info(
        "Starting parallel processing for %d files using %d processes",
        total_files,
        processes,
    )

    for i in range(0, total_files, batch_size_files):
        batch_files = files[i : i + batch_size_files]
        LOGGER.info(
            "Processing batch %d (%d/%d files)...",
            i // batch_size_files + 1,
            i + len(batch_files),
            total_files,
        )
        with Pool(processes=processes) as pool:
            results = pool.map(process_single_file, batch_files)
        data = [row for sub in results for row in sub]
        save_data_batch(data)
        del data, results
        gc.collect()
        remaining = total_files - (i + len(batch_files))
        LOGGER.info("Batch complete. %d files remaining.", remaining)


def setup_table():
    STMT_STATION_TIMESERIE = """
CREATE TABLE IF NOT EXISTS station_timeserie (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    id_code INTEGER NOT NULL,
    timestamp INTEGER NOT NULL,
    year INTEGER NOT NULL,
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


def main():
    setup_table()
    files = get_all_files()
    extract_and_save_in_batches(files)
    LOGGER.info("All processing completed successfully.")


if __name__ == "__main__":
    main()
