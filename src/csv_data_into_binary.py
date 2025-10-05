from datetime import datetime
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
from typing import Any, NamedTuple
import csv
import gc
import logging
import sqlite3
import struct

from configs import CONFIGS


logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("csv_data_into_binary.log"),
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
    precipitacao_total_horario: float  # can be nan
    pressao_atmosferica_ao_nivel_da_estacao_horaria: float  # can be nan
    pressao_atmosferica_max_na_hora_ant: float  # can be nan
    pressao_atmosferica_min_na_hora_ant: float  # can be nan
    radiacao_global: float  # can be nan
    temperatura_do_ar_bulbo_seco_horaria: float  # can be nan
    temperatura_do_ponto_de_orvalho: float  # can be nan
    temperatura_maxima_na_hora_ant: float  # can be nan
    temperatura_minima_na_hora_ant: float  # can be nan
    temperatura_orvalho_max_na_hora_ant: float  # can be nan
    temperatura_orvalho_min_na_hora_ant: float  # can be nan
    umidade_relativa_max_na_hora_ant: float  # can be nan
    umidade_relativa_min_na_hora_ant: float  # can be nan
    umidade_relativa_do_ar_horaria: float  # can be nan
    vento_direcao_horaria: float  # can be nan
    vento_rajada_maxima: float  # can be nan
    vento_velocidade_horaria: float  # can be nan


# Binary format: 2 integers (i) + 17 doubles (d)
# Total: 2*4 + 17*8 = 144 bytes per record
BINARY_FORMAT = "ii" + "d" * 17
RECORD_SIZE = struct.calcsize(BINARY_FORMAT)


def get_all_files() -> list[Mapper]:
    STMT = "SELECT id_code, year, filename FROM station_metadata;"
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        cursor = conn.execute(STMT)
        rows = cursor.fetchall()
    return [Mapper(r[0], CONFIGS.DATA_CSV_FOLDER / str(r[1]) / r[2]) for r in rows]


def safe_float(value: str, /) -> float:
    cleaned = value.strip().replace(",", ".")
    if not cleaned:
        return float("nan")
    f = float(cleaned)
    return float("nan") if f in (-9999, -9999.0) else f


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


def save_single_id_code_data(
    args: tuple[int, list[StationTimeseries]], /
) -> tuple[int, int]:
    """Save data for a single id_code to a binary file."""
    id_code, records = args
    output_dir = CONFIGS.DATA_BINARY_FOLDER
    output_dir.mkdir(exist_ok=True)

    binary_file = output_dir / f"station_data_{id_code}_batch.bin"

    with open(binary_file, "ab") as f:
        for record in records:
            binary_data = struct.pack(BINARY_FORMAT, *record[1:])
            f.write(binary_data)

    LOGGER.info("Saved %d records to %s", len(records), binary_file.name)
    return id_code, len(records)


def save_data_batch(list_station_timeseries: list[StationTimeseries], /):
    """Save data to binary files using multiprocessing, splitting by id_code."""
    data_by_id_code: dict[int, list[StationTimeseries]] = {}
    for record in list_station_timeseries:
        id_code = record.id_code
        if id_code not in data_by_id_code:
            data_by_id_code[id_code] = []
        data_by_id_code[id_code].append(record)

    save_args = list(data_by_id_code.items())

    processes = max((cpu_count() or 4) - 1, 2)
    LOGGER.info(
        "Saving data for %d id_codes using %d processes",
        len(save_args),
        processes,
    )

    with Pool(processes=processes) as pool:
        results = pool.map(save_single_id_code_data, save_args)

    total_records = sum(count for _, count in results)
    LOGGER.info(
        "Saved total of %d records across %d files", total_records, len(results)
    )


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


def create_metadata_file():
    """Create a metadata file describing the binary format."""
    output_dir = CONFIGS.DATA_BINARY_FOLDER
    output_dir.mkdir(exist_ok=True)

    metadata = f"""Binary Data Format Specification
====

Record Size: {RECORD_SIZE} bytes
Format String: {BINARY_FORMAT}

Field Layout:
----
Offset | Type    | Size | Field Name
----|----|----|----
0    | int32   | 4    | id_code
4    | int32   | 4    | timestamp
8    | int32   | 4    | year
12    | float64 | 8    | precipitacao_total_horario
20    | float64 | 8    | pressao_atmosferica_ao_nivel_da_estacao_horaria
28    | float64 | 8    | pressao_atmosferica_max_na_hora_ant
36    | float64 | 8    | pressao_atmosferica_min_na_hora_ant
44    | float64 | 8    | radiacao_global
52    | float64 | 8    | temperatura_do_ar_bulbo_seco_horaria
60    | float64 | 8    | temperatura_do_ponto_de_orvalho
68    | float64 | 8    | temperatura_maxima_na_hora_ant
76    | float64 | 8    | temperatura_minima_na_hora_ant
84    | float64 | 8    | temperatura_orvalho_max_na_hora_ant
92    | float64 | 8    | temperatura_orvalho_min_na_hora_ant
100    | float64 | 8    | umidade_relativa_max_na_hora_ant
108    | float64 | 8    | umidade_relativa_min_na_hora_ant
116    | float64 | 8    | umidade_relativa_do_ar_horaria
124    | float64 | 8    | vento_direcao_horaria
132    | float64 | 8    | vento_rajada_maxima
140    | float64 | 8    | vento_velocidade_horaria

Notes:
----
- All integers are 32-bit signed (little-endian)
- All floats are 64-bit IEEE 754 doubles (little-endian)
- NULL values are represented as NaN for float fields
- Files are named: station_data_<YEAR>_batch_<BATCH_NUMBER>.bin

Reading Example (Python):
----
import struct

BINARY_FORMAT = "{BINARY_FORMAT}"
RECORD_SIZE = {RECORD_SIZE}

with open("station_data_2020_batch_0000.bin", "rb") as f:
    while True:
        data = f.read(RECORD_SIZE)
        if not data:
            break
        record = struct.unpack(BINARY_FORMAT, data)
        # record[0] = id_code
        # record[1] = timestamp
        # record[2] = year
        # record[3:20] = float values
"""

    metadata_file = output_dir / "FORMAT_SPECIFICATION.txt"
    with open(metadata_file, "w") as f:
        f.write(metadata)

    LOGGER.info("Metadata file created: %s", metadata_file)


def main():
    create_metadata_file()
    files = get_all_files()
    extract_and_save_in_batches(files)
    LOGGER.info("All processing completed successfully.")


if __name__ == "__main__":
    main()
