from concurrent.futures import ProcessPoolExecutor, as_completed
from os import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd

from configs import CONFIGS

# Binary layout: 2 ints (int32) + 17 doubles (float64)
DTYPE = np.dtype([("int_fields", np.int32, 2), ("double_fields", np.float64, 17)])

COLUMNS = [
    "timestamp",
    "year",
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


def read_binary_file(file_path: Path, /) -> pd.DataFrame:
    """Read one binary file into a pandas DataFrame, extracting id_code from filename."""
    # Extract id_code from filename pattern: station_data_<id_code>_batch.bin
    id_code = int(file_path.stem.split("_")[2])

    # Read binary data using numpy
    data = np.fromfile(file_path, dtype=DTYPE)

    # Flatten the structured array into a 2D array
    int_data = data["int_fields"]
    double_data = data["double_fields"]
    records = np.column_stack([int_data, double_data])

    df = pd.DataFrame(records, columns=COLUMNS)
    df.insert(0, "id_code", id_code)
    return df


def read_all_binary_files(folder: Path, /) -> pd.DataFrame:
    """Read all *.bin files in a folder into a single DataFrame using parallel processing."""
    max_workers = max((cpu_count() or 4) - 1, 2)

    file_paths = list(folder.glob("*.bin"))

    if not file_paths:
        return pd.DataFrame(columns=["id_code"] + COLUMNS)

    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_binary_file, p): p for p in file_paths}
        for future in as_completed(futures):
            dfs.append(future.result())

    return pd.concat(dfs, ignore_index=True)


def read_binary_files_by_id(folder: Path, id_codes: list[int], /) -> pd.DataFrame:
    """Read only the binary files matching the given id_codes using parallel processing."""
    max_workers = max((cpu_count() or 4) - 1, 2)

    file_paths = []
    for code in id_codes:
        pattern = f"station_data_{code}_batch.bin"
        file_paths.extend(folder.glob(pattern))

    if not file_paths:
        return pd.DataFrame(columns=["id_code"] + COLUMNS)

    dfs = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(read_binary_file, p): p for p in file_paths}
        for future in as_completed(futures):
            dfs.append(future.result())

    return pd.concat(dfs, ignore_index=True)


def main():
    folder = CONFIGS.DATA_BINARY_FOLDER
    df = read_all_binary_files(folder)
    df.head().to_csv("text.txt")


if __name__ == "__main__":
    main()
