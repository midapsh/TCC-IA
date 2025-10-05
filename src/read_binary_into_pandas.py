from pathlib import Path
import struct

import pandas as pd

# Binary layout: 2 ints + 17 doubles
BINARY_FORMAT = "ii" + "d" * 17
RECORD_SIZE = struct.calcsize(BINARY_FORMAT)

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
    records = []
    with open(file_path, "rb") as f:
        while chunk := f.read(RECORD_SIZE):
            record = struct.unpack(BINARY_FORMAT, chunk)
            records.append(record)
    df = pd.DataFrame(records, columns=COLUMNS)
    df.insert(0, "id_code", id_code)
    return df


def read_all_binary_files(folder: Path, /) -> pd.DataFrame:
    """Read all *.bin files in a folder into a single DataFrame."""
    dfs = [read_binary_file(p) for p in folder.glob("*.bin")]
    return (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=["id_code"] + COLUMNS)
    )


def read_binary_files_by_id(folder: Path, id_codes: list[int], /) -> pd.DataFrame:
    """Read only the binary files matching the given id_codes."""
    dfs = []
    for code in id_codes:
        pattern = f"station_data_{code}_batch.bin"
        for p in folder.glob(pattern):
            dfs.append(read_binary_file(p))
    return (
        pd.concat(dfs, ignore_index=True)
        if dfs
        else pd.DataFrame(columns=["id_code"] + COLUMNS)
    )
