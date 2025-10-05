from pathlib import Path
import sqlite3

import pandas as pd

# ======================
# CONFIG
# ======================
DATA_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/data")
HTML_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/html")
IMAGES_FOLDER = Path("/home/dolores/Documents/matheus-ferreira/TCC-IA/images")
DATABASE_URI = str(DATA_FOLDER / "database.db")
# In-memory db
# DATABASE_URI = "/mnt/ramdisk/database.db"


# ======================
# SQL QUERY
# ======================


def load_df() -> pd.DataFrame:
    stmt = """
    SELECT *
    FROM station_timeserie a
    where id_code = (
        SELECT id_code
        FROM station_metadata
        WHERE name LIKE "%BAURU%"
    );
    """
    # Columns to load (avoid SELECT * for performance and memory)
    cols = [
        "id_code",
        "timestamp",
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

    # stmt = f"""
    # SELECT {", ".join(cols)}
    # FROM station_timeserie
    # """

    # Optional: downcast floats to save memory; adjust as needed
    dtype_map = {
        "id_code": "int64",
        "timestamp": "int64",  # parsed separately as datetime; keep as int on read
        "precipitacao_total_horario": "float32",
        "pressao_atmosferica_ao_nivel_da_estacao_horaria": "float32",
        "pressao_atmosferica_max_na_hora_ant": "float32",
        "pressao_atmosferica_min_na_hora_ant": "float32",
        "radiacao_global": "float32",
        "temperatura_do_ar_bulbo_seco_horaria": "float32",
        "temperatura_do_ponto_de_orvalho": "float32",
        "temperatura_maxima_na_hora_ant": "float32",
        "temperatura_minima_na_hora_ant": "float32",
        "temperatura_orvalho_max_na_hora_ant": "float32",
        "temperatura_orvalho_min_na_hora_ant": "float32",
        "umidade_relativa_max_na_hora_ant": "float32",
        "umidade_relativa_min_na_hora_ant": "float32",
        "umidade_relativa_do_ar_horaria": "float32",
        "vento_direcao_horaria": "float32",
        "vento_rajada_maxima": "float32",
        "vento_velocidade_horaria": "float32",
    }

    with sqlite3.connect(DATABASE_URI) as conn:
        # Optional read-optimized pragmas
        # try:
        #     conn.execute("PRAGMA journal_mode=WAL;")
        #     conn.execute("PRAGMA synchronous = NORMAL;")
        # except Exception:
        #     pass

        # Read only selected columns; keep timestamp as integer for precise conversion
        df = pd.read_sql_query(stmt, conn, dtype=dtype_map)

    # Ensure expected columns exist (defensive)
    missing = set(cols) - set(df.columns)
    if missing:
        raise KeyError(f"Missing expected column(s) in query result: {missing}")

    # Sort by timestamp before conversion (faster on int)
    df.sort_values("timestamp", inplace=True, kind="mergesort", ignore_index=True)

    # Convert epoch seconds to pandas datetime (UTC assumed; adjust if needed)
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="s")

    # Drop raw timestamp column
    df.drop(columns=["timestamp"], inplace=True)

    return df


__all__ = [
    "load_df",
]
