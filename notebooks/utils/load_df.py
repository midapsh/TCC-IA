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
    STMT = """
    WITH estacao_bauru AS (
        SELECT id
        FROM estacoes
        WHERE nome_do_arquivo LIKE "%BAURU%"
    )
    SELECT *
    FROM medicoes a
    INNER JOIN estacao_bauru b
    ON a.id_fk = b.id;
    """

    with sqlite3.connect(DATABASE_URI) as conn:
        df = pd.read_sql_query(STMT, conn)

    # ======================
    # DATETIME CLEANING
    # ======================
    def standard_format(datetime_hour: str, /) -> str:
        datetime_hour = datetime_hour.replace("-", "/")
        datetime_hour = str(datetime_hour).replace("UTC", "").strip()
        if ":" in datetime_hour:
            return datetime_hour.replace(":", "")
        return datetime_hour

    df["datetime"] = pd.to_datetime(
        (df["data"] + df["hora"]).apply(standard_format), format="%Y/%m/%d%H%M"
    )

    # Selecionar apenas colunas úteis
    df = df[
        [
            "datetime",
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
    ]

    df.sort_values("datetime", inplace=True)

    return df


__all__ = [
    "load_df",
]
