import sqlite3
import csv
import re
import logging
from pathlib import Path

from configs import CONFIGS

# =====================
# LOGGING
# =====================
logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("csv_into_db.log"),
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)16d] [TID:%(thread)20d] "
        "%(module)s-%(lineno)d-%(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)
LOGGER = logging.getLogger(__file__)


# =====================
# CRIAÇÃO DO BANCO
# =====================
def create_db(db_file: Path, /) -> sqlite3.Connection:
    LOGGER.info("Criando banco de dados em %s", db_file)
    conn = sqlite3.connect(db_file)
    cur = conn.cursor()

    try:
        # Dropar tabelas antigas
        cur.execute("DROP TABLE IF EXISTS variaveis")
        cur.execute("DROP TABLE IF EXISTS estacoes")
        cur.execute("DROP TABLE IF EXISTS medicoes")

        # Tabelas
        cur.execute(
            """
        CREATE TABLE variaveis (
            coluna TEXT,
            unidade_de_medida TEXT
        )
        """
        )
        cur.execute(
            """
        CREATE TABLE estacoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ano INTEGER,
            nome_do_arquivo TEXT,
            regiao TEXT,
            uf TEXT,
            estacao TEXT,
            codigo TEXT,
            latitude REAL,
            longitude REAL,
            altitude REAL,
            data_de_fundacao TEXT
        )
        """
        )
        cur.execute(
            """
        CREATE TABLE medicoes (
            id_fk INTEGER,
            data TEXT,
            hora TEXT,
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
            FOREIGN KEY(id_fk) REFERENCES estacoes(id)
        )
        """
        )

        # Índices
        cur.execute("CREATE INDEX idx_medicoes_data ON medicoes(data)")
        cur.execute("CREATE INDEX idx_medicoes_hora ON medicoes(hora)")
        cur.execute("CREATE INDEX idx_medicoes_hora_data ON medicoes(hora, data)")

        cur.execute("CREATE INDEX idx_estacoes_ano ON estacoes(ano)")
        cur.execute(
            "CREATE INDEX idx_estacoes_nome_do_arquivo ON estacoes(nome_do_arquivo)"
        )
        cur.execute("CREATE INDEX idx_estacoes_regiao ON estacoes(regiao)")
        cur.execute("CREATE INDEX idx_estacoes_uf ON estacoes(uf)")
        cur.execute("CREATE INDEX idx_estacoes_estacao ON estacoes(estacao)")
        cur.execute("CREATE INDEX idx_estacoes_codigo ON estacoes(codigo)")
        cur.execute("CREATE INDEX idx_estacoes_latitude ON estacoes(latitude)")
        cur.execute("CREATE INDEX idx_estacoes_longitude ON estacoes(longitude)")
        cur.execute("CREATE INDEX idx_estacoes_altitude ON estacoes(altitude)")
        cur.execute(
            "CREATE INDEX idx_estacoes_data_de_fundacao ON estacoes(data_de_fundacao)"
        )

        conn.commit()
        LOGGER.info("Banco de dados criado com sucesso.")
    except Exception as e:
        LOGGER.exception("Erro ao criar banco de dados: %s", e)
        raise

    return conn


# =====================
# PROCESSAR UM CSV


# =====================
def process_csv_file(csv_file: Path, conn: sqlite3.Connection, /) -> None:
    LOGGER.info("Processando arquivo %s", csv_file)
    cur = conn.cursor()

    try:
        with open(csv_file) as f:
            reader = csv.reader(f, delimiter=";")
            rows = list(reader)

        # Metadados da estação
        meta = {row[0].replace(":", "").strip(): row[1].strip() for row in rows[:8]}
        header = rows[8]

        # Extrair ano do nome do arquivo
        match = re.search(r"_(\d{2})-(\d{2})-(\d{4})_", csv_file.name)
        ano = int(match.group(3)) if match else None

        # Inserir estação
        LOGGER.info("Insert estacao")
        cur.execute(
            """
        INSERT INTO estacoes (
            ano, nome_do_arquivo, regiao, uf, estacao, codigo,
            latitude, longitude, altitude, data_de_fundacao
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                ano,
                csv_file.name,
                meta.get("REGIÃO"),
                meta.get("UF"),
                meta.get("ESTAÇÃO"),
                meta.get("CODIGO (WMO)"),
                float(meta["LATITUDE"].replace(",", ".")),
                float(meta["LONGITUDE"].replace(",", ".")),
                float(meta["ALTITUDE"].replace(",", ".")),
                meta.get("DATA DE FUNDAÇÃO (YYYY-MM-DD)"),
            ),
        )

        estacao_id = cur.lastrowid

        # Inserir variáveis
        LOGGER.info("Insert variaveis")
        for col in header:
            if "(" in col:
                nome, unidade = col.split("(", 1)
                unidade = unidade.strip(") ")
            else:
                nome, unidade = col, None
            cur.execute(
                "INSERT INTO variaveis (coluna, unidade_de_medida) VALUES (?, ?)",
                (nome.strip(), unidade),
            )

        # Inserir medições
        LOGGER.info("Insert medicoes")
        for row in rows[9:]:
            if not row or len(row) < 2:
                continue
            values = [None if v in ("", "-9999") else v.replace(",", ".") for v in row]

            # fix: garantir só 19 colunas (data + hora + 17 medições)
            values = values[:19]

            cur.execute(
                """
            INSERT INTO medicoes VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
            """,
                [estacao_id] + values,
            )

        conn.commit()
        LOGGER.info("Arquivo %s importado com sucesso.", csv_file)
        return
    except Exception:
        LOGGER.exception("Erro ao processar arquivo %s", csv_file)
        return


# =====================
# PROCESSAR VÁRIOS CSVs
# =====================
def process_all_csvs(db_file: Path, /) -> None:
    conn = create_db(db_file)

    root = CONFIGS.DATA_CSV_FOLDER
    if not root.exists():
        LOGGER.error("Pasta de CSVs não encontrada: %s", root)
        conn.close()
        return

    # percorre subpastas (ex.: 2000, 2001, ...)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        LOGGER.info("📂 Processing folder: %s", folder)
        # percorre os arquivos da pasta e filtra por extensão (case-insensitive)
        for csv_path in sorted(folder.iterdir()):
            if not csv_path.is_file():
                continue
            if csv_path.suffix.lower() != ".csv":
                LOGGER.debug("Ignorando (não-CSV): %s", csv_path.name)
                continue

            LOGGER.info("   - Encontrado CSV: %s", csv_path.name)
            process_csv_file(csv_path, conn)

    conn.close()
    LOGGER.info("✅ Importação concluída em %s", db_file)


if __name__ == "__main__":
    try:
        LOGGER.info("🚀 Iniciando importação de CSVs em %s", CONFIGS.DATA_CSV_FOLDER)
        process_all_csvs(CONFIGS.DATABASE_URI)
    except Exception:
        LOGGER.exception("💥 Erro durante a importação")
        raise
