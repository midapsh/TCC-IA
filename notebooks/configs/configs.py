from dataclasses import dataclass, field
from os import getenv
from pathlib import Path


@dataclass(frozen=True, kw_only=True, slots=True)
class Configs:
    DATA_ZIP_FOLDER: Path = field(
        default_factory=lambda: Path(
            getenv(
                "DATA_ZIP_FOLDER",
                "/home/dolores/Documents/matheus-ferreira/TCC-IA/data/zip_files",
            )
        )
    )
    DATA_CSV_FOLDER: Path = field(
        default_factory=lambda: Path(
            getenv(
                "DATA_ZIP_FOLDER",
                "/home/dolores/Documents/matheus-ferreira/TCC-IA/data/csv_files",
            )
        )
    )
    LOG_FOLDER: Path = field(
        default_factory=lambda: Path(
            getenv("LOG_FOLDER", "/home/dolores/Documents/matheus-ferreira/TCC-IA/logs")
        )
    )
    DATABASE_URI: Path = field(
        default_factory=lambda: Path(
            getenv(
                "DATABASE_URI",
                "/home/dolores/Documents/matheus-ferreira/TCC-IA/data/database.db",
            )
        )
    )
    DUCKDB_DATABASE_URI: Path = field(
        default_factory=lambda: Path(
            getenv(
                "DUCKDB_DATABASE_URI",
                "/home/dolores/Documents/matheus-ferreira/TCC-IA/data/database.duckdb",
            )
        )
    )


CONFIGS = Configs()

__all__ = [
    "CONFIGS",
]
