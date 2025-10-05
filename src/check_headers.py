"""
Make sure that headers are ok
"""

from pathlib import Path
from typing import NamedTuple
import logging
import sqlite3

from configs import CONFIGS


logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("check_headers.log"),
    level=logging.DEBUG,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)16d] [TID:%(thread)20d] "
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
    data = None
    with sqlite3.connect(CONFIGS.DATABASE_URI) as conn:
        cursor = conn.execute(STMT)
        rows = cursor.fetchall()
        data = [
            Mapper(row[0], CONFIGS.DATA_CSV_FOLDER / str(row[1]) / row[2])
            for row in rows
        ]
        return data


def extract_data(list_mapper: list[Mapper], /) -> list[str]:
    data: set[str] = set()
    for mapper in list_mapper:
        with mapper.filepath.open(mode="r") as f:
            # Skip metadata
            for _ in range(8):
                f.readline()
            header = f.readline()
            data.add(header)
    return list(data)


def save_stuff(data: list[str], /) -> None:
    with open("headers.txt", "w") as f:
        f.writelines(data)


def main() -> None:
    stuff = get_all_files()
    data = extract_data(stuff)
    save_stuff(data)

    return


if __name__ == "__main__":
    main()
