from datetime import datetime
from os import makedirs
from os.path import join, exists, getmtime
from urllib.parse import urljoin
import logging

import requests

from configs import CONFIGS


URL = "https://portal.inmet.gov.br/dadoshistoricos"


logging.basicConfig(
    filename=join(CONFIGS.LOG_FOLDER, "download_all_files.log"),
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)16d] [TID:%(thread)20d] "
        "%(module)s-%(lineno)d-%(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)

LOGGER = logging.getLogger(__file__)


def download_all_files() -> None:
    now = datetime.now()
    latest_year = now.year

    # NOTE(HSPADIM1): Baixa arquivos históricos desde 2000 até penúltimo ano
    for year in range(2000, latest_year):
        file_path = join(CONFIGS.DATA_ZIP_FOLDER, f"{year}.zip")
        if not exists(file_path):
            LOGGER.info("%s.zip not found, downloading...", year)
            _download_file(year)
        else:
            LOGGER.info("%s.zip already exists, skipping.", year)

    # NOTE(HSPADIM1): Tratar o ano mais recente
    latest_file = join(CONFIGS.DATA_ZIP_FOLDER, f"{latest_year}.zip")
    if not exists(latest_file):
        LOGGER.info(f"{latest_year}.zip not found, downloading...")
        _download_file(latest_year)
        return
    file_mtime = datetime.fromtimestamp(getmtime(latest_file))

    if file_mtime.year < latest_year or file_mtime.month == 12:
        LOGGER.info(
            "%s.zip is from %s, attempting to fetch %s.zip",
            latest_year,
            file_mtime,
            latest_year + 1,
        )
        _download_file(latest_year + 1)
        return
    elif file_mtime.month != now.month:
        LOGGER.info(
            "%s.zip is from %s, re-downloading for update.", latest_year, file_mtime
        )
        _download_file(latest_year)
        return
    else:
        LOGGER.info("%s.zip already up-to-date (%s).", latest_year, file_mtime)
        return


def _download_file(year: int, /) -> None:
    """Baixa o arquivo de um ano específico via streaming."""
    filename = f"{year}.zip"
    url = urljoin(URL, filename)
    file_path = join(CONFIGS.DATA_ZIP_FOLDER, filename)

    try:
        with requests.get(url, stream=True, timeout=60) as response:
            response.raise_for_status()
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

        LOGGER.info("Downloaded: %s", filename)
        return
    except Exception:
        LOGGER.error("Failed to download %s from %s", filename, url, exc_info=True)
        return


if __name__ == "__main__":
    makedirs(CONFIGS.DATA_ZIP_FOLDER, mode=0o755, exist_ok=True)
    download_all_files()
