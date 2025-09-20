from os import makedirs
from pathlib import Path
from zipfile import ZipFile
import logging

from configs import CONFIGS


logging.basicConfig(
    filename=CONFIGS.LOG_FOLDER.joinpath("unzip_all_files.log"),
    level=logging.INFO,
    format=(
        "%(asctime)s.%(msecs)03d [%(levelname)-8s] "
        "[PID:%(process)16d] [TID:%(thread)20d] "
        "%(module)s-%(lineno)d-%(name)s: %(message)s"
    ),
    datefmt="%Y-%m-%dT%H:%M:%S",
)

LOGGER = logging.getLogger(__file__)


def unzip_all_files() -> None:
    csv_folder = CONFIGS.DATA_CSV_FOLDER

    for zip_file in CONFIGS.DATA_ZIP_FOLDER.iterdir():
        if not zip_file.is_file() or not zip_file.suffix.lower().endswith("zip"):
            continue  # pula coisas que não são .zip

        csv_folder_by_year = csv_folder.joinpath(zip_file.stem)
        makedirs(csv_folder_by_year, exist_ok=True)

        try:
            with ZipFile(zip_file, "r") as zf:
                for file_in_zip in zf.namelist():
                    if file_in_zip.endswith("/"):
                        continue  # ignora diretórios

                    target_path = csv_folder_by_year.joinpath(Path(file_in_zip).name)

                    with zf.open(file_in_zip) as src, open(target_path, "w") as dst:
                        dst.write(src.read().decode("ISO-8859-1"))

                    LOGGER.info("Extracted %s -> %s", file_in_zip, target_path)
        except Exception:
            LOGGER.error("Failed to unzip %s", zip_file, exc_info=True)


if __name__ == "__main__":
    makedirs(CONFIGS.DATA_CSV_FOLDER, exist_ok=True)
    unzip_all_files()
