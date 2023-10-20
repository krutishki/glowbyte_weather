import gzip
import os
import shutil

import requests

CITY = 26702  # https://rp5.ru/%D0%90%D1%80%D1%85%D0%B8%D0%B2_%D0%BF%D0%BE%D0%B3%D0%BE%D0%B4%D1%8B_%D0%B2_%D0%9A%D0%B0%D0%BB%D0%B8%D0%BD%D0%B8%D0%BD%D0%B3%D1%80%D0%B0%D0%B4%D0%B5
START_DATE = "01.01.2019"
END_DATE = "30.09.2023"
FILENAME = "weather"
WEATHER_URL = f"https://ru5.rp5.ru/download/files.synop/26/{CITY}.{START_DATE}.{END_DATE}.1.0.0.ru.utf8.00000000.csv.gz"


def main() -> None:
    download_weather()
    unzip_weather()
    cleanup()


def download_weather() -> None:
    os.makedirs("data/tmp", exist_ok=True)

    with open(f"data/tmp/{FILENAME}.csv.gz", "wb") as f:
        r = requests.get(WEATHER_URL)
        f.write(r.content)


def unzip_weather() -> None:
    with gzip.open(f"data/tmp/{FILENAME}.csv.gz", "rb") as f_in:
        with open(f"data/{FILENAME}.csv", "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)


def cleanup() -> None:
    os.system("rm -rf data/tmp")


if __name__ == "__main__":
    main()
