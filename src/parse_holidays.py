import csv
import os
from typing import Any
from xml.etree import ElementTree

import requests

START_YEAR = 2019
END_YEAR = 2024
CALENDAR_URL = "https://xmlcalendar.ru/data/ru/{}/calendar.xml"


def main() -> None:
    download_calendars()
    make_holidays_csv()
    cleanup_calendars()


def download_calendars() -> None:
    os.makedirs("data/tmp", exist_ok=True)

    for year in range(START_YEAR, END_YEAR):
        r = requests.get(CALENDAR_URL.format(year))
        with open(f"data/tmp/{year}.xml", "wb") as f:
            f.write(r.content)


def cleanup_calendars() -> None:
    os.system("rm -rf data/tmp")


def make_holidays_csv() -> None:
    fieldnames = ["day", "type", "holiday", "from"]
    file = open("data/holidays.csv", "w")
    writer = csv.DictWriter(file, fieldnames=fieldnames)

    writer.writeheader()
    for year in range(START_YEAR, END_YEAR):
        holidays = parse_xml(f"data/tmp/{year}.xml")
        writer.writerows(holidays)

    file.close()


def parse_xml(file: str) -> list[dict[str, Any]]:
    tree = ElementTree.parse(file)
    root = tree.getroot()
    year = root.attrib["year"]

    holidays = {holiday.attrib["id"]: holiday.attrib["title"] for holiday in root.iter("holiday")}

    days: list[dict[str, Any]] = []
    for day in root.iter("day"):
        dt = f'{year}.{day.attrib["d"]}'
        from_day = f'{year}.{day.attrib["f"]}' if day.attrib.get("f") else None
        holiday = holidays.get(day.attrib.get("h"))
        days.append({"day": dt, "type": day.attrib["t"], "holiday": holiday, "from": from_day})

    return days


if __name__ == "__main__":
    main()
