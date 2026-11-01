
"""Download the official UCI Bike Sharing dataset."""

import ssl
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import certifi

from src.config import RAW_CSV

UCI_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"


def main() -> None:
    if RAW_CSV.exists():
        print(f"Dataset already exists: {RAW_CSV}")
        return

    context = ssl.create_default_context(cafile=certifi.where())

    with urlopen(UCI_URL, timeout=45, context=context) as response:
        archive_bytes = response.read()

    with ZipFile(BytesIO(archive_bytes)) as archive:
        matches = [name for name in archive.namelist() if name.endswith("hour.csv")]

        if len(matches) != 1:
            raise RuntimeError(f"Expected one hour.csv in UCI archive, got {matches}")

        RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
        RAW_CSV.write_bytes(archive.read(matches[0]))

    print(f"Saved original UCI hourly CSV to {RAW_CSV}")


if __name__ == "__main__":
    main()
