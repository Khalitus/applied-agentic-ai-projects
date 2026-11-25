"""Retrieve the official UCI bank-full.csv. No third-party download package required."""

import io
import urllib.request
import zipfile

from src.config import RAW_CSV

URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"


def main():
    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    if RAW_CSV.exists():
        print(f"Already available: {RAW_CSV}")
        return

    request = urllib.request.Request(URL, headers={"User-Agent": "campaign-intelligence-agent/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=45) as response:
            archive_bytes = response.read()
        with zipfile.ZipFile(io.BytesIO(archive_bytes)) as outer:
            paths = [n for n in outer.namelist() if n.endswith("bank-full.csv")]
            if paths:
                contents = outer.read(paths[0])
            else:
                nested = [n for n in outer.namelist() if n.endswith("bank.zip")]
                if not nested:
                    raise ValueError("The UCI archive layout has changed")
                with zipfile.ZipFile(io.BytesIO(outer.read(nested[0]))) as inner:
                    names = [n for n in inner.namelist() if n.endswith("bank-full.csv")]
                    if not names:
                        raise ValueError("bank-full.csv was not found in the archive")
                    contents = inner.read(names[0])
        if not contents.startswith(b'"age"'):
            raise ValueError("Unexpected source file header; please check download")
        RAW_CSV.write_bytes(contents)
        print(f"Downloaded the official UCI Bank Marketing CSV to: {RAW_CSV}")
    except Exception as exc:
        raise SystemExit(
            f"Official download failed: {exc}\n"
            "Manual fallback: download bank.zip from https://archive.ics.uci.edu/dataset/222/bank+marketing\n"
            f"Extract bank-full.csv into: {RAW_CSV.parent}"
        ) from exc


if __name__ == "__main__":
    main()
