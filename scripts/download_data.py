"""Download UCI Online Retail II and verify it against recorded checksums.

Source:  https://archive.ics.uci.edu/dataset/502/online+retail+ii  (CC BY 4.0)
Output:  data/raw/online_retail_ii.zip and data/raw/online_retail_II.xlsx

The raw data is never committed (data/raw/ is git-ignored). Run:

    python scripts/download_data.py            # download if missing, then verify
    python scripts/download_data.py --verify   # verify an existing download only

The script exits non-zero if the download fails or a checksum does not match.
"""
import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/502/online+retail+ii.zip"
ZIP_NAME = "online_retail_ii.zip"
XLSX_NAME = "online_retail_II.xlsx"

# Recorded on 2026-10-14 (UTC) from a fresh download of the URL above.
ZIP_BYTES = 45_622_418
ZIP_SHA256 = "572e36277c2390fbfde10664750731e0a86f55e33470d91919085f0408e67bfb"
XLSX_SHA256 = "bcbe73b35f5b7babf197fb0cb983a11f5d9ff929078d4aa53d171b1f2df2e980"

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(target: Path) -> None:
    print(f"Downloading {URL}")
    request = urllib.request.Request(URL, headers={"User-Agent": "ecommerce-recommender-download/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as out:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            out.write(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--verify", action="store_true", help="verify an existing download, never fetch")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME
    xlsx_path = RAW_DIR / XLSX_NAME

    if not zip_path.exists():
        if args.verify:
            print(f"FAIL: {zip_path} not found")
            return 1
        download(zip_path)

    size = zip_path.stat().st_size
    digest = sha256(zip_path)
    print(f"zip   size={size:,}  sha256={digest}")
    if size != ZIP_BYTES or digest != ZIP_SHA256:
        print("FAIL: the zip does not match the recorded size/checksum")
        return 1

    if not xlsx_path.exists():
        with zipfile.ZipFile(zip_path) as archive:
            if XLSX_NAME not in archive.namelist():
                print(f"FAIL: {XLSX_NAME} missing from the archive")
                return 1
            with archive.open(XLSX_NAME) as src, xlsx_path.open("wb") as dst:
                dst.write(src.read())
    xlsx_digest = sha256(xlsx_path)
    print(f"xlsx  size={xlsx_path.stat().st_size:,}  sha256={xlsx_digest}")
    if xlsx_digest != XLSX_SHA256:
        print("FAIL: the extracted workbook does not match the recorded checksum")
        return 1

    print("OK: data verified (UCI Online Retail II, CC BY 4.0)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
