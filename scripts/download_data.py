"""Download the public EPF benchmark files into data/raw/epf/.

    python scripts/download_data.py            # official epftoolbox files (Zenodo 4624805)
    python scripts/download_data.py --mirror   # also the price-only public mirror

ISO-NE and Tetouan files must be placed manually; see data/README.md.
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

EPF_URL = "https://zenodo.org/records/4624805/files/{market}.csv"
MIRROR_URL = ("https://raw.githubusercontent.com/Nixtla/transfer-learning-time-series/"
              "main/datasets/electricity.csv")
MARKETS = ("PJM", "NP", "DE", "BE", "FR")
TARGET = Path(__file__).resolve().parents[1] / "data" / "raw" / "epf"


def fetch(url: str, path: Path) -> None:
    if path.exists():
        print(f"exists   {path.name}")
    else:
        urllib.request.urlretrieve(url, path)
        print(f"fetched  {path.name}")
    print(f"         sha256 {hashlib.sha256(path.read_bytes()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mirror", action="store_true", help="also fetch the price-only mirror")
    args = parser.parse_args()
    TARGET.mkdir(parents=True, exist_ok=True)
    for market in MARKETS:
        fetch(EPF_URL.format(market=market), TARGET / f"{market}.csv")
    if args.mirror:
        fetch(MIRROR_URL, TARGET / "nixtla_electricity_mirror.csv")


if __name__ == "__main__":
    main()
