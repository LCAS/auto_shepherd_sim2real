"""Download and extract configured simulation and real-world datasets."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from urllib.parse import urlparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add one or more direct ZIP download URLs to each list when the dataset
# locations are available. The lists are intentionally independent.
DATASET_URLS: dict[str, tuple[str, ...]] = {
    "sim": (),
    "real": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and extract the configured Sim2Real datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=("sim", "real", "all"),
        default="all",
        help="Dataset to download (default: all).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Directory containing the sim/ and real/ output directories.",
    )
    return parser.parse_args()


def archive_name(url: str, dataset: str, index: int) -> str:
    name = Path(urlparse(url).path).name
    return name if name else f"{dataset}_{index}.zip"


def extract_zip(archive_path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    resolved_destination = destination.resolve()

    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            member_path = (destination / member.filename).resolve()
            try:
                member_path.relative_to(resolved_destination)
            except ValueError as exc:
                raise RuntimeError(
                    f"Refusing to extract path outside {destination}: "
                    f"{member.filename}"
                ) from exc
        archive.extractall(destination)


def download_archive(url: str, destination: Path, dataset: str, index: int) -> None:
    print(f"Downloading {url}")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "AutoShepherd-Sim2Real-dataset-downloader"},
    )

    with tempfile.TemporaryDirectory(prefix="autoshepherd-download-") as temp_dir:
        archive_path = Path(temp_dir) / archive_name(url, dataset, index)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with archive_path.open("wb") as output_file:
                    shutil.copyfileobj(response, output_file)
            extract_zip(archive_path, destination)
        except (OSError, urllib.error.URLError, zipfile.BadZipFile) as exc:
            raise RuntimeError(
                f"Could not download or extract {url} for {dataset}: {exc}"
            ) from exc


def download_dataset(dataset: str, output_root: Path) -> None:
    urls = DATASET_URLS[dataset]
    if not urls:
        raise RuntimeError(
            f"No download URLs configured for {dataset}. "
            "Add direct ZIP URLs to DATASET_URLS in src/download_data.py."
        )

    destination = output_root / dataset
    for index, url in enumerate(urls, start=1):
        download_archive(url, destination, dataset, index)
    print(f"Extracted {len(urls)} archive(s) into {destination}")


def main() -> None:
    args = parse_args()
    datasets = ("sim", "real") if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        download_dataset(dataset, args.output_root)


if __name__ == "__main__":
    main()
