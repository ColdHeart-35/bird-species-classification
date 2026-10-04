"""Download and safely extract the CUB-200-2011 dataset archive."""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

# Official CaltechDATA download endpoint (the publisher may redirect this URL).
URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    """Extract only members whose resolved path remains inside destination."""
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if not target.is_relative_to(root):
            raise RuntimeError(f"Unsafe archive path: {member.name}")
    # Validation above prevents path traversal; omit newer ``filter=`` support
    # so this script also runs on the documented Python 3.10 minimum.
    archive.extractall(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download CUB-200-2011")
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--url", default=URL, help="Dataset archive URL")
    args = parser.parse_args()
    target = args.output_dir / "CUB_200_2011"
    if target.exists():
        print(f"Dataset already exists: {target}")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = args.output_dir / "CUB_200_2011.tgz"
    print("Downloading CUB-200-2011 (this may take several minutes)...")
    urllib.request.urlretrieve(args.url, archive_path)
    print("Extracting archive...")
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, args.output_dir)
    archive_path.unlink()
    print(f"Ready: {target}")


if __name__ == "__main__":
    main()
