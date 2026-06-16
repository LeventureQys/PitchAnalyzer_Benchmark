import argparse
import sys
from pathlib import Path

import requests
from tqdm import tqdm


def download_file(url: str, dest: Path, chunk_size: int = 1024 * 1024, retries: int = 3):
    dest.parent.mkdir(parents=True, exist_ok=True)
    temp_dest = dest.with_suffix(dest.suffix + ".tmp")

    for attempt in range(1, retries + 1):
        try:
            existing_size = temp_dest.stat().st_size if temp_dest.exists() else 0

            headers = {}
            if existing_size > 0:
                headers["Range"] = f"bytes={existing_size}-"

            with requests.get(url, headers=headers, stream=True, timeout=30) as resp:
                if resp.status_code == 416:
                    print(f"File already fully downloaded: {dest}")
                    temp_dest.rename(dest)
                    return True

                if resp.status_code not in (200, 206):
                    print(f"Unexpected status code: {resp.status_code}")
                    if attempt < retries:
                        print(f"Retrying ({attempt}/{retries})...")
                        continue
                    return False

                server_supports_range = resp.status_code == 206
                if existing_size > 0 and not server_supports_range:
                    print("Server does not support resume, restarting download...")
                    existing_size = 0

                total = int(resp.headers.get("content-length", 0))
                if server_supports_range and "content-range" in resp.headers:
                    total = int(resp.headers["content-range"].split("/")[-1])

                mode = "ab" if server_supports_range and existing_size > 0 else "wb"
                downloaded = existing_size

                with open(temp_dest, mode) as f:
                    with tqdm(
                        total=total,
                        initial=downloaded,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name,
                    ) as pbar:
                        for chunk in resp.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                pbar.update(len(chunk))

            temp_dest.rename(dest)
            print(f"Download complete: {dest}")
            return True

        except (requests.exceptions.RequestException, IOError) as e:
            print(f"\nError during download: {e}")
            if attempt < retries:
                print(f"Retrying ({attempt}/{retries})...")
            else:
                print("Max retries reached. You can re-run the script to resume.")
                return False

    return False


def main():
    parser = argparse.ArgumentParser(description="Download PTDB_TUG dataset from Zenodo")
    parser.add_argument(
        "--url",
        default="https://zenodo.org/records/3921794/files/PTDB_TUG.zip?download=1",
        help="Download URL",
    )
    parser.add_argument(
        "--output",
        default="datasets/PTDB_TUG.zip",
        help="Output file path (default: datasets/PTDB_TUG.zip)",
    )
    parser.add_argument("--chunk-size", type=int, default=1048576, help="Chunk size in bytes")
    parser.add_argument("--retries", type=int, default=3, help="Max retry attempts")
    args = parser.parse_args()

    dest = Path(args.output)
    print(f"Downloading: {args.url}")
    print(f"Saving to:   {dest.resolve()}")

    success = download_file(args.url, dest, chunk_size=args.chunk_size, retries=args.retries)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
