import argparse
import sys
import zipfile
from pathlib import Path


def extract_zip(zip_path: Path, dest: Path):
    if not zip_path.exists():
        print(f"Error: {zip_path} not found")
        return False

    print(f"Extracting: {zip_path}")
    print(f"Destination: {dest.resolve()}")

    dest.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        total = len(members)
        for i, member in enumerate(members, 1):
            zf.extract(member, dest)
            if i % 500 == 0 or i == total:
                print(f"\r  Progress: {i}/{total}", end="", flush=True)

    print("\nDone!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Extract PTDB_TUG dataset")
    parser.add_argument("--zip", default="datasets/PTDB_TUG.zip", help="Zip file path")
    parser.add_argument("--output", default="datasets", help="Extract destination")
    args = parser.parse_args()

    success = extract_zip(Path(args.zip), Path(args.output))
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
