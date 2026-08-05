import argparse
import os
import glob
import zipfile
import shutil
from tqdm import tqdm

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        default="hypersim",
        help="Root directory for extracted files (default: hypersim)",
    )
    parser.add_argument(
        "--subsets",
        nargs="+",
        required=True,
        help="Subsets to extract, e.g. ai_009 ai_010 ai_011",
    )
    args = parser.parse_args()

    # Extract all archives matching the selected subsets
    os.makedirs(args.output_dir, exist_ok=True)
    for subset in args.subsets:
        archives = sorted(glob.glob(os.path.join(args.output_dir, f"{subset}_*.zip")))
        if not archives:
            print(f"No archives found for {subset}")
            continue
        print(f"{subset}: found {len(archives)} archives")
        for archive in tqdm(archives, desc=f"Extracting {subset}", unit="zip"):
            with zipfile.ZipFile(archive, "r") as zf:
                zf.extractall(args.output_dir)

    # Remove all *_preview directories
    print("Removing _preview directories...")
    for root, dirs, files in os.walk(args.output_dir, topdown=False):
        for d in dirs:
            if d.endswith("_preview"):
                path = os.path.join(root, d)
                print(f"Removing {path}")
                shutil.rmtree(path)
    print("Done.")
