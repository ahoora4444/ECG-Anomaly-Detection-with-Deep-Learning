"""
Download the MIT-BIH Arrhythmia Database from PhysioNet.
Run this script once before training: python download_data.py
"""

import os
import sys
import wfdb
from config import DATA_RAW_DIR, MITBIH_RECORDS


def download_mitbih():
    os.makedirs(DATA_RAW_DIR, exist_ok=True)

    already = [f.replace('.dat', '') for f in os.listdir(DATA_RAW_DIR) if f.endswith('.dat')]
    missing = [r for r in MITBIH_RECORDS if r not in already]

    if not missing:
        print(f"All {len(MITBIH_RECORDS)} records already present in {DATA_RAW_DIR}")
        return

    print(f"Downloading {len(missing)} missing records from PhysioNet (MIT-BIH Arrhythmia DB)...")
    print("This may take a few minutes depending on your connection.\n")

    for i, record in enumerate(missing, 1):
        try:
            wfdb.dl_files('mitdb', DATA_RAW_DIR, [
                f'{record}.dat',
                f'{record}.hea',
                f'{record}.atr',
            ])
            print(f"  [{i}/{len(missing)}] {record} ✓")
        except Exception as e:
            print(f"  [{i}/{len(missing)}] {record} FAILED: {e}", file=sys.stderr)

    downloaded = [f.replace('.dat', '') for f in os.listdir(DATA_RAW_DIR) if f.endswith('.dat')]
    print(f"\nDownload complete: {len(downloaded)}/{len(MITBIH_RECORDS)} records in {DATA_RAW_DIR}")


if __name__ == '__main__':
    download_mitbih()
