#!/usr/bin/env python3
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve()
for parent in _repo_root.parents:
    if (parent / ".gitignore").exists():
        _repo_root = parent
        break
else:
    _repo_root = _repo_root.parent

_repo_root_str = str(_repo_root)
if _repo_root_str not in sys.path:
    sys.path.append(_repo_root_str)

import sys
import os
import pandas as pd

def main():
    if len(sys.argv) != 3:
        print("Usage: python remove_offset.py <csv_path> <offset>")
        sys.exit(1)

    in_path = sys.argv[1]
    try:
        offset = float(sys.argv[2])
    except ValueError:
        print(f"Error: offset must be a number, got '{sys.argv[2]}'")
        sys.exit(1)

    if not os.path.isfile(in_path):
        print(f"Error: file not found: {in_path}")
        sys.exit(1)

    df = pd.read_csv(in_path)

    target_cols = ["minE", "maxE", "meanE"]
    missing = [c for c in target_cols if c not in df.columns]
    if missing:
        print(f"Error: CSV is missing required columns: {', '.join(missing)}")
        sys.exit(1)

    # Ensure numeric and subtract offset
    for c in target_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce") - offset

    root, ext = os.path.splitext(in_path)
    out_path = f"{root}{ext or '.csv'}"
    df.to_csv(out_path, index=False)
    print(f"Offset {offset} removed from {', '.join(target_cols)}.")
    print(f"Wrote: {out_path}")

if __name__ == "__main__":
    main()
