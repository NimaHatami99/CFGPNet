#!/usr/bin/env python3
"""
Strip the 6th number from every line of each .txt file in a source directory,
writing files with the same names into a destination directory.

Usage:
    python strip_6th_column.py /path/to/source_dir /path/to/dest_dir
"""

import argparse
from pathlib import Path

def process_file(src_path: Path, dst_path: Path) -> None:
    """Read src_path, remove the 6th whitespace-separated token on each non-empty line,
    and write the result to dst_path. The source file remains unchanged.
    """
    lines_out = []
    with src_path.open('r', encoding='utf-8', errors='replace') as f:
        for raw in f:
            # Preserve completely blank lines
            if raw.strip() == '':
                lines_out.append(raw)
                continue

            # Split by any whitespace; remove the 6th token if present
            parts = raw.strip().split()
            if len(parts) >= 6:
                parts.pop(5)  # remove the 6th item (index 5)
                lines_out.append(' '.join(parts) + '\n')
            else:
                # If a line unexpectedly has fewer than 6 items, leave it unchanged
                lines_out.append(raw)

    # Ensure destination directory exists
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with dst_path.open('w', encoding='utf-8', errors='strict') as f:
        f.writelines(lines_out)

def main():
    p = argparse.ArgumentParser(description="Copy .txt files from source to destination while removing the 6th number from each line.")
    p.add_argument('source_dir', type=Path, help='Directory containing input .txt files')
    p.add_argument('dest_dir', type=Path, help='Directory to write modified .txt files')
    args = p.parse_args()

    src: Path = args.source_dir
    dst: Path = args.dest_dir

    if not src.is_dir():
        raise SystemExit(f"Source directory does not exist or is not a directory: {src}")

    txt_files = sorted(src.glob('*.txt'))
    if not txt_files:
        raise SystemExit(f"No .txt files found in: {src}")

    for i, src_file in enumerate(txt_files, start=1):
        dst_file = dst / src_file.name
        process_file(src_file, dst_file)

if __name__ == "__main__":
    main()
