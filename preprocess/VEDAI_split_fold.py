#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
import re
import math
from typing import Dict, List, Set, Tuple


ID_RE = re.compile(r"^\d{8}$")
IMG_W = 1024.0
IMG_H = 1024.0

# Optional mapping from VEDAI original ids to contiguous 0..8.
# Disable with --no_class_map if you don't want it.
VEDAI_CLASS_MAP: Dict[int, int] = {
    1: 0,   # car
    2: 1,   # truck
    23: 2,  # boat
    4: 3,   # tractor
    5: 4,   # camping car
    11: 5,  # pick-up
    31: 6,  # plane
    7: 7,   # motorbike -> other
    8: 7,   # bus -> other
    10: 7,  # other -> other
    9: 8,   # van
}


def parse_test_ids_from_car01(split_file: Path) -> List[str]:
    ids: Set[str] = set()
    with split_file.open("r", encoding="utf-8", errors="ignore") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            first = line.split()[0]
            if not first.isdigit():
                raise ValueError(f"{split_file}:{ln}: first token {first!r} is not numeric")
            img_id = first.zfill(8)
            if not ID_RE.match(img_id):
                raise ValueError(f"{split_file}:{ln}: parsed id {img_id!r} is not 8 digits")
            ids.add(img_id)
    return sorted(ids)


def build_eligible_ids(img_dir: Path, ann_dir: Path) -> Tuple[Set[str], Dict[str, int]]:
    eligible: Set[str] = set()
    stats = {"total_co_seen": 0, "missing_ir": 0, "missing_ann": 0, "bad_id": 0}

    for co_path in img_dir.glob("*_co.png"):
        stats["total_co_seen"] += 1
        stem = co_path.stem
        if not stem.endswith("_co"):
            continue
        img_id = stem[:-3]
        if not ID_RE.match(img_id):
            stats["bad_id"] += 1
            continue

        ir_path = img_dir / f"{img_id}_ir.png"
        ann_path = ann_dir / f"{img_id}.txt"

        if not ir_path.exists():
            stats["missing_ir"] += 1
            continue
        if not ann_path.exists():
            stats["missing_ann"] += 1
            continue

        eligible.add(img_id)

    return eligible, stats


def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def remove_if_exists(p: Path) -> None:
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def copy_file(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _all_finite(nums: List[float]) -> bool:
    return all(math.isfinite(x) for x in nums)


def convert_annotation_file_to_yolo_strict(
    src_ann: Path,
    *,
    filter_not_contained: bool,
    apply_class_map: bool,
) -> Tuple[bool, List[str], Dict[str, int]]:
    """
    STRICT conversion:
      - If ANY "bad line" (non-numeric / unexpected column count / non-finite) => skip sample
      - If ANY "invalid box after clipping" (w<=0 or h<=0) => skip sample (your new rule)

    Supported per-line format:
      (A) 14 values:
        x_c y_c orient class is_contained is_occluded x1 x2 x3 x4 y1 y2 y3 y4
    """
    stats = {
        "kept": 0,
        "skipped_not_contained": 0,
        "bad_lines": 0,
        "invalid_box_lines": 0,  # <- now fatal
    }

    out_lines: List[str] = []

    with src_ann.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            try:
                vals = [float(x) for x in parts]
            except ValueError:
                stats["bad_lines"] += 1
                return False, [], stats

            if not _all_finite(vals):
                stats["bad_lines"] += 1
                return False, [], stats

            if len(vals) == 14:
                is_contained = int(round(vals[4]))
                cls = int(round(vals[3]))
                xs = vals[6:10]
                ys = vals[10:14]
            else:
                stats["bad_lines"] += 1
                return False, [], stats

            if filter_not_contained and is_contained != 1:
                stats["skipped_not_contained"] += 1
                continue

            # bbox from corners
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)

            # clip to image bounds (keeps cropped objects by clipping)
            min_x = clamp(min_x, 0.0, IMG_W)
            max_x = clamp(max_x, 0.0, IMG_W)
            min_y = clamp(min_y, 0.0, IMG_H)
            max_y = clamp(max_y, 0.0, IMG_H)

            w = max_x - min_x
            h = max_y - min_y

            # NEW RULE: if ANY invalid box after clipping => skip entire sample
            if w <= 1e-6 or h <= 1e-6:
                stats["invalid_box_lines"] += 1
                return False, [], stats

            xc = (min_x + max_x) / 2.0
            yc = (min_y + max_y) / 2.0

            # normalize to [0,1]
            xc_n = clamp(xc / IMG_W, 0.0, 1.0)
            yc_n = clamp(yc / IMG_H, 0.0, 1.0)
            w_n = clamp(w / IMG_W, 0.0, 1.0)
            h_n = clamp(h / IMG_H, 0.0, 1.0)

            if apply_class_map and cls in VEDAI_CLASS_MAP:
                cls = VEDAI_CLASS_MAP[cls]

            out_lines.append(f"{cls} {xc_n:.6f} {yc_n:.6f} {w_n:.6f} {h_n:.6f}")
            stats["kept"] += 1

    return True, out_lines, stats


def process_split(
    ids: List[str],
    img_dir: Path,
    ann_dir: Path,
    out_rgb_dir: Path,
    out_ir_dir: Path,
    out_lbl_dir: Path,
    overwrite: bool,
    *,
    filter_not_contained: bool,
    apply_class_map: bool,
) -> Dict[str, int]:
    stats = {
        "samples_copied": 0,
        "samples_skipped_bad_or_invalid_ann": 0,
        "bad_lines_total": 0,
        "invalid_box_lines_total": 0,
        "labels_kept_total": 0,
        "labels_skipped_not_contained_total": 0,
    }

    for img_id in ids:
        src_rgb = img_dir / f"{img_id}_co.png"
        src_ir = img_dir / f"{img_id}_ir.png"
        src_ann = ann_dir / f"{img_id}.txt"

        # If any missing (ann or either modality) => skip
        if not (src_rgb.exists() and src_ir.exists() and src_ann.exists()):
            continue

        dst_rgb = out_rgb_dir / f"{img_id}.png"   # remove _co
        dst_ir  = out_ir_dir / f"{img_id}.png"    # remove _ir
        dst_ann = out_lbl_dir / f"{img_id}.txt"

        ok, yolo_lines, conv = convert_annotation_file_to_yolo_strict(
            src_ann,
            filter_not_contained=filter_not_contained,
            apply_class_map=apply_class_map,
        )

        if not ok:
            stats["samples_skipped_bad_or_invalid_ann"] += 1
            stats["bad_lines_total"] += conv["bad_lines"]
            stats["invalid_box_lines_total"] += conv["invalid_box_lines"]

            # Ensure nothing for this id exists in output
            remove_if_exists(dst_rgb)
            remove_if_exists(dst_ir)
            remove_if_exists(dst_ann)
            continue

        # Write labels (may be empty => valid negative sample)
        dst_ann.parent.mkdir(parents=True, exist_ok=True)
        if overwrite or not dst_ann.exists():
            dst_ann.write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")

        # Copy images only after labels succeed
        copy_file(src_rgb, dst_rgb, overwrite)
        copy_file(src_ir, dst_ir, overwrite)

        stats["samples_copied"] += 1
        stats["labels_kept_total"] += conv["kept"]
        stats["labels_skipped_not_contained_total"] += conv["skipped_not_contained"]

    return stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vedai_root", type=Path, required=True)
    ap.add_argument("--split_file", type=Path, required=True)
    ap.add_argument("--out_root", type=Path, required=True)
    ap.add_argument("--overwrite", action="store_true")

    ap.add_argument("--keep_not_contained", action="store_true",
                    help="Keep cropped/not-fully-contained objects by clipping boxes to bounds.")
    ap.add_argument("--no_class_map", action="store_true",
                    help="Disable VEDAI class-id remapping (keep raw class IDs).")

    args = ap.parse_args()

    img_dir = args.vedai_root / "Vehicules1024"
    ann_dir = args.vedai_root / "Annotations1024"

    if not img_dir.is_dir():
        raise SystemExit(f"ERROR: images dir not found: {img_dir}")
    if not ann_dir.is_dir():
        raise SystemExit(f"ERROR: annotations dir not found: {ann_dir}")
    if not args.split_file.is_file():
        raise SystemExit(f"ERROR: split file not found: {args.split_file}")

    raw_test_ids = parse_test_ids_from_car01(args.split_file)
    eligible_ids, elig_stats = build_eligible_ids(img_dir, ann_dir)

    test_ids = sorted([i for i in raw_test_ids if i in eligible_ids])
    train_ids = sorted(list(eligible_ids.difference(test_ids)))

    out_train_rgb = args.out_root / "train" / "img"
    out_train_ir  = args.out_root / "train" / "imgr"
    out_train_lbl = args.out_root / "train" / "label"

    out_test_rgb  = args.out_root / "test" / "img"
    out_test_ir   = args.out_root / "test" / "imgr"
    out_test_lbl  = args.out_root / "test" / "label"

    filter_not_contained = not args.keep_not_contained
    apply_class_map = not args.no_class_map

    print("=== VEDAI split + YOLOv5 conversion (STRICT bad/invalid annotation policy) ===")
    print(f"CO images found: {elig_stats['total_co_seen']}")
    print(f"Eligible (co+ir+ann): {len(eligible_ids)}")
    print(f"Test IDs in car_01.txt: {len(raw_test_ids)}")
    print(f"Test IDs eligible: {len(test_ids)} (skipped from test: {len(raw_test_ids)-len(test_ids)})")
    print(f"Train IDs: {len(train_ids)}")
    print("Eligibility scan skips:")
    print(f"  missing IR pair: {elig_stats['missing_ir']}")
    print(f"  missing annotation: {elig_stats['missing_ann']}")
    print(f"  bad id format: {elig_stats['bad_id']}")
    print(f"Conversion options: filter_not_contained={filter_not_contained}, apply_class_map={apply_class_map}")
    print(f"Output root: {args.out_root.resolve()}")

    train_stats = process_split(
        train_ids, img_dir, ann_dir,
        out_train_rgb, out_train_ir, out_train_lbl,
        overwrite=args.overwrite,
        filter_not_contained=filter_not_contained,
        apply_class_map=apply_class_map,
    )
    test_stats = process_split(
        test_ids, img_dir, ann_dir,
        out_test_rgb, out_test_ir, out_test_lbl,
        overwrite=args.overwrite,
        filter_not_contained=filter_not_contained,
        apply_class_map=apply_class_map,
    )

    def show_stats(name: str, st: Dict[str, int]) -> None:
        print(f"\n[{name}]")
        print(f"  samples copied: {st['samples_copied']}")
        print(f"  samples skipped (bad line OR invalid box): {st['samples_skipped_bad_or_invalid_ann']}")
        print(f"  bad lines total (across skipped samples): {st['bad_lines_total']}")
        print(f"  invalid-box lines total (across skipped samples): {st['invalid_box_lines_total']}")
        print(f"  yolo labels kept (objects): {st['labels_kept_total']}")
        print(f"  objects skipped not-contained: {st['labels_skipped_not_contained_total']}")

    show_stats("train", train_stats)
    show_stats("test", test_stats)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# usage: python split_vedai_fold.py --vedai_root "/media/nima/7CEEFEC0EEFE71AE/space/VEDAI/" --split_file "/media/nima/7CEEFEC0EEFE71AE/space/VEDAI/DevKit/Resultats/car_01.txt" --out_root "/media/nima/7CEEFEC0EEFE71AE/space/VEDAI_1/" --overwrite --keep_not_contained 
