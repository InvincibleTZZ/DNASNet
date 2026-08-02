#!/usr/bin/env python3
import argparse
import csv
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path


def resolve_existing_path(root: Path, candidates):
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def load_val_rows(csv_path: Path):
    if csv_path.suffix == ".zip":
        with zipfile.ZipFile(csv_path) as zf:
            csv_names = [name for name in zf.namelist() if name.endswith(".csv")]
            if not csv_names:
                raise FileNotFoundError(f"No CSV found inside {csv_path}")
            with zf.open(csv_names[0], "r") as handle:
                text_stream = io.TextIOWrapper(handle, encoding="utf-8")
                yield from csv.DictReader(text_stream)
    else:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            yield from csv.DictReader(handle)


def ensure_symlink(link_path: Path, target_path: Path):
    if link_path.is_symlink():
        if link_path.resolve() == target_path.resolve():
            return
        link_path.unlink()
    elif link_path.exists():
        raise FileExistsError(f"{link_path} already exists and is not a symlink")
    link_path.symlink_to(target_path)


def build_val_tree(root: Path, val_source: Path, val_csv: Path, overwrite: bool):
    val_target = root / "val"
    if val_target.exists() and overwrite:
        if val_target.is_symlink() or val_target.is_file():
            val_target.unlink()
        else:
            shutil.rmtree(val_target)
    val_target.mkdir(parents=True, exist_ok=True)

    linked = 0
    skipped = 0
    missing = 0

    for row in load_val_rows(val_csv):
        image_id = row["ImageId"].strip()
        prediction = row["PredictionString"].strip()
        if not prediction:
            skipped += 1
            continue

        synset = prediction.split()[0]
        image_name = f"{image_id}.JPEG"
        source_image = val_source / image_name
        if not source_image.exists():
            missing += 1
            continue

        class_dir = val_target / synset
        class_dir.mkdir(parents=True, exist_ok=True)
        link_path = class_dir / image_name

        if link_path.exists() or link_path.is_symlink():
            if overwrite:
                link_path.unlink()
            else:
                skipped += 1
                continue

        link_path.symlink_to(source_image)
        linked += 1

    return linked, skipped, missing


def main():
    parser = argparse.ArgumentParser(description="Prepare Kaggle ImageNet layout for DNASNet/ImageFolder")
    parser.add_argument("--root", type=Path, required=True, help="ImageNet root, e.g. /mnt/sda/lyk/data/datasets/ILSVRC2012")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild val links if they already exist")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root not found: {root}")

    train_source = resolve_existing_path(
        root,
        [
            "train",
            "ILSVRC/Data/CLS-LOC/train",
            "ILSVRC2012/ILSVRC/Data/CLS-LOC/train",
        ],
    )
    val_source = resolve_existing_path(
        root,
        [
            "ILSVRC/Data/CLS-LOC/val",
            "ILSVRC2012/ILSVRC/Data/CLS-LOC/val",
            "val_flat",
        ],
    )
    val_csv = resolve_existing_path(
        root,
        [
            "LOC_val_solution.csv",
            "LOC_val_solution.csv.zip",
            "ILSVRC/LOC_val_solution.csv",
            "ILSVRC/LOC_val_solution.csv.zip",
        ],
    )

    if train_source is None:
        raise FileNotFoundError("Train source not found under expected Kaggle ImageNet paths")
    if val_source is None:
        raise FileNotFoundError("Validation image source not found under expected Kaggle ImageNet paths")
    if val_csv is None:
        raise FileNotFoundError("LOC_val_solution.csv(.zip) not found under expected Kaggle ImageNet paths")

    train_link = root / "train"
    if train_source != train_link:
        ensure_symlink(train_link, train_source)

    linked, skipped, missing = build_val_tree(root, val_source, val_csv, overwrite=args.overwrite)

    print(f"Prepared train link: {train_link}")
    print(f"Validation symlinks created: {linked}")
    print(f"Validation entries skipped: {skipped}")
    print(f"Validation images missing: {missing}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
