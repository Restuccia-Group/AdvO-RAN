#!/usr/bin/env python3
"""Filter a CSV to rows allowed by action combos in config_em_filtered.py."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, Set, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config_em_filtered as cfg

ALLOWED_BY_SLICE: Dict[int, Set[Tuple[int, int]]] = {}
for prb_alloc in cfg.feasible_prb_allocation_all:
    for sched_combo in cfg.scheduling_combos:
        for slice_id in (0, 1, 2):
            ALLOWED_BY_SLICE.setdefault(slice_id, set()).add(
                (int(prb_alloc[slice_id]), int(sched_combo[slice_id]))
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter dataset using config_em_filtered.py"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV path (must contain slice_id, slice_prb, scheduling_policy)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output filtered CSV path",
    )
    return parser.parse_args()


def to_int(cell: str) -> int:
    return int(float(cell))


def required_columns(fieldnames: Iterable[str] | None) -> None:
    header = set(fieldnames or [])
    needed = {"slice_id", "slice_prb", "scheduling_policy"}
    missing = sorted(needed - header)
    if missing:
        raise ValueError(
            "Input CSV missing required columns: " + ", ".join(missing)
        )


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    kept_rows = 0
    skipped_invalid = 0

    with args.input.open("r", newline="") as fin, args.output.open(
        "w", newline=""
    ) as fout:
        reader = csv.DictReader(fin)
        required_columns(reader.fieldnames)

        writer = csv.DictWriter(fout, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            total_rows += 1
            try:
                slice_id = to_int(row["slice_id"])
                prb = to_int(row["slice_prb"])
                sched = to_int(row["scheduling_policy"])
            except (ValueError, TypeError, KeyError):
                skipped_invalid += 1
                continue

            if (prb, sched) in ALLOWED_BY_SLICE.get(slice_id, set()):
                writer.writerow(row)
                kept_rows += 1

    print(f"Input rows: {total_rows}")
    print(f"Kept rows: {kept_rows}")
    print(f"Dropped rows: {total_rows - kept_rows}")
    if skipped_invalid:
        print(f"Skipped invalid rows: {skipped_invalid}")
    print(f"Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
