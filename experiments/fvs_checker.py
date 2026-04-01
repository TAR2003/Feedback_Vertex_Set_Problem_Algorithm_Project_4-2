#!/usr/bin/env python3
"""
Compare brute-force FVS results against IC and BST results.

Inputs per family:
- results/<family>_brute_force.csv
- results/<family>_IC.csv
- results/<family>_BST.csv

Outputs per family:
- results/<family>_fvs_check.csv

The checker does not run algorithms; it only compares existing CSV results.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results"


def read_csv_map(csv_path: Path) -> Dict[str, dict]:
    if not csv_path.exists():
        return {}

    rows: Dict[str, dict] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = str(row.get("file", "")).strip()
            if key:
                rows[key] = row
    return rows


def int_or_none(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    s = str(value).strip()
    if s.lstrip("-").isdigit():
        return int(s)
    return None


def compare_to_bruteforce(brute_val: Optional[int], ref_val: Optional[int]) -> str:
    if brute_val is None or ref_val is None:
        return "N/A"
    return "True" if brute_val == ref_val else "False"


def build_rows_for_family(results_dir: Path, family: str) -> List[dict]:
    brute_map = read_csv_map(results_dir / f"{family}_brute_force.csv")
    ic_map = read_csv_map(results_dir / f"{family}_IC.csv")
    bst_map = read_csv_map(results_dir / f"{family}_BST.csv")

    rows: List[dict] = []
    if not brute_map:
        return rows

    for file_name in sorted(brute_map.keys()):
        brute_row = brute_map[file_name]
        ic_row = ic_map.get(file_name)
        bst_row = bst_map.get(file_name)

        brute_size_raw = brute_row.get("FVS_size", "N/A")
        ic_size_raw = ic_row.get("FVS_size", "N/A") if ic_row else "N/A"
        bst_size_raw = bst_row.get("FVS_size", "N/A") if bst_row else "N/A"

        brute_size = int_or_none(brute_size_raw)
        ic_size = int_or_none(ic_size_raw)
        bst_size = int_or_none(bst_size_raw)

        rows.append(
            {
                "file": file_name,
                "n": brute_row.get("n", ""),
                "m": brute_row.get("m", ""),
                "brute_force_result": brute_size_raw,
                "IC_result": ic_size_raw,
                "BST_result": bst_size_raw,
                "IC_match?": compare_to_bruteforce(brute_size, ic_size),
                "BST_match?": compare_to_bruteforce(brute_size, bst_size),
                "brute_validity": brute_row.get("validity", ""),
                "brute_runtime": brute_row.get("runtime", ""),
            }
        )

    return rows


def write_check_csv(output_path: Path, rows: List[dict]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "file",
                "n",
                "m",
                "brute_force_result",
                "IC_result",
                "BST_result",
                "IC_match?",
                "BST_match?",
                "brute_validity",
                "brute_runtime",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def run_family(results_dir: Path, family: str) -> Path:
    rows = build_rows_for_family(results_dir, family)
    out_csv = results_dir / f"{family}_fvs_check.csv"
    write_check_csv(out_csv, rows)
    print(f"[OK] {family}: wrote {len(rows)} row(s) to {out_csv}")
    return out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare brute-force results with IC and BST result CSVs")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    args = parser.parse_args()

    out_und = run_family(args.results_dir, "undirected")
    out_dir = run_family(args.results_dir, "directed")

    print("\n[DONE] FVS comparison complete")
    print(f"  - {out_und}")
    print(f"  - {out_dir}")


if __name__ == "__main__":
    main()
