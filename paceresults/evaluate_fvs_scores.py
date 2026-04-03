#!/usr/bin/env python3
"""Evaluate FVS solver outputs across instances and normalize scores.

Usage:
  python evaluate_fvs_scores.py

Generates:
  - detailed_scores.csv : per-instance fvs/score for each solver + optimal
  - summary_scores.csv  : per-solver average score across instances

Rules:
  - considers h_001..h_200 only (if present in data)
  - best FVS is minimum FVS_size across solvers per instance
  - score = 100 * optimal / solver_fvs (for valid entries)
"""

from __future__ import annotations

import csv
import glob
import os
import statistics
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = SCRIPT_DIR

CSV_GLOB = os.path.join(DATA_DIR, "*.csv")

# Instance filter letters: h_001 .. h_200
def is_instance_of_interest(instance: str) -> bool:
    if not instance.startswith("h_"):
        return False
    try:
        n = int(instance[2:])
    except ValueError:
        return False
    return 1 <= n <= 200


def parse_fvs_size(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() in {"N/A", "TIMEOUT", "ERROR", "NONE"}:
        return None
    try:
        if "." in s:
            v = float(s)
            if v.is_integer():
                return int(v)
            return v
        return int(s)
    except ValueError:
        return None


def main() -> int:
    csv_files = sorted(glob.glob(CSV_GLOB))
    if not csv_files:
        print(f"No CSV files found under {DATA_DIR}")
        return 1

    instances: dict[str, dict[str, float]] = {}
    solvers: list[str] = []

    for csv_file in csv_files:
        solver_name = os.path.splitext(os.path.basename(csv_file))[0]
        solvers.append(solver_name)

        with open(csv_file, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            # look for common fvs columns
            instance_key = None
            for candidate in ["instance", "file", "Instance", "FILE"]:
                if candidate in reader.fieldnames:
                    instance_key = candidate
                    break
            if instance_key is None:
                raise ValueError(f"No instance/file column in {csv_file}")

            size_key = None
            for candidate in ["fvs_size", "FVS_size", "fvs-size", "FVS_size"]:
                if candidate in reader.fieldnames:
                    size_key = candidate
                    break
            if size_key is None:
                raise ValueError(f"No fvs_size/FVS_size column in {csv_file}")

            for row in reader:
                instance = str(row.get(instance_key, "")).strip()
                if not instance or not is_instance_of_interest(instance):
                    continue
                fvs_raw = row.get(size_key, "")
                fvs_size = parse_fvs_size(fvs_raw)
                if fvs_size is None:
                    continue
                instances.setdefault(instance, {})[solver_name] = float(fvs_size)

    if not instances:
        print("No valid instance entries found in input CSVs.")
        return 1

    all_instances = sorted(instances.keys(), key=lambda x: int(x[2:]))

    # compute per-instance optimal & per-solver scores
    per_instance_scores: dict[str, dict[str, float]] = {}
    solver_scores: dict[str, list[float]] = {solver: [] for solver in solvers}

    for instance in all_instances:
        solver_map = instances[instance]
        if not solver_map:
            continue
        optimal = min(solver_map.values())
        scores_for_instance = {}
        for solver in solvers:
            fvs = solver_map.get(solver)
            if fvs is None or fvs <= 0:
                scores_for_instance[f"{solver}_fvs"] = "N/A"
                scores_for_instance[f"{solver}_score"] = "N/A"
                continue
            score = 100.0 * optimal / fvs
            scores_for_instance[f"{solver}_fvs"] = fvs
            scores_for_instance[f"{solver}_score"] = round(score, 4)
            solver_scores[solver].append(score)

        per_instance_scores[instance] = {
            "optimal_fvs": optimal,
            **scores_for_instance,
        }

    # write detailed output
    detailed_csv = os.path.join(DATA_DIR, "detailed_scores.csv")
    with open(detailed_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["instance", "optimal_fvs"]
        for solver in solvers:
            fieldnames.extend([f"{solver}_fvs", f"{solver}_score"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for instance in all_instances:
            row = {"instance": instance}
            row.update(per_instance_scores.get(instance, {}))
            writer.writerow(row)

    summary_csv = os.path.join(DATA_DIR, "summary_scores.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["solver", "mean_score", "instance_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for solver in solvers:
            values = solver_scores.get(solver, [])
            if values:
                mean_val = statistics.mean(values)
                writer.writerow({"solver": solver, "mean_score": round(mean_val, 4), "instance_count": len(values)})
            else:
                writer.writerow({"solver": solver, "mean_score": "N/A", "instance_count": 0})

    # print the final answer (mean of each solver)
    print("Final solver mean normalized score (100*optimal/fvs):")
    for solver in solvers:
        values = solver_scores.get(solver, [])
        if values:
            print(f"  {solver}: {statistics.mean(values):.4f} (based on {len(values)} instances)")
        else:
            print(f"  {solver}: N/A")

    print(f"\nWrote detailed per-instance scores to: {detailed_csv}")
    print(f"Wrote summary per-solver mean scores to: {summary_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
