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

# Instance filter supports PACE benchmark names and generic graph instance filenames.
def is_instance_of_interest(instance: str) -> bool:
    if not instance:
        return False

    # Keep the original PACE `h_001`..`h_200` behavior when present.
    if instance.startswith("h_"):
        try:
            n = int(instance[2:])
        except ValueError:
            return False
        return 1 <= n <= 200

    # Accept common graph file naming patterns such as dag_000000.txt and grid_000000.txt.
    base = os.path.basename(instance)
    if "_" in base:
        _, suffix = base.split("_", 1)
        suffix = suffix.split(".", 1)[0]
        if suffix.isdigit():
            return True

    # Fallback: accept any non-empty instance name.
    return True


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


def parse_node_count(raw: str):
    if raw is None:
        return None
    s = str(raw).strip()
    if s == "" or s.upper() in {"N/A", "UNKNOWN", "NONE"}:
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def parse_validity(raw: str):
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s == "":
        return None
    if s in {"true", "yes", "ok", "1", "success", "succ"}:
        return True
    if s in {"false", "no", "fail", "failed", "0", "error", "invalid"}:
        return False
    return None


def main() -> int:
    detailed_csv = os.path.join(DATA_DIR, "detailed_scores.csv")
    summary_csv = os.path.join(DATA_DIR, "summary_scores.csv")
    for old_csv in (detailed_csv, summary_csv):
        if os.path.exists(old_csv):
            os.remove(old_csv)

    csv_files = sorted(glob.glob(CSV_GLOB))
    csv_files = [f for f in csv_files if os.path.basename(f) not in {"detailed_scores.csv", "summary_scores.csv"}]
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
                print(f"Skipping {csv_file}: no fvs_size/FVS_size column found.")
                continue

            node_key = None
            for candidate in ["nodes_n", "n", "nodes", "vertices", "num_vertices", "node_count"]:
                if candidate in reader.fieldnames:
                    node_key = candidate
                    break

            status_key = None
            exit_code_key = None
            validity_key = None
            for candidate in ["status", "validity", "valid", "is_valid"]:
                if candidate in reader.fieldnames:
                    status_key = candidate
                    break
            for candidate in ["exit_code", "exitcode", "code"]:
                if candidate in reader.fieldnames:
                    exit_code_key = candidate
                    break

            for row in reader:
                instance = str(row.get(instance_key, "")).strip()
                if not instance or not is_instance_of_interest(instance):
                    continue

                fvs_raw = row.get(size_key, "")
                fvs_size = parse_fvs_size(fvs_raw)
                node_count = parse_node_count(row.get(node_key)) if node_key else None

                validity = None
                if status_key:
                    validity = parse_validity(row.get(status_key))
                if validity is None and exit_code_key:
                    exit_code_raw = row.get(exit_code_key)
                    if exit_code_raw is not None:
                        try:
                            validity = int(str(exit_code_raw).strip()) == 0
                        except ValueError:
                            validity = None

                if validity is False:
                    if node_count is None:
                        continue
                    fvs_size = float(node_count)

                if fvs_size is None:
                    continue

                instances.setdefault(instance, {})[solver_name] = float(fvs_size)

    if not instances:
        print("No valid instance entries found in input CSVs.")
        return 1

    def instance_sort_key(instance: str):
        base = os.path.basename(instance)
        if "_" in base:
            prefix, suffix = base.split("_", 1)
            digits = "".join(ch for ch in suffix if ch.isdigit())
            if digits:
                try:
                    return (0, int(digits), prefix, suffix)
                except ValueError:
                    pass
        return (1, base)

    all_instances = sorted(instances.keys(), key=instance_sort_key)

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
            if fvs is None:
                scores_for_instance[f"{solver}_fvs"] = "N/A"
                scores_for_instance[f"{solver}_score"] = "N/A"
                continue
            if optimal == 0:
                if fvs == 0:
                    score = 100.0
                else:
                    score = 0.0
            elif fvs <= 0:
                scores_for_instance[f"{solver}_fvs"] = fvs
                scores_for_instance[f"{solver}_score"] = "N/A"
                continue
            else:
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
