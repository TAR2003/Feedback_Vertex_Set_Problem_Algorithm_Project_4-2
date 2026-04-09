import argparse
import csv
import os
import re
from pathlib import Path

DEFAULT_HEADER = [
    "file",
    "n",
    "m",
    "FVS_size",
    "runtime",
    "validity",
    "kernelization_time",
    "gnn_candidate_time",
    "initial_kernel_size",
    "final_kernel_size",
    "n_dynamic_reductions",
    "solution_size",
    "time_seconds",
]

FILE_LINE_RE = re.compile(r"^.*\bFile\s*:\s*(?P<file>\S+)", re.IGNORECASE)
GRAPH_LINE_RE = re.compile(
    r"^.*\bGraph\s*:\s*(?P<n>\d+)\s+vertices\s*,\s*(?P<m>\d+)\s+directed\s+edges",
    re.IGNORECASE,
)
RESULT_LINE_RE = re.compile(
    r"^.*\bDFVS\s+size\s*=\s*(?P<fvs>\d+)\s*\|\s*Time\s*=\s*(?P<ms>[0-9.]+)\s*ms\s*\|\s*(?P<validity>.*)$",
    re.IGNORECASE,
)
VALID_LINE_RE = re.compile(r"✓\s*VALID", re.IGNORECASE)
INVALID_LINE_RE = re.compile(r"✗\s*INVALID|INVALID", re.IGNORECASE)


def parse_score_file(path: Path):
    rows = []
    current = {}

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            file_match = FILE_LINE_RE.match(line)
            if file_match:
                if current.get("file") and "FVS_size" in current:
                    rows.append(current)
                current = {
                    "file": file_match.group("file"),
                    "n": None,
                    "m": None,
                    "FVS_size": None,
                    "runtime": None,
                    "validity": None,
                    "kernelization_time": 0.0,
                    "gnn_candidate_time": 0.0,
                    "initial_kernel_size": None,
                    "final_kernel_size": None,
                    "n_dynamic_reductions": 0,
                    "solution_size": None,
                    "time_seconds": None,
                }
                continue

            graph_match = GRAPH_LINE_RE.match(line)
            if graph_match and current:
                current["n"] = int(graph_match.group("n"))
                current["m"] = int(graph_match.group("m"))
                current["initial_kernel_size"] = current["n"]
                current["final_kernel_size"] = current["n"]
                continue

            result_match = RESULT_LINE_RE.match(line)
            if result_match and current:
                fvs_size = int(result_match.group("fvs"))
                ms = float(result_match.group("ms"))
                validity_text = result_match.group("validity")
                current["FVS_size"] = fvs_size
                current["solution_size"] = fvs_size
                current["runtime"] = round(ms / 1000.0, 6)
                current["time_seconds"] = round(ms / 1000.0, 6)
                if VALID_LINE_RE.search(validity_text):
                    current["validity"] = True
                elif INVALID_LINE_RE.search(validity_text):
                    current["validity"] = False
                else:
                    current["validity"] = None
                continue

    if current.get("file") and "FVS_size" in current and current["FVS_size"] is not None:
        rows.append(current)

    return rows


def detect_output_filename(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.lower().endswith(".csv"):
                return line
            break
    return None


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=DEFAULT_HEADER)
        writer.writeheader()
        for row in rows:
            normalized = {key: row.get(key) for key in DEFAULT_HEADER}
            writer.writerow(normalized)


def main():
    parser = argparse.ArgumentParser(
        description="Parse paceresults/score.txt and write a CSV file with standard result columns."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="score.txt",
        help="Input score text file to parse (default: score.txt).",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output CSV filename. If omitted, the first CSV name found in the score file is used.",
    )

    args = parser.parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_filename = args.output or detect_output_filename(input_path)
    if output_filename is None:
        raise ValueError("Could not detect the output CSV filename from the score file. Please provide --output.")

    output_path = input_path.parent / output_filename
    rows = parse_score_file(input_path)
    if not rows:
        raise ValueError("No valid entries were parsed from the score file.")

    write_csv(output_path, rows)
    print(f"Parsed {len(rows)} entries and wrote: {output_path}")


if __name__ == "__main__":
    main()
