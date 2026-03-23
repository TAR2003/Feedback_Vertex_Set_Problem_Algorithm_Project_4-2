"""
analysis/report_writer.py
-------------------------
Handles all writes to results/report.csv.

Usage:
    writer = ReportWriter(results_dir)
    writer.write_row(experiment_id="EXP1", ...)
"""

import csv
import logging
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CSV schema — column order for report.csv
# ---------------------------------------------------------------------------
REPORT_COLUMNS = [
    "experiment_id",
    "instance_id",
    "graph_type",
    "n_vertices",
    "n_edges",
    "graph_density",
    "algorithm",
    "run_number",
    "fvs_size",
    "optimal_fvs_size",
    "approximation_ratio",
    "optimality_gap_pct",
    "wall_time_sec",
    "cpu_time_sec",
    "peak_memory_mb",
    "is_valid_solution",
    "notes",
    "timestamp",
]


class ReportWriter:
    """
    Thread-safe writer for results/report.csv.

    Writes are immediately flushed to disk so partial results are preserved
    even if the process is interrupted.
    """

    def __init__(self, results_dir: Path):
        self.path = results_dir / "report.csv"
        self._lock = threading.Lock()
        self._ensure_header()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def write_row(
        self,
        experiment_id: str,
        instance_id: str,
        graph_type: str,
        n_vertices: int,
        n_edges: int,
        graph_density: float,
        algorithm: str,
        run_number: int,
        fvs_size: int,
        wall_time_sec: float,
        cpu_time_sec: float,
        peak_memory_mb: float,
        is_valid_solution: bool,
        optimal_fvs_size: Any = "",
        approximation_ratio: Any = "",
        optimality_gap_pct: Any = "",
        notes: str = "",
    ) -> None:
        """
        Append one row to report.csv.

        Args match report.csv column names exactly.
        Optional fields default to empty string (per spec).
        """
        row = {
            "experiment_id":      experiment_id,
            "instance_id":        instance_id,
            "graph_type":         graph_type,
            "n_vertices":         n_vertices,
            "n_edges":            n_edges,
            "graph_density":      f"{graph_density:.6f}",
            "algorithm":          algorithm,
            "run_number":         run_number,
            "fvs_size":           fvs_size,
            "optimal_fvs_size":   optimal_fvs_size,
            "approximation_ratio": approximation_ratio,
            "optimality_gap_pct": optimality_gap_pct,
            "wall_time_sec":      f"{wall_time_sec:.6f}",
            "cpu_time_sec":       f"{cpu_time_sec:.6f}",
            "peak_memory_mb":     f"{peak_memory_mb:.3f}",
            "is_valid_solution":  is_valid_solution,
            "notes":              notes,
            "timestamp":          datetime.utcnow().isoformat(),
        }
        self._append(row)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_header(self) -> None:
        """Create report.csv with header row if it does not exist yet."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            try:
                with open(self.path, "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS)
                    writer.writeheader()
                logger.info("Created report.csv at %s", self.path)
            except Exception as exc:
                logger.error("Failed to create report.csv: %s", exc)

    def _append(self, row: dict) -> None:
        """Append a single row to the CSV file (thread-safe)."""
        with self._lock:
            try:
                with open(self.path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=REPORT_COLUMNS,
                                            extrasaction="ignore")
                    writer.writerow(row)
                    f.flush()
            except Exception as exc:
                logger.error("Failed to write to report.csv: %s", exc)
