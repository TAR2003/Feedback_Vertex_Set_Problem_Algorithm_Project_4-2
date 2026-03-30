#!/usr/bin/env python3
"""
Quick test to verify HYBRID import works (from WSL where cpp_engine.so is available)
"""

import sys
from pathlib import Path

SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
for candidate in ("build-linux", "build-macos", "build-win", "build"):
    sys.path.insert(0, str(PROJECT_ROOT / "cpp_engine" / candidate))

try:
    import cpp_engine
    print("✓ cpp_engine imported successfully")
except ImportError as e:
    print(f"✗ cpp_engine import failed: {e}")
    sys.exit(1)

try:
    from experiments.run_hybrid import hybrid_solve_directed, hybrid_solve_undirected
    print("✓ hybrid_solve_directed imported successfully")
    print("✓ hybrid_solve_undirected imported successfully")
except ImportError as e:
    print(f"✗ Failed to import hybrid solvers: {e}")
    sys.exit(1)

print("\n✓ All imports successful! HYBRID integration is working.")
