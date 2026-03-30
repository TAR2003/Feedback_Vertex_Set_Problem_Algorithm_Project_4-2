#!/usr/bin/env python3
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR / "experiments"))
sys.path.insert(0, str(SCRIPT_DIR / "cpp_engine" / "build"))

try:
    import cpp_engine
    print(f"✓ cpp_engine imported successfully")
    print(f"  Version: {cpp_engine.__version__}")
except ImportError as e:
    print(f"✗ Failed to import cpp_engine: {e}")
    print(f"  sys.path: {sys.path[:3]}")
    exit(1)
