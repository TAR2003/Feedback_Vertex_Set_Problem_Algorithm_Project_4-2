#!/usr/bin/env python3
"""
Build script for the C++ engine using cmake
"""
import subprocess
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
CPP_ENGINE = PROJECT_ROOT / "cpp_engine"
BUILD_DIR = CPP_ENGINE / "build"

# Create build directory
BUILD_DIR.mkdir(exist_ok=True)

# Change to build directory and run cmake
os.chdir(BUILD_DIR)

print("Running cmake...")
result = subprocess.run([sys.executable, "-m", "cmake", ".."], check=False)
if result.returncode != 0:
    print("CMake configuration failed!")
    sys.exit(1)

print("\nBuilding with cmake...")
result = subprocess.run([sys.executable, "-m", "cmake", "--build", ".", "--config", "Release"], check=False)
if result.returncode != 0:
    print("Build failed!")
    sys.exit(1)

print("\n✓ Build complete!")
