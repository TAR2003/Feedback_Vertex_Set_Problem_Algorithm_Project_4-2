#!/usr/bin/env bash
set -euo pipefail

# 1. Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_SLUG="tawkirazizrahman/fvs-synthetic-dataset-20k"
TARGET_DIR="./data/synthetic"

PACE_URL="https://heibox.uni-heidelberg.de/f/97634323e3cb4aab8291/?dl=1"
PACE_TAR="pace_temp.tar.gz"
PACE_TARGET_DIR="./data/pace2022"
PACE_EXTRACTED_FOLDER="./data/heuristic_track_final_instances_all"
PACE_TIMEOUT=60

echo "--- [1/5] Installing dependencies and building the C++ engine ---"
# Ensure your build_engine.py is executable or run via python
python3 build_engine.py

echo "--- [3/5] Downloading and Extracting PACE 2022 Dataset ---"
# Ensure data directory exists
mkdir -p ./data

# Download the tar.gz file
echo "Downloading PACE 2022 instances..."
wget -q --show-progress -O "$PACE_TAR" "$PACE_URL" || curl -L -o "$PACE_TAR" "$PACE_URL"

# Remove the target pace2022 directory if it already exists to avoid conflicts
if [ -d "$PACE_TARGET_DIR" ]; then
    rm -rf "$PACE_TARGET_DIR"
fi

# Extract into the data folder
echo "Extracting archive..."
tar -xzf "$PACE_TAR" -C ./data/

# Rename the extracted folder to pace2022
if [ -d "$PACE_EXTRACTED_FOLDER" ]; then
    mv "$PACE_EXTRACTED_FOLDER" "$PACE_TARGET_DIR"
    echo "✅ PACE 2022 Dataset moved to $PACE_TARGET_DIR"
else
    echo "⚠️ Warning: Expected folder $PACE_EXTRACTED_FOLDER not found after extraction. Please check the archive contents."
fi

# Clean up the downloaded .tar.gz file
rm -f "$PACE_TAR"


# You can also generate a new synthetic benchmark dataset locally with:
# python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100 --seed 7

echo "--- Remove all the previous csv files from the results folder ---"

echo "--- [4/5] Running the PACE benchmark pipeline ---"
