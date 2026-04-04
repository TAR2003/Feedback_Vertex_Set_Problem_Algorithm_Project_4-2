#!/usr/bin/env bash
set -euo pipefail

# 1. Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_SLUG="tawkirazizrahman/fvs-synthetic-dataset-20k"
TARGET_DIR="./data/synthetic"

PACE_URL="https://heibox.uni-heidelberg.de/seafhttp/files/abe81dc1-652d-44eb-bcd2-375f71061e55/heuristic_track_final_instances_all.tar.gz"
PACE_TAR="pace_temp.tar.gz"
PACE_TARGET_DIR="./data/pace2022"
PACE_EXTRACTED_FOLDER="./data/heuristic_track_final_instances_all"

echo "--- [1/5] Installing dependencies and building the C++ engine ---"
# Ensure your build_engine.py is executable or run via python
python3 build_engine.py

echo "--- [2/5] Downloading Kaggle Dataset (Public) ---"
# Install kagglehub if missing, then download and move data
python3 -m pip install -q kagglehub
python3 -c "
import kagglehub
import shutil
import os

print(f'Downloading {os.environ.get(\"DATASET_SLUG\", \"$DATASET_SLUG\")}...')
path = kagglehub.dataset_download(\"$DATASET_SLUG\")

target = \"$TARGET_DIR\"
if os.path.exists(target):
    shutil.rmtree(target)
os.makedirs(os.path.dirname(target), exist_ok=True)

shutil.copytree(path, target)
print(f'✅ Kaggle Dataset moved to {target}')
"

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

echo "--- [4/5] Running the default pipeline ---"
python3 experiments/run_pipeline.py --mode all --algo ALL --timeout 30

echo "--- [5/5] Pipeline finished ---"

cat <<'EOF'

To customize the run, edit run.sh or invoke the pipeline directly:
  python3 experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50
  python3 experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100
EOF