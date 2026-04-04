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

echo "--- [1/5] Installing dependencies and building the C++ engine ---"
python3 build_engine.py

echo "--- [2/5] Downloading Kaggle Dataset (Public) ---"
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
mkdir -p ./data

echo "Downloading PACE 2022 instances..."
wget -q --show-progress -O "$PACE_TAR" "$PACE_URL" || curl -L -o "$PACE_TAR" "$PACE_URL"

if [ -d "$PACE_TARGET_DIR" ]; then
    rm -rf "$PACE_TARGET_DIR"
fi

echo "Extracting archive..."
tar -xzf "$PACE_TAR" -C ./data/

if [ -d "$PACE_EXTRACTED_FOLDER" ]; then
    mv "$PACE_EXTRACTED_FOLDER" "$PACE_TARGET_DIR"
    echo "✅ PACE 2022 Dataset moved to $PACE_TARGET_DIR"
else
    echo "⚠️ Warning: Expected folder $PACE_EXTRACTED_FOLDER not found after extraction. Please check the archive contents."
fi

rm -f "$PACE_TAR"

echo "--- [4/5] Generating GNN dataset and training models ---"

echo "--- Version 1 ---"
python3 gnn_model/dataset_gen.py --family all
python3 gnn_model/train.py --type both --epochs 100

echo "--- Version 2 ---"
python3 gnn_model/dataset_gen.py --family all --variant v2
python3 gnn_model/train.py --type both --variant v2 --epochs 100

echo "--- [5/5] Training finished ---"

echo "Check the gnn_model/weights folder to see the generated models."
