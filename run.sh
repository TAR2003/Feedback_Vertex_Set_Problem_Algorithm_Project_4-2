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
echo "--- Remove all the previous csv files the results folder ---"
rm -f ./results/*.csv
echo "--- [4/5] Running the default pipeline ---"
python3 experiments/run_pipeline.py --mode all --algo ALL --timeout 30

echo "--- [5/5] check the fvs ---"
python3 experiments/brute_force.py 

python3 experiments/fvs_checker.py

rm -f directed_results/*.csv

mv results/directed_* directed_results/

python3 directed_results/evaluate_fvs_scores.py

rm -f undirected_results/*.csv

mv results/undirected_* undirected_results/

python3 undirected_results/evaluate_fvs_scores.py



echo "--- running the pace pipeline ---"
python3 experiments/benchmark_directed.py --algo MA --test data/pace2022/ --pop 20 --gens 100 --timeout $PACE_TIMEOUT
python3 experiments/benchmark_directed.py --algo KMA --test data/pace2022/ --pop 20 --gens 100 --timeout $PACE_TIMEOUT
python3 experiments/benchmark_directed.py --algo GNN-KMA --test data/pace2022/ --pop 20 --gens 100 --timeout $PACE_TIMEOUT
python3 experiments/benchmark_directed.py --algo GNN-KMA-2 --test data/pace2022/ --pop 20 --gens 100 --timeout $PACE_TIMEOUT

rm -f ./paceresults/*.csv
cp pace2022_winner.csv paceresults/
mv results/* paceresults/
python3 paceresults/evaluate_fvs_scores.py
echo "--- [5/5] Pipeline finished ---"

cat <<'EOF'
5/5] Pipeline finished ---"

cat <<'EOF'

To customize the run, edit run.sh or invoke the pipeline directly:
  python3 experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50
  python3 experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100
EOF