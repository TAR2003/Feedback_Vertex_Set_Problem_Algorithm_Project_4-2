#!/usr/bin/env bash
set -euo pipefail

# 1. Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

DATASET_SLUG="tawkirazizrahman/fvs-synthetic-dataset-20k"
TARGET_DIR="./data/synthetic"

echo "--- [1/4] Installing dependencies and building the C++ engine ---"
# Ensure your build_engine.py is executable or run via python
python3 build_engine.py

echo "--- [2/4] Downloading Kaggle Dataset (Public) ---"
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
print(f'✅ Dataset moved to {target}')
"

# You can also generate a new synthetic benchmark dataset locally with:
# python data/setup_benchmark_inputs.py --total-undirected 100 --total-directed 100 --seed 7

echo "--- [3/4] Running the default pipeline ---"
python3 experiments/run_pipeline.py --mode all --algo ALL --timeout 30

echo "--- [4/4] Pipeline finished ---"

cat <<'EOF'

To customize the run, edit run.sh or invoke the pipeline directly:
  python3 experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50
  python3 experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100
EOF