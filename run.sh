#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/3] Installing dependencies and building the C++ engine"
python build_engine.py

echo "[2/3] Running the default pipeline"
python experiments/run_pipeline.py --mode all --algo ALL --total-undirected 100 --total-directed 100

echo "[3/3] Pipeline finished"

cat <<'EOF'
To customize the run, edit run.sh or invoke the pipeline directly:
  python experiments/run_pipeline.py --mode directed --algo MA --include-pace --total-directed 50
  python experiments/run_pipeline.py --mode undirected --algo IC --prepare-only --total-undirected 100
EOF
