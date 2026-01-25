#!/usr/bin/env bash
set -euo pipefail

# Build
mkdir -p build && cd build
cmake ..
make -j
cd ..

BIN=./build/fvs
OUT=results.csv
rm -f $OUT

echo "Running examples..."
$BIN -i data/graphs/sample_triangle.txt -a twoapprox -o $OUT
$BIN -i data/graphs/sample_triangle.txt -a exact -k 1 -o $OUT
$BIN -i data/graphs/sample_k4.txt -a twoapprox -o $OUT
$BIN -i data/graphs/sample_k4.txt -a greedy -o $OUT
$BIN -i data/graphs/sample_k4.txt -a ga --ga-pop 200 --ga-gen 200 -o $OUT

echo "Done. Results in $OUT"
