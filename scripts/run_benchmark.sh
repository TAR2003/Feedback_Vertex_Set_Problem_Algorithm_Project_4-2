#!/bin/bash

# Comprehensive benchmark script for FVS algorithms
# Runs all implemented algorithms on generated test graphs

OUTPUT_DIR="benchmark_results"
GRAPH_DIR="data/graphs"
RESULTS_CSV="$OUTPUT_DIR/all_results.csv"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clear previous results
rm -f "$RESULTS_CSV"

echo "========================================="
echo "FVS Algorithm Benchmark Suite"
echo "========================================="
echo ""

# Array of algorithms to test
ALGORITHMS=("twoapprox" "greedy" "ic" "kernelbst" "ga" "memetic")
K_VALUES=(5 10 15 20)

# Find all graph files
GRAPH_FILES=$(find "$GRAPH_DIR" -name "*.txt" | sort)

if [ -z "$GRAPH_FILES" ]; then
    echo "Error: No graph files found in $GRAPH_DIR"
    echo "Run ./generate_graphs $GRAPH_DIR first"
    exit 1
fi

TOTAL_GRAPHS=$(echo "$GRAPH_FILES" | wc -l)
echo "Found $TOTAL_GRAPHS graph files"
echo ""

counter=0

# Run benchmarks
for graph in $GRAPH_FILES; do
    counter=$((counter + 1))
    graph_name=$(basename "$graph")
    echo "[$counter/$TOTAL_GRAPHS] Testing $graph_name..."
    
    # Test approximation and heuristic algorithms (no k parameter)
    for alg in "twoapprox" "greedy" "ga" "memetic"; do
        echo "  Running $alg..."
        
        if [ "$alg" = "ga" ] || [ "$alg" = "memetic" ]; then
            # Run with reasonable population/generation for benchmarking
            ./build/fvs -i "$graph" -a "$alg" -o "$RESULTS_CSV" \
                --ga-pop 50 --ga-gen 100 2>/dev/null
        else
            ./build/fvs -i "$graph" -a "$alg" -o "$RESULTS_CSV" 2>/dev/null
        fi
    done
    
    # Test exact algorithms (with k parameter)
    # Only test on smaller graphs (n <= 50) to avoid long runtimes
    graph_size=$(./build/fvs -i "$graph" -a "greedy" -o /dev/null 2>&1 | grep -oP 'n=\K\d+' || echo "100")
    
    if [ "$graph_size" -le 50 ]; then
        for alg in "ic" "kernelbst"; do
            for k in "${K_VALUES[@]}"; do
                echo "  Running $alg with k=$k..."
                timeout 60s ./build/fvs -i "$graph" -a "$alg" -k "$k" -o "$RESULTS_CSV" 2>/dev/null || \
                    echo "    (timeout or error)"
            done
        done
    else
        echo "  Skipping exact algorithms (graph too large: n=$graph_size)"
    fi
    
    echo ""
done

echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
echo "Results saved to: $RESULTS_CSV"
echo ""
echo "Summary:"
wc -l "$RESULTS_CSV"
echo ""
echo "You can analyze results using:"
echo "  - Excel/LibreOffice Calc"
echo "  - Python pandas: pd.read_csv('$RESULTS_CSV')"
echo "  - R: read.csv('$RESULTS_CSV')"
echo ""
