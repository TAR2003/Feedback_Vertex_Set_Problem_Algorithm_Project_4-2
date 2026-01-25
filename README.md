# Feedback Vertex Set - Implementation & Experiments (C++) ✅

This repository contains a concise, easy-to-run C++ implementation of multiple algorithms for the Feedback Vertex Set (FVS) problem, dataset examples, and tooling to measure runtime and memory. The goal is accurate time and space measurements and easy reproducibility.

---

## Implemented algorithms

- `exact` (branching exact solver up to a given k): recursive branching on found cycles; decides if an FVS of size ≤ k exists.
- `twoapprox` (2-approximation): find a cycle, remove two vertices on that cycle, repeat.
- `greedy` (max-degree heuristic): repeatedly remove the highest-degree vertex until graph is acyclic.
- `ga` (genetic algorithm): randomized metaheuristic producing high-quality solutions for larger graphs.

---

## Build (Linux)

Requirements:
- CMake >= 3.10
- A modern C++ compiler (g++/clang supporting C++17)

Build steps:

```bash
mkdir -p build && cd build
cmake ..
make -j
```

The binary is `build/fvs`.

---

## Graph format

Simple edge-list (0-based vertex ids). Each non-comment line is `u v` for an undirected edge.
Comment lines may start with `#`.

Example files are in `data/graphs/` (e.g., `sample_triangle.txt`).

---

## Run examples

A convenience script runs a set of sample commands and outputs `results.csv`:

```bash
bash scripts/run_examples.sh
```

Example direct commands:

```bash
# Two-approx on a graph
./build/fvs -i data/graphs/sample_triangle.txt -a twoapprox -o results.csv

# Exact bounded search (k=1)
./build/fvs -i data/graphs/sample_triangle.txt -a exact -k 1 -o results.csv

# Genetic algorithm with custom params
./build/fvs -i data/graphs/sample_k4.txt -a ga --ga-pop 200 --ga-gen 200 -o results.csv
```

---

## Output & Measured Metrics

Each run appends a CSV row to the output file (default `results.csv`) with the following columns:

- `graph` : path to instance
- `algorithm` : `exact`, `twoapprox`, `greedy`, or `ga`
- `n` : number of vertices
- `m` : number of edges
- `k_or_` : parameter for exact (or `-`)
- `time_ms` : wall-clock runtime in milliseconds (measured with high-resolution timer)
- `mem_kb` : peak resident set size in kilobytes (measured using `getrusage` ru_maxrss)
- `fvs_size` : size of returned vertex set
- `valid` : `1` if returned set leaves graph acyclic, else `0`
- `remaining_nodes` : number of nodes left in the reduced graph

Notes on measurement accuracy:
- Time: measured by high-resolution timer (std::chrono). For short runs, run multiple repeats and take median.
- Memory: measured via `getrusage(RUSAGE_SELF).ru_maxrss` (kilobytes). This is portable on Linux but may differ across platforms.

---

## How to measure performance (recommended protocol) ⚙️

1. For randomized algorithms (`ga`) run several repeats (e.g., 10) with different seeds to compute mean and standard deviation. Use the `--ga-pop` and `--ga-gen` options to tune runtime/quality.
2. For deterministic algorithms run once per instance but use multiple instances to get robust statistics.
3. For short runs (<10 ms) increase instance size so runtime is more meaningful or repeat the run many times and average.
4. To report memory use, prefer the `mem_kb` values as an approximate peak resident memory.

Tip: scripts and the CSV output are set up so you can import into Pandas/R for plotting (runtime vs. n, boxplots of solutions, histograms of fvs_size, etc.).

---

## Extending (what to add next)

- Add more graph generators and a dataset folder with reproducible instance lists.
- Add a timeout wrapper to cap long exact solver runs.
- Add richer GA fitness and local search hybrid.
- Add unit tests for correctness and larger-benchmark integration.

---

If you want, I can: add more graph generators, add a driver to run a whole benchmark suite, or implement a smarter FPT iterative compression routine. Which would you like next? 🚀
