#pragma once
#include "graph.h"
#include <vector>
#include <string>
#include <sys/resource.h>

struct MeasureResult {
    double runtime_ms;
    long memory_kb; // max RSS
};

MeasureResult measure_function_runtime(std::function<void()> f);

// Validity check: is G - removed vertices acyclic?
bool is_acyclic_after_removal(const Graph &G, const std::vector<char> &removed, int &remaining_nodes);

// helpers
std::vector<int> read_subset_from_file(const std::string &path); // optional
