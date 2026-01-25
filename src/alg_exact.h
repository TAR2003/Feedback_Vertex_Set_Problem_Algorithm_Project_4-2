#pragma once
#include "graph.h"
#include <vector>

// Exact branching solver: decide whether there is an FVS of size <= k.
// If found, returns true and fills fvs_out with removed vertex IDs.
bool exact_fvs_bounded(const Graph &G, int k, std::vector<int> &fvs_out);

// A simple exact solver that attempts to find the minimum FVS (brute force up to limit)
int exact_fvs_min(const Graph &G, int k_limit, std::vector<int> &fvs_out);
