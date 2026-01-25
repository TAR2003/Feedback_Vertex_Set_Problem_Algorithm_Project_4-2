#pragma once
#include "graph.h"
#include <vector>

// 2-approximation algorithm (cycle-based): removes two vertices per found cycle
std::vector<int> two_approximation(const Graph &G);

// Greedy max-degree heuristic
std::vector<int> greedy_max_degree(const Graph &G);
