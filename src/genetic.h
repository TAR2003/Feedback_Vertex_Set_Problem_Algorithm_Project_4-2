#pragma once
#include "graph.h"
#include <vector>

struct GA_Params {
    int population = 100;
    int generations = 300;
    double crossover_rate = 0.8;
    double mutation_rate = 0.05;
    int tournament_k = 3;
    unsigned seed = 42;
};

std::vector<int> genetic_fvs(const Graph &G, const GA_Params &params, bool verbose=false);
