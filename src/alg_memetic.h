#pragma once
#include "graph.h"
#include <vector>

/**
 * @file alg_memetic.h
 * @brief Memetic Algorithm (GA + Local Search) for FVS
 * 
 * Combines evolutionary search with problem-specific local optimization
 * Particularly effective for large-scale instances (n > 1000)
 * 
 * Key Components:
 * 1. Smart Initialization: Mix of greedy and random solutions
 * 2. FVS-Aware Crossover: Preserve cycle-breaking structure
 * 3. Centrality-Based Mutation: Target high-impact vertices
 * 4. Local Search: Hill-climbing to refine solutions
 * 5. Diversity Management: Maintain population variety
 * 
 * Innovations:
 * - Cycle-preserving crossover operator
 * - Hybrid with kernelization for preprocessing
 * - Adaptive mutation rates
 * - Multi-objective optimization (size + robustness)
 * 
 * Reference: Moscato (1989), Bäck & Fogel (1997)
 */

struct Memetic_Params {
    int population = 100;
    int generations = 300;
    double crossover_rate = 0.8;
    double mutation_rate = 0.05;
    int tournament_k = 3;
    int local_search_iterations = 10;
    double elite_ratio = 0.1;  // Keep top 10% unchanged
    bool use_local_search = true;
    bool smart_initialization = true;
    unsigned seed = 42;
};

/**
 * @brief Memetic Algorithm for FVS
 * 
 * @param G Input graph
 * @param params Algorithm parameters
 * @param verbose Print progress
 * @return FVS solution
 */
std::vector<int> memetic_fvs(const Graph &G, const Memetic_Params &params, 
                             bool verbose = false);

/**
 * @brief Local search to improve FVS solution
 * 
 * Strategies:
 * - Remove vertices and check if still valid
 * - Swap vertices (remove + add different ones)
 * - 2-opt style exchanges
 * 
 * @param G Input graph
 * @param solution Current FVS
 * @param iterations Max local search iterations
 * @return Improved FVS
 */
std::vector<int> local_search(const Graph &G, const std::vector<int> &solution, 
                              int iterations);

/**
 * @brief Smart initialization using mix of strategies
 * 
 * @param G Input graph
 * @param population_size Number of solutions to generate
 * @param seed Random seed
 * @return Initial population
 */
std::vector<std::vector<int>> smart_initialization(const Graph &G, 
                                                   int population_size,
                                                   unsigned seed);

/**
 * @brief Cycle-aware crossover operator
 * 
 * Preserves important cycle-breaking vertices from both parents
 * 
 * @param parent1 First parent
 * @param parent2 Second parent
 * @param G Graph context
 * @return Child solution
 */
std::vector<int> cycle_aware_crossover(const std::vector<int> &parent1,
                                       const std::vector<int> &parent2,
                                       const Graph &G);

/**
 * @brief Adaptive mutation based on vertex centrality
 * 
 * More likely to mutate high-degree vertices (more impact)
 * 
 * @param solution Current solution
 * @param G Graph context
 * @param mutation_rate Base mutation rate
 * @return Mutated solution
 */
std::vector<int> adaptive_mutation(const std::vector<int> &solution,
                                   const Graph &G,
                                   double mutation_rate);

/**
 * @brief Evaluate solution quality
 * 
 * @param G Graph
 * @param solution FVS to evaluate
 * @param is_valid Output: whether solution is valid
 * @return Fitness score (lower is better)
 */
int evaluate_solution(const Graph &G, const std::vector<int> &solution, 
                     bool &is_valid);

/**
 * @brief Convert chromosome (bitvector) to vertex list
 * 
 * @param chromosome Binary encoding
 * @return Vertex list
 */
std::vector<int> chromosome_to_vertices(const std::vector<char> &chromosome);

/**
 * @brief Convert vertex list to chromosome (bitvector)
 * 
 * @param vertices Vertex list
 * @param n Graph size
 * @return Binary encoding
 */
std::vector<char> vertices_to_chromosome(const std::vector<int> &vertices, int n);
