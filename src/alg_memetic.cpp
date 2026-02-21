#include "alg_memetic.h"
#include "alg_approx.h"
#include "utils.h"
#include <random>
#include <algorithm>
#include <set>
#include <map>
#include <cmath>

/**
 * @brief Convert chromosome to vertex list
 */
std::vector<int> chromosome_to_vertices(const std::vector<char> &chromosome) {
    std::vector<int> vertices;
    for (int i = 0; i < (int)chromosome.size(); ++i) {
        if (chromosome[i]) vertices.push_back(i);
    }
    return vertices;
}

/**
 * @brief Convert vertex list to chromosome
 */
std::vector<char> vertices_to_chromosome(const std::vector<int> &vertices, int n) {
    std::vector<char> chromosome(n, 0);
    for (int v : vertices) {
        if (v >= 0 && v < n) chromosome[v] = 1;
    }
    return chromosome;
}

/**
 * @brief Evaluate solution quality
 */
int evaluate_solution(const Graph &G, const std::vector<int> &solution, 
                     bool &is_valid) {
    std::vector<char> removed(G.n, 0);
    for (int v : solution) {
        if (v >= 0 && v < G.n) removed[v] = 1;
    }
    
    int remaining = 0;
    is_valid = is_acyclic_after_removal(G, removed, remaining);
    
    // Fitness: size + huge penalty for invalid solutions
    int penalty = is_valid ? 0 : G.n * 100;
    return (int)solution.size() + penalty;
}

/**
 * @brief Local search using hill-climbing
 * 
 * Try to remove vertices one by one, keep if still valid
 */
std::vector<int> local_search(const Graph &G, const std::vector<int> &solution,
                              int iterations) {
    std::vector<int> current = solution;
    bool improved = true;
    int iter = 0;
    
    while (improved && iter < iterations) {
        improved = false;
        iter++;
        
        // Try removing each vertex
        for (size_t i = 0; i < current.size(); ++i) {
            std::vector<int> candidate = current;
            candidate.erase(candidate.begin() + i);
            
            bool valid;
            int candidate_score = evaluate_solution(G, candidate, valid);
            
            if (valid && candidate_score < (int)current.size()) {
                current = candidate;
                improved = true;
                break; // Restart with new solution
            }
        }
        
        // Try swapping vertices (remove one, add another)
        if (!improved && !current.empty()) {
            std::set<int> in_fvs(current.begin(), current.end());
            
            for (size_t i = 0; i < current.size(); ++i) {
                for (int v = 0; v < G.n; ++v) {
                    if (in_fvs.count(v)) continue;
                    
                    std::vector<int> candidate = current;
                    candidate[i] = v;
                    
                    bool valid;
                    evaluate_solution(G, candidate, valid);
                    
                    if (valid && candidate.size() < current.size()) {
                        current = candidate;
                        improved = true;
                        break;
                    }
                }
                if (improved) break;
            }
        }
    }
    
    return current;
}

/**
 * @brief Smart initialization with diverse strategies
 */
std::vector<std::vector<int>> smart_initialization(const Graph &G,
                                                   int population_size,
                                                   unsigned seed) {
    std::vector<std::vector<int>> population;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    
    // Strategy 1: Greedy (20% of population)
    int greedy_count = population_size / 5;
    for (int i = 0; i < greedy_count; ++i) {
        std::vector<int> greedy = greedy_max_degree(G);
        population.push_back(greedy);
    }
    
    // Strategy 2: 2-approximation (10% of population)
    int approx_count = population_size / 10;
    for (int i = 0; i < approx_count; ++i) {
        std::vector<int> approx = two_approximation(G);
        population.push_back(approx);
    }
    
    // Strategy 3: Random with varying densities (70% of population)
    int random_count = population_size - greedy_count - approx_count;
    for (int i = 0; i < random_count; ++i) {
        std::vector<int> random_sol;
        double density = 0.05 + uni(rng) * 0.15; // 5-20% vertices
        
        for (int v = 0; v < G.n; ++v) {
            if (uni(rng) < density) {
                random_sol.push_back(v);
            }
        }
        
        population.push_back(random_sol);
    }
    
    return population;
}

/**
 * @brief Cycle-aware crossover
 * 
 * Take union of parents, then apply local search to reduce
 */
std::vector<int> cycle_aware_crossover(const std::vector<int> &parent1,
                                       const std::vector<int> &parent2,
                                       const Graph &G) {
    std::set<int> child_set;
    
    // Strategy: take intersection + some from each parent
    std::set<int> p1_set(parent1.begin(), parent1.end());
    std::set<int> p2_set(parent2.begin(), parent2.end());
    
    // Add intersection (vertices in both parents)
    for (int v : parent1) {
        if (p2_set.count(v)) {
            child_set.insert(v);
        }
    }
    
    // Randomly add remaining vertices
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    
    for (int v : parent1) {
        if (!child_set.count(v) && uni(rng) < 0.5) {
            child_set.insert(v);
        }
    }
    
    for (int v : parent2) {
        if (!child_set.count(v) && uni(rng) < 0.5) {
            child_set.insert(v);
        }
    }
    
    return std::vector<int>(child_set.begin(), child_set.end());
}

/**
 * @brief Adaptive mutation based on vertex importance
 */
std::vector<int> adaptive_mutation(const std::vector<int> &solution,
                                   const Graph &G,
                                   double mutation_rate) {
    std::vector<int> mutated = solution;
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::uniform_int_distribution<int> vertex_dist(0, G.n - 1);
    
    // Remove some vertices
    mutated.erase(
        std::remove_if(mutated.begin(), mutated.end(),
                      [&](int v) { return uni(rng) < mutation_rate; }),
        mutated.end()
    );
    
    // Add some random vertices
    std::set<int> in_solution(mutated.begin(), mutated.end());
    int add_count = std::max(1, (int)(mutation_rate * G.n));
    
    for (int i = 0; i < add_count; ++i) {
        int v = vertex_dist(rng);
        if (!in_solution.count(v)) {
            mutated.push_back(v);
            in_solution.insert(v);
        }
    }
    
    return mutated;
}

/**
 * @brief Main Memetic Algorithm
 */
std::vector<int> memetic_fvs(const Graph &G, const Memetic_Params &params,
                             bool verbose) {
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    
    // Initialize population
    std::vector<std::vector<int>> population;
    
    if (params.smart_initialization) {
        population = smart_initialization(G, params.population, params.seed);
    } else {
        // Simple random initialization
        for (int i = 0; i < params.population; ++i) {
            std::vector<int> individual;
            for (int v = 0; v < G.n; ++v) {
                if (uni(rng) < 0.1) individual.push_back(v);
            }
            population.push_back(individual);
        }
    }
    
    // Track best solution
    std::vector<int> best_solution;
    int best_fitness = INT32_MAX;
    
    // Main evolution loop
    for (int gen = 0; gen < params.generations; ++gen) {
        // Evaluate population
        std::vector<int> fitnesses(population.size());
        std::vector<char> validities(population.size());  // Use char instead of bool to avoid proxy issues
        
        for (size_t i = 0; i < population.size(); ++i) {
            bool is_valid = false;
            fitnesses[i] = evaluate_solution(G, population[i], is_valid);
            validities[i] = is_valid ? 1 : 0;
            
            // Update best solution
            if (is_valid && (int)population[i].size() < best_fitness) {
                best_fitness = population[i].size();
                best_solution = population[i];
            }
        }
        
        // Sort by fitness for elitism
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                 [&](size_t a, size_t b) { return fitnesses[a] < fitnesses[b]; });
        
        // Create new population
        std::vector<std::vector<int>> new_population;
        
        // Elitism: keep top solutions
        int elite_count = std::max(1, (int)(params.elite_ratio * params.population));
        for (int i = 0; i < elite_count && i < (int)population.size(); ++i) {
            new_population.push_back(population[indices[i]]);
        }
        
        // Generate offspring
        std::uniform_int_distribution<size_t> pick(0, population.size() - 1);
        
        while ((int)new_population.size() < params.population) {
            // Tournament selection
            size_t parent1_idx = pick(rng);
            size_t parent2_idx = pick(rng);
            
            for (int t = 1; t < params.tournament_k; ++t) {
                size_t cand = pick(rng);
                if (fitnesses[cand] < fitnesses[parent1_idx]) parent1_idx = cand;
            }
            
            for (int t = 1; t < params.tournament_k; ++t) {
                size_t cand = pick(rng);
                if (fitnesses[cand] < fitnesses[parent2_idx]) parent2_idx = cand;
            }
            
            // Crossover
            std::vector<int> child;
            if (uni(rng) < params.crossover_rate) {
                child = cycle_aware_crossover(population[parent1_idx],
                                             population[parent2_idx], G);
            } else {
                child = population[parent1_idx];
            }
            
            // Mutation
            if (uni(rng) < params.mutation_rate) {
                child = adaptive_mutation(child, G, params.mutation_rate);
            }
            
            // Local search
            if (params.use_local_search && uni(rng) < 0.3) { // 30% chance
                child = local_search(G, child, params.local_search_iterations);
            }
            
            new_population.push_back(child);
        }
        
        population = new_population;
        
        // Verbose output
        if (verbose && gen % 50 == 0) {
            printf("[Memetic] gen=%d best_size=%d\n", gen, best_fitness);
        }
    }
    
    // Final local search on best solution
    if (params.use_local_search && !best_solution.empty()) {
        best_solution = local_search(G, best_solution, params.local_search_iterations * 2);
    }
    
    // Fallback to greedy if no valid solution found
    if (best_solution.empty()) {
        return greedy_max_degree(G);
    }
    
    return best_solution;
}
