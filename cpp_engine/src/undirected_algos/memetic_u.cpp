/**
 * @file memetic_u.cpp
 * @brief Memetic Algorithm (Genetic Algorithm + Local Search) for undirected FVS.
 *
 * ── Algorithm Overview ────────────────────────────────────────────────────────
 *
 * A Memetic Algorithm combines a population-based Genetic Algorithm (GA)
 * with individual local search refinement.  This is especially effective for
 * FVS because:
 *   - GA explores the search space globally (crossover mixes two good solutions)
 *   - Local search exploits each solution locally (remove safe vertices)
 *
 * ── Representation ───────────────────────────────────────────────────────────
 *
 * Each individual is a binary vector `x` of length n:
 *   x[v] = 1 → vertex v is in the FVS
 *   x[v] = 0 → vertex v is NOT in the FVS (stays in G - FVS)
 *
 * ── Fitness Function ─────────────────────────────────────────────────────────
 *
 * fitness(x) = |FVS(x)| + α × cycles_remaining(x)
 *
 * where cycles_remaining(x) = number of connected components of G - FVS(x)
 * that still contain a cycle (measured by DFS).
 * α = n (large penalty ensures infeasible solutions rank below feasible ones).
 *
 * Lower fitness = better solution.
 *
 * ── Operators ────────────────────────────────────────────────────────────────
 *
 * Crossover (uniform): for each bit, randomly inherit from parent A or B.
 * Mutation:            flip each bit with probability p_mut = 1/n.
 * Local Search:        iteratively remove vertices from FVS while still valid.
 * Selection:           tournament selection (tournament size = 3).
 */

#include "undirected_fvs.h"
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>
#include <climits>
#include <iostream>
#include <chrono>

// ─── Helpers ─────────────────────────────────────────────────────────────────

using Individual = std::vector<int>; ///< Binary vector of length n

/** Count active vertices set to 1 in individual `x`. */
static int fvs_size(const Individual &x)
{
    return std::count(x.begin(), x.end(), 1);
}

/**
 * Decode an individual and check feasibility.
 * Removes all FVS vertices from graph g, then checks for cycles.
 * @return 0 if feasible (no cycles remain), number of cycle-containing
 *         components otherwise.
 */
static int cycles_remaining(const UndirectedGraph &g, const Individual &x)
{
    UndirectedGraph g_copy = g.copy();
    for (int v = 0; v < g.n; ++v)
    {
        if (x[v] == 1 && g_copy.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            g_copy.deactivate_full(v, dummy);
        }
    }
    // Count how many times DFS finds a cycle
    // (Approximate: just return 1 if any cycle, 0 otherwise)
    return g_copy.has_cycle() ? 1 : 0;
}

/**
 * Fitness: lower is better.
 * Infeasible solutions are penalised by n (the maximum FVS size).
 */
static int fitness(const UndirectedGraph &g, const Individual &x)
{
    int cyc = cycles_remaining(g, x);
    return fvs_size(x) + g.n * cyc;
}

// ─── Local Search ─────────────────────────────────────────────────────────────

/**
 * Greedy local search: try removing each FVS vertex; keep removal if valid.
 * Repeat until no improvement.  O(|FVS| × (n + m)) per pass.
 */
static void local_search(const UndirectedGraph &g, Individual &x)
{
    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int v = 0; v < g.n; ++v)
        {
            if (x[v] == 0)
                continue;
            x[v] = 0;
            if (cycles_remaining(g, x) == 0)
            {
                improved = true; // keep removal
            }
            else
            {
                x[v] = 1; // revert
            }
        }
    }
}

// ─── Genetic Operators ───────────────────────────────────────────────────────

/**
 * Uniform crossover: each gene inherited randomly from one of two parents.
 */
static Individual crossover(const Individual &a, const Individual &b,
                            std::mt19937 &rng)
{
    int n = static_cast<int>(a.size());
    Individual child(n);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < n; ++i)
    {
        child[i] = coin(rng) ? a[i] : b[i];
    }
    return child;
}

/**
 * Bit-flip mutation: each gene flipped with probability p_mut = 1/n.
 * Guarantees at least one flip on average per individual.
 */
static void mutate(Individual &x, std::mt19937 &rng)
{
    int n = static_cast<int>(x.size());
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    double p_mut = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i)
    {
        if (prob(rng) < p_mut)
            x[i] ^= 1;
    }
}

/**
 * Tournament selection: pick `t` random individuals, return the best.
 * @param pop     Current population
 * @param scores  Pre-computed fitness values
 * @param t       Tournament size
 */
static int tournament_select(const std::vector<int> &scores,
                             std::mt19937 &rng, int t = 3)
{
    int pop_size = static_cast<int>(scores.size());
    std::uniform_int_distribution<int> rand_idx(0, pop_size - 1);
    int best_idx = rand_idx(rng);
    int best_score = scores[best_idx];
    for (int i = 1; i < t; ++i)
    {
        int idx = rand_idx(rng);
        if (scores[idx] < best_score)
        {
            best_score = scores[idx];
            best_idx = idx;
        }
    }
    return best_idx;
}

// ─── Initial Population ───────────────────────────────────────────────────────

/**
 * Initialise population:
 *   - First individual: greedy by degree (high-degree first)
 *   - Remaining: random binary, repaired by adding vertices until feasible
 */
static std::vector<Individual> init_population(
    const UndirectedGraph &g, int pop_size, std::mt19937 &rng)
{

    int n = g.n;
    std::vector<Individual> pop;

    // ── Greedy seed ──────────────────────────────────────────────────────────
    {
        Individual seed(n, 0);
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b)
                  { return g.degree(a) > g.degree(b); });
        UndirectedGraph g_tmp = g.copy();
        for (int v : order)
        {
            if (!g_tmp.has_cycle())
                break;
            seed[v] = 1;
            std::vector<std::pair<int, int>> dummy;
            g_tmp.deactivate_full(v, dummy);
        }
        local_search(g, seed);
        pop.push_back(seed);
    }

    // ── Random individuals ───────────────────────────────────────────────────
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    while (static_cast<int>(pop.size()) < pop_size)
    {
        Individual ind(n, 0);
        // Random inclusion with probability 0.5
        for (int v = 0; v < n; ++v)
            ind[v] = (prob(rng) < 0.5) ? 1 : 0;
        // Repair: add vertices until feasible
        if (cycles_remaining(g, ind) > 0)
        {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);
            for (int v : order)
            {
                if (cycles_remaining(g, ind) == 0)
                    break;
                ind[v] = 1;
            }
        }
        local_search(g, ind);
        pop.push_back(ind);
    }
    return pop;
}

// ─── Main Memetic Algorithm ───────────────────────────────────────────────────

std::vector<int> solve_undirected_MA(int n,
                                     const std::vector<std::pair<int, int>> &edges,
                                     int pop_size, int max_gens,
                                     int patience,
                                     int max_time_seconds)
{

    if (n == 0)
        return {};

    UndirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    // Trivial case
    if (!g.has_cycle())
        return {};

    std::mt19937 rng(42); // fixed seed for reproducibility

    // ── Initialise population ────────────────────────────────────────────────
    std::vector<Individual> pop = init_population(g, pop_size, rng);

    // Compute initial fitness
    std::vector<int> scores(pop_size);
    for (int i = 0; i < pop_size; ++i)
        scores[i] = fitness(g, pop[i]);

    // Track best solution
    int best_idx = static_cast<int>(
        std::min_element(scores.begin(), scores.end()) - scores.begin());
    Individual best_ind = pop[best_idx];
    int best_score = scores[best_idx];
    int best_fitness_ever = INT_MAX;
    int gens_without_improvement = 0;
    int patience_limit = (patience > 0) ? patience : 20;
    const auto start_time = std::chrono::steady_clock::now();

    // ── Main loop ────────────────────────────────────────────────────────────
    for (int gen = 0; gen < max_gens; ++gen)
    {
        if (max_time_seconds > 0)
        {
            const auto elapsed_seconds = std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::steady_clock::now() - start_time)
                                             .count();
            if (elapsed_seconds >= max_time_seconds)
            {
                std::cout << "Hard time limit reached. Stopping early." << std::endl;
                break;
            }
        }

        // Select two parents by tournament
        int p1 = tournament_select(scores, rng);
        int p2 = tournament_select(scores, rng);

        // Crossover
        Individual child = crossover(pop[p1], pop[p2], rng);

        // Mutation
        mutate(child, rng);

        // Repair: if infeasible, add high-degree vertices until feasible
        if (cycles_remaining(g, child) > 0)
        {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      { return g.degree(a) > g.degree(b); });
            for (int v : order)
            {
                if (cycles_remaining(g, child) == 0)
                    break;
                child[v] = 1;
            }
        }

        // Local search refinement
        local_search(g, child);

        // Replace worst individual in population
        int child_score = fitness(g, child);
        int worst_idx = static_cast<int>(
            std::max_element(scores.begin(), scores.end()) - scores.begin());
        if (child_score < scores[worst_idx])
        {
            pop[worst_idx] = child;
            scores[worst_idx] = child_score;
        }

        // Keep global best individual by objective score.
        if (child_score < best_score)
        {
            best_score = child_score;
            best_ind = child;
        }

        // Early stopping is based on generation-best FVS size.
        int gen_best_idx = static_cast<int>(
            std::min_element(scores.begin(), scores.end()) - scores.begin());
        int gen_best_fvs_size = fvs_size(pop[gen_best_idx]);

        if (gen_best_fvs_size < best_fitness_ever)
        {
            best_fitness_ever = gen_best_fvs_size;
            gens_without_improvement = 0;

            // Keep returned solution aligned with the generation-best FVS size.
            if (scores[gen_best_idx] <= best_score)
            {
                best_score = scores[gen_best_idx];
                best_ind = pop[gen_best_idx];
            }
        }
        else
        {
            ++gens_without_improvement;
            if (gens_without_improvement >= patience_limit)
            {
                std::cout << "Early stopping triggered at generation "
                          << gen << std::endl;
                break;
            }
        }
    }

    // ── Decode best individual ────────────────────────────────────────────────
    std::vector<int> result;
    for (int v = 0; v < n; ++v)
    {
        if (best_ind[v] == 1)
            result.push_back(v);
    }
    return result;
}

std::vector<int> solve_undirected_KMA(int n,
                                      const std::vector<std::pair<int, int>> &edges,
                                      int pop_size, int max_gens,
                                      int patience,
                                      int max_time_seconds)
{
    if (n == 0)
        return {};

    UndirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    std::vector<int> forced;
    int k = n;
    kernelize_undirected(g, forced, k);

    std::vector<int> kernel_old_to_new(n, -1);
    std::vector<int> kernel_new_to_old;
    kernel_new_to_old.reserve(n);

    for (int v = 0; v < n; ++v)
    {
        if (!g.is_active(v))
            continue;
        kernel_old_to_new[v] = static_cast<int>(kernel_new_to_old.size());
        kernel_new_to_old.push_back(v);
    }

    std::vector<std::pair<int, int>> kernel_edges;
    for (int u : kernel_new_to_old)
    {
        int nu = kernel_old_to_new[u];
        for (int v : g.adj[u])
        {
            if (!g.is_active(v))
                continue;
            int nv = kernel_old_to_new[v];
            if (nu < nv)
                kernel_edges.push_back({nu, nv});
        }
    }

    std::vector<int> kernel_sol;
    if (!kernel_new_to_old.empty())
    {
        kernel_sol = solve_undirected_MA(
            static_cast<int>(kernel_new_to_old.size()),
            kernel_edges,
            pop_size,
            max_gens,
            patience,
            max_time_seconds);
    }

    std::vector<int> result = forced;
    result.reserve(forced.size() + kernel_sol.size());
    for (int kv : kernel_sol)
    {
        if (kv >= 0 && kv < static_cast<int>(kernel_new_to_old.size()))
            result.push_back(kernel_new_to_old[kv]);
    }

    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

std::vector<int> solve_undirected_KME(int n,
                                      const std::vector<std::pair<int, int>> &edges,
                                      int pop_size, int max_gens,
                                      int patience,
                                      int max_time_seconds)
{
    return solve_undirected_KMA(n, edges, pop_size, max_gens, patience, max_time_seconds);
}