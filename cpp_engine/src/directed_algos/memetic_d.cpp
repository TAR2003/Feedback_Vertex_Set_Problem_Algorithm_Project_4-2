/**
 * @file memetic_d.cpp
 * @brief Memetic Algorithm for directed FVS (DFVS).
 *
 * Identical structure to the undirected version but uses:
 *   - has_directed_cycle() instead of has_cycle()
 *   - in_degree + out_degree for the greedy ordering heuristic
 *
 * The main algorithmic difference is the repair operator:
 * In the directed case, we prioritize vertices with high min(in, out) degree,
 * because such vertices participate in many directed cycles and their removal
 * is most likely to eliminate cycles efficiently.
 *
 * Population diversity is maintained by initializing half the population
 * with the greedy heuristic and half with random repair.
 */

#include "directed_fvs.h"
#include <vector>
#include <algorithm>
#include <random>
#include <functional>
#include <numeric>

// ─── Helpers ─────────────────────────────────────────────────────────────────

using IndividualD = std::vector<int>; ///< Binary vector of length n

static int dfvs_size(const IndividualD &x)
{
    return std::count(x.begin(), x.end(), 1);
}

static int dcycles_remaining(const DirectedGraph &g, const IndividualD &x)
{
    DirectedGraph g_copy = g.copy();
    for (int v = 0; v < g.n; ++v)
    {
        if (x[v] == 1 && g_copy.is_active(v))
        {
            std::vector<std::pair<int, int>> dummy;
            g_copy.deactivate_full(v, dummy);
        }
    }
    return g_copy.has_directed_cycle() ? 1 : 0;
}

static int dfitness(const DirectedGraph &g, const IndividualD &x)
{
    return dfvs_size(x) + g.n * dcycles_remaining(g, x);
}

// ─── Directed Local Search ────────────────────────────────────────────────────

/**
 * Remove FVS vertices one at a time as long as the remaining set is still
 * a valid DFVS.  Use a randomised order to escape local optima.
 */
static void directed_local_search(const DirectedGraph &g, IndividualD &x,
                                  std::mt19937 &rng)
{
    int n = g.n;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);

    bool improved = true;
    while (improved)
    {
        improved = false;
        for (int v : order)
        {
            if (x[v] == 0)
                continue;
            x[v] = 0;
            if (dcycles_remaining(g, x) == 0)
            {
                improved = true;
            }
            else
            {
                x[v] = 1;
            }
        }
        if (improved)
            std::shuffle(order.begin(), order.end(), rng);
    }
}

// ─── Genetic Operators ───────────────────────────────────────────────────────

static IndividualD d_crossover(const IndividualD &a, const IndividualD &b,
                               std::mt19937 &rng)
{
    int n = static_cast<int>(a.size());
    IndividualD child(n);
    std::uniform_int_distribution<int> coin(0, 1);
    for (int i = 0; i < n; ++i)
        child[i] = coin(rng) ? a[i] : b[i];
    return child;
}

static void d_mutate(IndividualD &x, std::mt19937 &rng)
{
    int n = static_cast<int>(x.size());
    std::uniform_real_distribution<double> prob(0.0, 1.0);
    double p_mut = 1.0 / n;
    for (int i = 0; i < n; ++i)
        if (prob(rng) < p_mut)
            x[i] ^= 1;
}

static int d_tournament(const std::vector<int> &scores,
                        std::mt19937 &rng, int t = 3)
{
    int sz = static_cast<int>(scores.size());
    std::uniform_int_distribution<int> idx(0, sz - 1);
    int best = idx(rng);
    for (int i = 1; i < t; ++i)
    {
        int j = idx(rng);
        if (scores[j] < scores[best])
            best = j;
    }
    return best;
}

// ─── Population initialisation ───────────────────────────────────────────────

static std::vector<IndividualD> d_init_population(
    const DirectedGraph &g, int pop_size, std::mt19937 &rng)
{

    int n = g.n;
    std::vector<IndividualD> pop;

    // Greedy seed: sort by min(in, out) descending
    {
        IndividualD seed(n, 0);
        std::vector<int> order(n);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int a, int b)
                  { return std::min(g.in_degree(a), g.out_degree(a)) >
                           std::min(g.in_degree(b), g.out_degree(b)); });
        DirectedGraph g_tmp = g.copy();
        for (int v : order)
        {
            if (!g_tmp.has_directed_cycle())
                break;
            seed[v] = 1;
            std::vector<std::pair<int, int>> dummy;
            g_tmp.deactivate_full(v, dummy);
        }
        directed_local_search(g, seed, rng);
        pop.push_back(seed);
    }

    std::uniform_real_distribution<double> prob(0.0, 1.0);
    while (static_cast<int>(pop.size()) < pop_size)
    {
        IndividualD ind(n, 0);
        for (int v = 0; v < n; ++v)
            ind[v] = (prob(rng) < 0.5) ? 1 : 0;
        // Repair
        if (dcycles_remaining(g, ind) > 0)
        {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::shuffle(order.begin(), order.end(), rng);
            for (int v : order)
            {
                if (dcycles_remaining(g, ind) == 0)
                    break;
                ind[v] = 1;
            }
        }
        directed_local_search(g, ind, rng);
        pop.push_back(ind);
    }
    return pop;
}

// ─── Main ────────────────────────────────────────────────────────────────────

std::vector<int> solve_directed_MA(int n,
                                   const std::vector<std::pair<int, int>> &edges,
                                   int pop_size, int max_gens)
{

    if (n == 0)
        return {};

    DirectedGraph g(n);
    for (auto &[u, v] : edges)
    {
        if (u >= 0 && u < n && v >= 0 && v < n)
            g.add_edge(u, v);
    }

    if (!g.has_directed_cycle())
        return {};

    std::mt19937 rng(42);

    auto pop = d_init_population(g, pop_size, rng);
    std::vector<int> scores(pop_size);
    for (int i = 0; i < pop_size; ++i)
        scores[i] = dfitness(g, pop[i]);

    int best_idx = static_cast<int>(
        std::min_element(scores.begin(), scores.end()) - scores.begin());
    IndividualD best_ind = pop[best_idx];
    int best_score = scores[best_idx];

    for (int gen = 0; gen < max_gens; ++gen)
    {
        int p1 = d_tournament(scores, rng);
        int p2 = d_tournament(scores, rng);

        IndividualD child = d_crossover(pop[p1], pop[p2], rng);
        d_mutate(child, rng);

        // Repair
        if (dcycles_remaining(g, child) > 0)
        {
            std::vector<int> order(n);
            std::iota(order.begin(), order.end(), 0);
            std::sort(order.begin(), order.end(), [&](int a, int b)
                      { return (g.in_degree(a) + g.out_degree(a)) >
                               (g.in_degree(b) + g.out_degree(b)); });
            for (int v : order)
            {
                if (dcycles_remaining(g, child) == 0)
                    break;
                child[v] = 1;
            }
        }

        directed_local_search(g, child, rng);

        int child_score = dfitness(g, child);
        int worst = static_cast<int>(
            std::max_element(scores.begin(), scores.end()) - scores.begin());
        if (child_score < scores[worst])
        {
            pop[worst] = child;
            scores[worst] = child_score;
        }

        if (child_score < best_score)
        {
            best_score = child_score;
            best_ind = child;
        }
    }

    std::vector<int> result;
    for (int v = 0; v < n; ++v)
        if (best_ind[v] == 1)
            result.push_back(v);
    return result;
}

std::vector<int> solve_directed_KME(int n,
                                    const std::vector<std::pair<int, int>> &edges,
                                    int pop_size, int max_gens)
{
    // Keep KME symbol available for pybind/import stability.
    // This currently delegates to the memetic solver.
    return solve_directed_MA(n, edges, pop_size, max_gens);
}