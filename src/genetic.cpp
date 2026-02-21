#include "genetic.h"
#include "utils.h"
#include "alg_approx.h"
#include <random>
#include <algorithm>
#include <numeric>
#include <bitset>
#include <stack>

static int evaluate_fitness(const Graph &G, const std::vector<char> &removed, int &violations) {
    // violations = number of backedges found across DFSs
    int n = G.n; violations = 0;
    std::vector<int> state(n,0), parent(n,-1);
    std::function<void(int)> dfs = [&](int s){
        std::stack<int> st; st.push(s); parent[s]=-1;
        while(!st.empty()){
            int u = st.top(); st.pop();
            if (state[u]==0){
                state[u]=1;
                for (int v: G.adj[u]){
                    if (removed[v]) continue;
                    if (state[v]==0){ parent[v]=u; st.push(v); }
                    else if (v!=parent[u]) violations++;
                }
            }
        }
    };
    for (int i=0;i<n;++i){ if (removed[i] || state[i]) continue; dfs(i);}    
    int removed_count = 0; for (int i=0;i<n;++i) if (removed[i]) removed_count++;
    int penalty = violations * (n + 1); // heavily penalize cycles
    return removed_count + penalty;
}

std::vector<int> genetic_fvs(const Graph &G, const GA_Params &params, bool verbose){
    int n = G.n;
    std::mt19937 rng(params.seed);
    std::uniform_real_distribution<double> uni(0.0,1.0);
    // population: vector<vector<char>> chromosomes
    std::vector<std::vector<char>> pop(params.population, std::vector<char>(n,0));
    // initialize with random deletions biased to small sets
    for (int i=0;i<params.population;++i){
        for (int v=0; v<n; ++v) if (uni(rng) < 0.05) pop[i][v]=1; else pop[i][v]=0;
    }

    auto tournament = [&](auto &fitnesses){
        std::uniform_int_distribution<int> pick(0, params.population-1);
        int best = pick(rng);
        for (int tt=1; tt<params.tournament_k; ++tt){ int cand = pick(rng); if (fitnesses[cand] < fitnesses[best]) best=cand; }
        return best;
    };

    std::vector<int> best_solution;
    int best_score = 1e9;

    for (int gen = 0; gen < params.generations; ++gen){
        // evaluate
        std::vector<int> fitnesses(params.population);
        std::vector<int> violations(params.population);
        for (int i=0;i<params.population;++i){ fitnesses[i] = evaluate_fitness(G, pop[i], violations[i]); }
        // track best feasible solution (violations==0)
        for (int i=0;i<params.population;++i){
            if (violations[i]==0){ int sz=0; for (auto c: pop[i]) if (c) ++sz; if (sz < best_score){ best_score = sz; best_solution.clear(); for (int v=0;v<n;++v) if (pop[i][v]) best_solution.push_back(v); } }
        }
        // produce new population
        std::vector<std::vector<char>> newpop;
        while ((int)newpop.size() < params.population){
            int a = tournament(fitnesses), b = tournament(fitnesses);
            std::vector<char> child(n);
            // crossover
            if (uni(rng) < params.crossover_rate){
                for (int v=0; v<n; ++v) child[v] = (uni(rng) < 0.5) ? pop[a][v] : pop[b][v];
            } else child = pop[a];
            // mutation
            for (int v=0; v<n; ++v) if (uni(rng) < params.mutation_rate) child[v] = 1 - child[v];
            newpop.push_back(child);
        }
        pop.swap(newpop);
        if (verbose && gen % 50 == 0) {
            if (best_solution.empty()) printf("[GA] gen=%d best=none\n", gen);
            else printf("[GA] gen=%d best_size=%d\n", gen, (int)best_solution.size());
        }
    }

    if (best_solution.empty()){
        // fall back to greedy
        auto greedy = greedy_max_degree(G);
        return greedy;
    }
    return best_solution;
}
