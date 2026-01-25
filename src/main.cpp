#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <getopt.h>

#include "graph.h"
#include "utils.h"
#include "alg_exact.h"
#include "alg_approx.h"
#include "genetic.h"

void print_usage(){
    std::cout << "FVS Project - usage:\n";
    std::cout << "  ./fvs -i <graph_file> -a <alg> [options]\n";
    std::cout << "Algorithms: exact, twoapprox, greedy, ga\n";
    std::cout << "Options:\n";
    std::cout << "  -k <k>        : parameter k for exact bounded search\n";
    std::cout << "  -o <csv_out>  : write CSV result (default: results.csv)\n";
    std::cout << "  --ga-pop <p>  : GA population (default 100)\n";
    std::cout << "  --ga-gen <g>  : GA generations (default 300)\n";
}

int main(int argc, char** argv){
    std::string infile;
    std::string alg = "twoapprox";
    int k = 10;
    std::string csv_out = "results.csv";
    GA_Params ga_params;

    static struct option long_options[] = {
        {"ga-pop", required_argument, 0, 0},
        {"ga-gen", required_argument, 0, 0},
        {0,0,0,0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:a:k:o:h", long_options, &option_index)) != -1){
        if (opt==0){
            std::string name = long_options[option_index].name;
            if (name=="ga-pop") ga_params.population = std::stoi(optarg);
            if (name=="ga-gen") ga_params.generations = std::stoi(optarg);
            continue;
        }
        switch(opt){
            case 'i': infile = optarg; break;
            case 'a': alg = optarg; break;
            case 'k': k = std::stoi(optarg); break;
            case 'o': csv_out = optarg; break;
            case 'h': print_usage(); return 0;
            default: print_usage(); return 1;
        }
    }

    if (infile.empty()){ std::cerr << "Error: input file required (-i)\n"; print_usage(); return 1; }

    Graph G;
    try { G = Graph::from_edge_list_file(infile); }
    catch (std::exception &e){ std::cerr << "Failed to read graph: " << e.what() << "\n"; return 1; }

    std::vector<int> result;
    MeasureResult mr;
    if (alg=="exact"){
        mr = measure_function_runtime([&]{ exact_fvs_bounded(G, k, result); });
    } else if (alg=="twoapprox"){
        mr = measure_function_runtime([&]{ result = two_approximation(G); });
    } else if (alg=="greedy"){
        mr = measure_function_runtime([&]{ result = greedy_max_degree(G); });
    } else if (alg=="ga"){
        mr = measure_function_runtime([&]{ result = genetic_fvs(G, ga_params, true); });
    } else {
        std::cerr << "Unknown algorithm: " << alg << "\n"; return 1;
    }

    // Validate
    std::vector<char> removed(G.n, 0);
    for (int v: result) if (v>=0 && v < G.n) removed[v]=1;
    int remaining_nodes=0;
    bool valid = is_acyclic_after_removal(G, removed, remaining_nodes);

    std::ofstream out(csv_out, std::ios::app);
    if (out.tellp()==0){ out << "graph,algorithm,n,m,k_or_,time_ms,mem_kb,fvs_size,valid,remaining_nodes\n"; }
    auto nm = G.edge_count();
    out << infile << "," << alg << "," << nm.first << "," << nm.second << "," << (alg=="exact"?std::to_string(k):"-") << ",";
    out << mr.runtime_ms << "," << mr.memory_kb << "," << result.size() << "," << (valid?"1":"0") << "," << remaining_nodes << "\n";
    out.close();

    std::cout << "Algorithm: " << alg << " | time: " << mr.runtime_ms << " ms | mem: " << mr.memory_kb << " KB | fvs_size: " << result.size() << " | valid: " << (valid?"YES":"NO") << "\n";
    if (!valid) std::cerr << "Warning: returned set is not a valid FVS\n";

    return 0;
}
