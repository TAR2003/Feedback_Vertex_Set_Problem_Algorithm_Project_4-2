#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include "graph_generators.h"

/**
 * @file generate_graphs.cpp
 * @brief Tool to generate benchmark graphs for FVS experiments
 * 
 * Usage: ./generate_graphs <output_dir>
 * 
 * Generates comprehensive test suite with various graph types
 */

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <output_dir>\n";
        std::cerr << "Generates benchmark graphs for FVS experiments\n";
        return 1;
    }
    
    std::string output_dir = argv[1];
    
    // Create output directory if it doesn't exist
    std::filesystem::create_directories(output_dir);
    
    std::cout << "Generating benchmark graphs in " << output_dir << "...\n";
    
    // Generate comprehensive benchmark suite
    generate_benchmark_suite(output_dir, 42);
    
    std::cout << "Benchmark generation complete!\n";
    std::cout << "Generated graphs include:\n";
    std::cout << "  - Erdős-Rényi random graphs (various sizes and densities)\n";
    std::cout << "  - Barabási-Albert scale-free graphs\n";
    std::cout << "  - Grid graphs\n";
    std::cout << "  - Random trees (sanity checks)\n";
    std::cout << "  - Cycle-heavy graphs\n";
    
    return 0;
}
