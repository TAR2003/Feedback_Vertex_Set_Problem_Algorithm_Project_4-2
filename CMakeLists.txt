cmake_minimum_required(VERSION 3.10)
project(fvs_project VERSION 0.1 LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-O3 -march=native -g)

add_executable(fvs
    src/main.cpp
    src/graph.cpp
    src/utils.cpp
    src/alg_exact.cpp
    src/alg_approx.cpp
    src/genetic.cpp
    src/alg_iterative_compression.cpp
    src/alg_kernelization.cpp
    src/alg_bounded_search_tree.cpp
    src/alg_memetic.cpp
    src/graph_generators.cpp
)

target_include_directories(fvs PRIVATE src)

# Tool to generate benchmark graphs
add_executable(generate_graphs
    src/generate_graphs.cpp
    src/graph.cpp
    src/graph_generators.cpp
)

target_include_directories(generate_graphs PRIVATE src)

# Install targets (optional)
install(TARGETS fvs generate_graphs RUNTIME DESTINATION bin)
