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
)

target_include_directories(fvs PRIVATE src)

# Install target (optional)
install(TARGETS fvs RUNTIME DESTINATION bin)
