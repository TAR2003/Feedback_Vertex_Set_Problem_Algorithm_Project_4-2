cmake_minimum_required(VERSION 3.14)
project(cpp_engine CXX)

# ── C++ standard ──────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Release build by default (optimized) ─────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Optimization flags
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
endif()

# ── Find Python ───────────────────────────────────────────────────────────────
# Finds the Python interpreter and development headers.
# If you have multiple Python installations, set:
# cmake -DPYTHON_EXECUTABLE=/path/to/python3 ..
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Python3 found: ${Python3_EXECUTABLE}")
message(STATUS "Python3 version: ${Python3_VERSION}")

# ── Find pybind11 ─────────────────────────────────────────────────────────────
# Try system install first, fall back to pip-installed pybind11.
# To install: pip install pybind11
find_package(pybind11 QUIET)

if(NOT pybind11_FOUND)
    # Try to find via Python
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE PYBIND11_PYTHON_RESULT
    )

    if(PYBIND11_PYTHON_RESULT EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found!\n"
            "Install it with: pip install pybind11\n"
            "Or: conda install pybind11")
    endif()
endif()

message(STATUS "pybind11 version: ${pybind11_VERSION}")

# ── Source files ──────────────────────────────────────────────────────────────
# List every .cpp that needs to be compiled into the shared library.
# Note: kernelization_*.cpp are #included directly by exact_solver_*.cpp,
# so they should NOT appear here as independent translation units.
set(CPP_ENGINE_SOURCES
    src/core/graph_undirected.cpp
    src/core/graph_directed.cpp
    src/undirected_algos/exact_solver_u.cpp # also includes kernelization_u.cpp
    src/undirected_algos/memetic_u.cpp
    src/directed_algos/exact_solver_d.cpp # also includes kernelization_d.cpp
    src/directed_algos/memetic_d.cpp
    bindings/pybind_wrapper.cpp
)

# ── Build the pybind11 Python module ─────────────────────────────────────────
# pybind11_add_module creates a shared library with the correct suffix
# (.so on Linux/Mac, .pyd on Windows) and the correct Python ABI flags.
pybind11_add_module(cpp_engine ${CPP_ENGINE_SOURCES})

# ── Include paths ─────────────────────────────────────────────────────────────
target_include_directories(cpp_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ── Compiler warnings ─────────────────────────────────────────────────────────
if(MSVC)
    target_compile_options(cpp_engine PRIVATE /W3)
else()
    target_compile_options(cpp_engine PRIVATE
        -Wall -Wextra -Wno-unused-parameter
    )
endif()

# ── Installation ──────────────────────────────────────────────────────────────
# After `make`, the .so file ends up in build/.
# The benchmark scripts add build/ to sys.path automatically.
# Optionally install to a specific place:
# cmake -DCMAKE_INSTALL_PREFIX=/path/to/install ..
# make install
install(TARGETS cpp_engine DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/../experiments)

# ── Print summary ─────────────────────────────────────────────────────────────
message(STATUS "─────────────────────────────────────────────────────")
message(STATUS " Build type  : ${CMAKE_BUILD_TYPE}")
message(STATUS " C++ standard: C++${CMAKE_CXX_STANDARD}")
message(STATUS " Output dir  : ${CMAKE_CURRENT_BINARY_DIR}")
message(STATUS "─────────────────────────────────────────────────────")
message(STATUS " After build, copy cpp_engine.so to experiments/ OR")
message(STATUS " use: cmake --install . (installs to experiments/)")
message(STATUS "─────────────────────────────────────────────────────")