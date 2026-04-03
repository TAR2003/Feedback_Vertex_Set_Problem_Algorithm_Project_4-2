#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "undirected_fvs.h"
#include "directed_fvs.h"

namespace py = pybind11;

PYBIND11_MODULE(cpp_engine, m) {
    m.doc() = "FVS C++ bindings - Feedback Vertex Set solvers";
    m.attr("__version__") = "1.0.0";

    // ────────────────────────────────────────────────────────────────────────
    // Undirected solvers
    // ────────────────────────────────────────────────────────────────────────

    m.def("solve_undirected_BST",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &>(
              &solve_undirected_BST),
          py::arg("n"), py::arg("edges"),
          "Bounded Search Tree exact solver for undirected FVS");

    m.def("solve_undirected_IC",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &>(
              &solve_undirected_IC),
          py::arg("n"), py::arg("edges"),
          "Iterative Compression exact solver for undirected FVS");

    m.def("solve_undirected_MA",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_undirected_MA),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 20,
          py::arg("max_time_seconds") = 600,
          "Memetic Algorithm solver for undirected FVS");

    m.def("solve_undirected_KMA",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_undirected_KMA),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 15,
          py::arg("max_time_seconds") = 600,
          "Kernelized Memetic Algorithm solver for undirected FVS");

    m.def("solve_undirected_KME",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_undirected_KME),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 15,
          py::arg("max_time_seconds") = 600,
          "Legacy alias for solve_undirected_KMA");

    // ────────────────────────────────────────────────────────────────────────
    // Directed solvers
    // ────────────────────────────────────────────────────────────────────────

    m.def("solve_directed_BST",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &>(
              &solve_directed_BST),
          py::arg("n"), py::arg("edges"),
          "Bounded Search Tree exact solver for directed FVS");

    m.def("solve_directed_IC",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &>(
              &solve_directed_IC),
          py::arg("n"), py::arg("edges"),
          "Iterative Compression exact solver for directed FVS");

    m.def("solve_directed_MA",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_directed_MA),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 20,
          py::arg("max_time_seconds") = 600,
          "Memetic Algorithm solver for directed FVS");

    m.def("solve_directed_KMA",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_directed_KMA),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 15,
          py::arg("max_time_seconds") = 600,
          "Kernelized Memetic Algorithm solver for directed FVS");

    m.def("solve_directed_KME",
          py::overload_cast<int, const std::vector<std::pair<int, int>> &, int, int, int, int>(
              &solve_directed_KME),
          py::arg("n"), py::arg("edges"), py::arg("pop_size") = 20,
          py::arg("max_gens") = 100, py::arg("patience") = 15,
          py::arg("max_time_seconds") = 600,
          "Legacy alias for solve_directed_KMA");
}
