#include <pybind11/pybind11.h>
#include "undirected_fvs.h"
#include "directed_fvs.h"

namespace py = pybind11;

PYBIND11_MODULE(fvs_cpp, m) {
    m.doc() = "FVS C++ bindings";
    // TODO: add wrappers for DirectedFVS and UndirectedFVS
}
