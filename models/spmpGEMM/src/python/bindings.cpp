#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(spmp_gemm, m) {
    m.doc() = "SPMP-GEMM Python Bindings"; 

    // Bind the Tile
    py::class_<spmpGEMM::Tile<int32_t>>(m, "Tile")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def_readwrite("rows", &spmpGEMM::Tile<int32_t>::rows)
        .def_readwrite("cols", &spmpGEMM::Tile<int32_t>::cols)
        .def_readwrite("data", &spmpGEMM::Tile<int32_t>::data)
        .def("at", static_cast<int32_t& (spmpGEMM::Tile<int32_t>::*)(size_t, size_t)>(&spmpGEMM::Tile<int32_t>::at));

    // Bind the SIMDEngine 
    py::class_<spmpGEMM::SIMDEngine>(m, "SIMDEngine")
        .def(py::init<const std::string&>())
        .def("compute", &spmpGEMM::SIMDEngine::compute,
             py::arg("activations"),
             py::arg("activation_threshold") = 0);
}