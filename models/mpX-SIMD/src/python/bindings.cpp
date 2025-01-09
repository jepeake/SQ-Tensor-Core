#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"

namespace py = pybind11;

PYBIND11_MODULE(mpx_simd, m) {
    m.doc() = "MPX-SIMD Python Bindings"; 

    // Bind the Tile
    py::class_<mpX::Tile<int32_t>>(m, "Tile")
        .def(py::init<>())
        .def(py::init<size_t, size_t>())
        .def_readwrite("rows", &mpX::Tile<int32_t>::rows)
        .def_readwrite("cols", &mpX::Tile<int32_t>::cols)
        .def_readwrite("data", &mpX::Tile<int32_t>::data)
        .def("at", static_cast<int32_t& (mpX::Tile<int32_t>::*)(size_t, size_t)>(&mpX::Tile<int32_t>::at));

    // Bind the SIMDEngine 
    py::class_<mpX::SIMDEngine>(m, "SIMDEngine")
        .def(py::init<const std::string&>())
        .def("compute", &mpX::SIMDEngine::compute);
}