#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "../include/simd_engine.hpp"
#include "../include/stats.hpp"

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

    // Bind the PEStats
    py::class_<spmpGEMM::PEStats>(m, "PEStats")
        .def_readonly("total_cycles", &spmpGEMM::PEStats::total_cycles)
        .def_readonly("masking_operations", &spmpGEMM::PEStats::masking_operations)
        .def_readonly("shifting_operations", &spmpGEMM::PEStats::shifting_operations)
        .def_readonly("addition_operations", &spmpGEMM::PEStats::addition_operations)
        .def_readonly("total_mask_ops", &spmpGEMM::PEStats::total_mask_ops)
        .def_readonly("total_shifts", &spmpGEMM::PEStats::total_shifts)
        .def_readonly("total_additions", &spmpGEMM::PEStats::total_additions);

    // Bind the SystemStats
    py::class_<spmpGEMM::SystemStats>(m, "SystemStats")
        .def_readonly("pe_stats", &spmpGEMM::SystemStats::pe_stats);

    // Bind the SIMDEngine 
    py::class_<spmpGEMM::SIMDEngine>(m, "SIMDEngine")
        .def(py::init<const std::string&>())
        .def("compute", &spmpGEMM::SIMDEngine::compute,
             py::arg("activations"),
             py::arg("activation_threshold") = 0)
        .def("get_stats", &spmpGEMM::SIMDEngine::getStats);
}