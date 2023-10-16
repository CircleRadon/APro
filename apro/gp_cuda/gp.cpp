#pragma once
#include <torch/extension.h>
#include "mst/mst.hpp"
#include "gp_process/gp_process.hpp"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mst_forward", &mst_forward, "mst forward");
    m.def("gp_forward", &gp_forward, "gp forward");
}

