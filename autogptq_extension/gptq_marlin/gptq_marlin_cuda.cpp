#include "gptq_marlin_cuda.h"
#include <torch/extension.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gptq_marlin_gemm", &gptq_marlin_gemm, "gptq_marlin Optimized Quantized GEMM for GPTQ");
  m.def("gptq_marlin_repack", &gptq_marlin_repack, "gptq_marlin repack from GPTQ");
}