#include <torch/extension.h>

torch::Tensor compute_trilinear_weights(
    const torch::Tensor& positions, const torch::Tensor& p0, const torch::Tensor& p1,
    int grid_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("compute_trilinear_weights", &compute_trilinear_weights);
}