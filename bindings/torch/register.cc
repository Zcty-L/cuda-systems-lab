#include <torch/library.h>

TORCH_LIBRARY(cuda_lab, module)
{
    module.def("vector_add(Tensor a, Tensor b) -> Tensor");
    module.def("vector_add_out(Tensor a, Tensor b, Tensor(o!) out) -> ()");
}
