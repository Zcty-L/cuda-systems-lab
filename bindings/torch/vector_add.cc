#include <ATen/ATen.h>
#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/library.h>

#include <vector_add.cuh>

namespace
{

void check_inputs(const at::Tensor &a, const at::Tensor &b)
{
    TORCH_CHECK(a.is_cuda() && b.is_cuda(), "vector_add 要求 CUDA 输入");
    TORCH_CHECK(a.device() == b.device(), "vector_add 输入必须位于同一设备");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat, "vector_add 仅支持 FP32");
    TORCH_CHECK(a.sizes() == b.sizes(), "vector_add 输入 shape 必须相同，不执行广播");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(), "vector_add 输入必须连续");
}

void launch(const at::Tensor &a, const at::Tensor &b, at::Tensor &out)
{
    const c10::cuda::CUDAGuard device_guard(a.device());
    const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    cuda_lab::vector_add(a.const_data_ptr<float>(), b.const_data_ptr<float>(), out.mutable_data_ptr<float>(),
                         static_cast<std::size_t>(a.numel()), stream.stream());
}

at::Tensor vector_add(const at::Tensor &a, const at::Tensor &b)
{
    check_inputs(a, b);
    const c10::cuda::CUDAGuard device_guard(a.device());
    auto out = at::empty(a.sizes(), a.options());
    launch(a, b, out);
    return out;
}

void vector_add_out(const at::Tensor &a, const at::Tensor &b, at::Tensor out)
{
    check_inputs(a, b);
    TORCH_CHECK(out.device() == a.device() && out.scalar_type() == at::kFloat,
                "vector_add_out 输出的设备和 dtype 必须与输入相同");
    TORCH_CHECK(out.sizes() == a.sizes() && out.is_contiguous(), "vector_add_out 输出 shape 必须匹配且连续");
    TORCH_CHECK(!a.requires_grad() && !b.requires_grad() && !out.requires_grad(),
                "vector_add_out 不支持 autograd, 请使用 vector_add");

    at::assert_no_internal_overlap(out);
    at::assert_no_overlap(out, a);
    at::assert_no_overlap(out, b);
    launch(a, b, out);
}

} // namespace

TORCH_LIBRARY_IMPL(cuda_lab, CUDA, module)
{
    module.impl("vector_add", &vector_add);
    module.impl("vector_add_out", &vector_add_out);
}
