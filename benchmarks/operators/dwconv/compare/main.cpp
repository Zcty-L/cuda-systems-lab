#include "common.h"

int main(int argc, char **argv)
{
    std::vector<dwconv_bench::Backend> backends{dwconv_bench::baseline_opt_backend(), dwconv_bench::torch_backend()};

#ifdef DWCONV_BENCH_HAS_CUDNN
    backends.push_back(dwconv_bench::cudnn_backend());
#endif

#ifdef DWCONV_BENCH_HAS_TRT
    backends.push_back(dwconv_bench::trt_backend());
#endif

    return dwconv_bench::run_benchmarks(argc, argv, backends);
}
