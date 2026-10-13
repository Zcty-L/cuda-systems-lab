#include "dwconv.cuh"
#include <cuda_lab/ptx_utils.cuh>

__global__ void conv2d_4x128x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                               Conv2DParam param)
{
    __shared__ __align__(2 * 1024) char smem[4 * 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(weights + blockIdx.y * 4 * param.KhKw +
                                                                threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + sizeof(float), threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    for(int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

            if(guard)
            {
                input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
                input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3],
                    weights_lds_addr + 0 * sizeof(float));
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                    weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                    weights_lds_addr + 256 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                    weights_lds_addr + 384 * sizeof(float));

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void conv2d_4x64x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                              Conv2DParam param)
{
    __shared__ __align__(1024) char smem[4 * 64 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(weights + blockIdx.y * 4 * param.KhKw +
                                                                threadIdx.x / 64 * param.KhKw + threadIdx.x % 64);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 64 < param.KhKw);

    ptx::sts32(weight_ldg_reg, weights_sts_addr);
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    for(int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

            if(guard)
            {
                input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
                input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                    weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                    weights_lds_addr + 128 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                    weights_lds_addr + 192 * sizeof(float));

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

__global__ void conv2d_4x32x256_groups_kernel(float *inputs, float *weights, float *bias, float *outputs,
                                              Conv2DParam param)
{
    __shared__ __align__(128) char smem[4 * 32 * 4 + 4 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(weights + blockIdx.y * 4 * param.KhKw +
                                                                threadIdx.x / 32 * param.KhKw + threadIdx.x % 32);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg;
    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    if(threadIdx.x < 128)
    {
        ptx::ldg_nc_0(weight_ldg_reg, weight_ldg_ptr, threadIdx.x % 32 < param.KhKw);
        ptx::sts32(weight_ldg_reg, weights_sts_addr);
    }
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    for(int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

            if(guard)
            {
                input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
                input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

        ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3], weights_lds_addr);
        ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                    weights_lds_addr + 32 * sizeof(float));
        ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                    weights_lds_addr + 64 * sizeof(float));
        ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                    weights_lds_addr + 96 * sizeof(float));

        weights_lds_addr += 4 * sizeof(float);

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

namespace
{
__device__ __forceinline__ void load_input(float *input_ptr, float (&input_frag)[4][4], const Conv2DParam &param,
                                           int posh_ori, int posw_ori, int k)
{
#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        int tap = k + i;
        int curH = posh_ori + tap / param.Kw;
        int curW = posw_ori + tap % param.Kw;
        int in_offset = curH * param.in_w + curW;
        bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

        if(guard)
        {
            input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
            input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
            input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
            input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
        }
        else
        {
            input_frag[0][i] = 0.0f;
            input_frag[1][i] = 0.0f;
            input_frag[2][i] = 0.0f;
            input_frag[3][i] = 0.0f;
        }
    }
}
__device__ __forceinline__ void accumulate(float (&output_frag)[4], const float (&input_frag)[4][4],
                                           uint32_t weights_lds_addr, int k)
{
    float weight_frag[16];
    ptx::lds128(weight_frag[0], weight_frag[1], weight_frag[2], weight_frag[3],
                weights_lds_addr + k * sizeof(float) + 0 * sizeof(float));
    ptx::lds128(weight_frag[4], weight_frag[5], weight_frag[6], weight_frag[7],
                weights_lds_addr + k * sizeof(float) + 128 * sizeof(float));
    ptx::lds128(weight_frag[8], weight_frag[9], weight_frag[10], weight_frag[11],
                weights_lds_addr + k * sizeof(float) + 256 * sizeof(float));
    ptx::lds128(weight_frag[12], weight_frag[13], weight_frag[14], weight_frag[15],
                weights_lds_addr + k * sizeof(float) + 384 * sizeof(float));

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
        output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
        output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
        output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
    }
}

} // namespace

// 权重一次加载；交替预取下一输入片段，保持 baseline 的加载与边界语义。
__global__ void conv2d_4x128x256_groups_kernel_prefetch(float *inputs, float *weights, float *bias, float *outputs,
                                                        Conv2DParam param)
{
    __shared__ __align__(2 * 1024) char smem[4 * 128 * 4];
    auto *smemweight = reinterpret_cast<float *>(smem);

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    uint32_t weights_sts_addr = ptx::smem_u32addr(smemweight + threadIdx.x * 2);
    uint32_t weights_lds_addr = ptx::smem_u32addr(smemweight);

    const char *weight_ldg_ptr = reinterpret_cast<const char *>(weights + blockIdx.y * 4 * param.KhKw +
                                                                threadIdx.x / 64 * param.KhKw + threadIdx.x % 64 * 2);
    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_ldg_reg[2];
    float input_a[4][4], input_b[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    ptx::ldg_nc_0(weight_ldg_reg[0], weight_ldg_ptr, threadIdx.x % 64 * 2 < param.KhKw);
    ptx::ldg_nc_0(weight_ldg_reg[1], weight_ldg_ptr + sizeof(float), threadIdx.x % 64 * 2 + 1 < param.KhKw);

    ptx::sts64(weight_ldg_reg[0], weight_ldg_reg[1], weights_sts_addr);
    // 权重合作写入后同步；后续 shared 只读，循环内无需 block 同步。
    __syncthreads();

    load_input(input_ptr, input_a, param, posh_ori, posw_ori, 0);
    for(int k = 0; k < param.KhKw; k += 8)
    {
        if(k + 4 < param.KhKw)
        {
            load_input(input_ptr, input_b, param, posh_ori, posw_ori, k + 4);
        }

        accumulate(output_frag, input_a, weights_lds_addr, k);

        if(k + 4 < param.KhKw)
        {
            if(k + 8 < param.KhKw)
            {
                load_input(input_ptr, input_a, param, posh_ori, posw_ori, k + 8);
            }

            accumulate(output_frag, input_b, weights_lds_addr, k + 4);
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

// 权重存储对照：保持 k128 baseline 输入地址、边界与累加顺序。

namespace
{
__device__ __forceinline__ void weights_gmem(float *inputs, float *weights, float *bias, float *outputs,
                                             Conv2DParam param)
{

    int out_pos = blockIdx.x * 256 + threadIdx.x;
    int posh_ori = (out_pos / param.out_w) * param.Sh - param.Ph;
    int posw_ori = (out_pos % param.out_w) * param.Sw - param.Pw;

    auto *input_ptr = inputs + blockIdx.z * param.inBatchNumel + blockIdx.y * 4 * param.inHW;

    float weight_frag[16];
    float input_frag[4][4];
    float output_frag[4];

#pragma unroll
    for(int i = 0; i < 4; ++i)
    {
        output_frag[i] = bias[blockIdx.y * 4 + i];
    }

    for(int k = 0; k < param.KhKw; k += 4)
    {
#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            int tap = k + i;
            int curH = posh_ori + tap / param.Kw;
            int curW = posw_ori + tap % param.Kw;
            int in_offset = curH * param.in_w + curW;
            bool guard = curH >= 0 && curW >= 0 && curW < param.in_w && curH < param.in_h;

            if(guard)
            {
                input_frag[0][i] = input_ptr[in_offset + param.inHW * 0];
                input_frag[1][i] = input_ptr[in_offset + param.inHW * 1];
                input_frag[2][i] = input_ptr[in_offset + param.inHW * 2];
                input_frag[3][i] = input_ptr[in_offset + param.inHW * 3];
            }
            else
            {
                input_frag[0][i] = 0.0f;
                input_frag[1][i] = 0.0f;
                input_frag[2][i] = 0.0f;
                input_frag[3][i] = 0.0f;
            }
        }

#pragma unroll
        for(int ch = 0; ch < 4; ++ch)
        {
#pragma unroll
            for(int i = 0; i < 4; ++i)
                weight_frag[ch * 4 + i] =
                    k + i < param.KhKw ? __ldg(weights + (blockIdx.y * 4 + ch) * param.KhKw + k + i) : 0.0f;
        }

#pragma unroll
        for(int i = 0; i < 4; ++i)
        {
            output_frag[0] += weight_frag[i + 0] * input_frag[0][i];
            output_frag[1] += weight_frag[i + 4] * input_frag[1][i];
            output_frag[2] += weight_frag[i + 8] * input_frag[2][i];
            output_frag[3] += weight_frag[i + 12] * input_frag[3][i];
        }
    }

    int outOffset = blockIdx.z * param.outBatchNumel + blockIdx.y * 4 * param.outHW + out_pos;
    if(out_pos < param.outHW)
    {
        outputs[outOffset + param.outHW * 0] = output_frag[0];
        outputs[outOffset + param.outHW * 1] = output_frag[1];
        outputs[outOffset + param.outHW * 2] = output_frag[2];
        outputs[outOffset + param.outHW * 3] = output_frag[3];
    }
}

} // namespace

void launch_baseline_fp32(float *inputs, float *weights, float *bias, float *outputs, const Conv2DParam &param, int n,
                          cudaStream_t stream)
{
    dim3 grid((param.outHW + 255) / 256, param.out_ch / 4, n);
    if(param.KhKw <= 32)
    {
        conv2d_4x32x256_groups_kernel<<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
    else if(param.KhKw <= 64)
    {
        conv2d_4x64x256_groups_kernel<<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
    else
    {
        conv2d_4x128x256_groups_kernel<<<grid, 256, 0, stream>>>(inputs, weights, bias, outputs, param);
    }
}

// 从 baseline 替换权重读取，输入和累加逻辑不变。
__global__ void conv2d_4x128x256_groups_kernel_gmem(float *inputs, float *weights, float *bias, float *outputs,
                                                    Conv2DParam param)
{
    weights_gmem(inputs, weights, bias, outputs, param);
}

// 启动方须为 weights 设置 L2 accessPolicyWindow；计算路径与 gmem 相同。
__global__ void conv2d_4x128x256_groups_kernel_gmem_l2(float *inputs, float *weights, float *bias, float *outputs,
                                                       Conv2DParam param)
{
    weights_gmem(inputs, weights, bias, outputs, param);
}
