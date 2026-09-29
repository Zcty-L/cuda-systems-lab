#pragma once
#include <cuda_runtime.h>
#include <cstdint>
namespace ptx
{
// 封装 smem_u32addr，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
// cvta.to.shared：将通用地址转换为共享内存地址。
__device__ __forceinline__ uint32_t smem_u32addr(const void *ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}
// 封装 read_clock64，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
__device__ __forceinline__ uint64_t read_clock64()
{
    uint64_t value;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) : : "memory");
    return value;
}
// 封装 ldg32_cg，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T> __device__ __forceinline__ void ldg32_cg(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 4, "ldg32_cg requires 4-byte type");
    asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(*reinterpret_cast<unsigned *>(&reg)) : "l"(ptr) : "memory");
}
// 封装 ldg128_cs，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T> __device__ __forceinline__ void ldg128_cs(T &reg0, T &reg1, T &reg2, T &reg3, const void *ptr)
{
    static_assert(sizeof(T) == 4, "ldg128_cs registers must be 4-byte types");
    asm volatile("ld.global.cs.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(*reinterpret_cast<unsigned *>(&reg0)), "=r"(*reinterpret_cast<unsigned *>(&reg1)),
                   "=r"(*reinterpret_cast<unsigned *>(&reg2)), "=r"(*reinterpret_cast<unsigned *>(&reg3))
                 : "l"(ptr)
                 : "memory");
}
// 封装 ldg64_nc，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T> __device__ __forceinline__ void ldg64_nc(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 8, "ldg64_nc requires 8-byte type");
    asm volatile("ld.global.nc.b64 %0, [%1];"
                 : "=l"(*reinterpret_cast<unsigned long long *>(&reg))
                 : "l"(ptr)
                 : "memory");
}
// 封装 ldg128_nc，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T> __device__ __forceinline__ void ldg128_nc(T &reg, const void *ptr)
{
    static_assert(sizeof(T) == 16, "ldg128_nc requires 16-byte type");
    unsigned *values = reinterpret_cast<unsigned *>(&reg);
    asm volatile("ld.global.nc.v4.b32 {%0, %1, %2, %3}, [%4];"
                 : "=r"(values[0]), "=r"(values[1]), "=r"(values[2]), "=r"(values[3])
                 : "l"(ptr)
                 : "memory");
}
// 封装 lds32，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T> __device__ __forceinline__ void lds32(T &reg0, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "lds32 requires 4-byte type");
    asm volatile("ld.shared.b32 %0, [%1];" : "=r"(*reinterpret_cast<unsigned *>(&reg0)) : "r"(addr));
}
// 封装 sts128，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T>
__device__ __forceinline__ void sts128(const T &reg0, const T &reg1, const T &reg2, const T &reg3, const uint32_t &addr)
{
    static_assert(sizeof(T) == 4, "sts128 registers must be 4-byte types");
    asm volatile("st.shared.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "r"(addr), "r"(*reinterpret_cast<const unsigned *>(&reg0)),
                   "r"(*reinterpret_cast<const unsigned *>(&reg1)), "r"(*reinterpret_cast<const unsigned *>(&reg2)),
                   "r"(*reinterpret_cast<const unsigned *>(&reg3)));
}
// 封装 stg128_cs，用于存储微基准的地址转换、访存或周期计时。
// 指令名称见函数内 PTX；来源：https://docs.nvidia.com/cuda/parallel-thread-execution/
template <typename T>
__device__ __forceinline__ void stg128_cs(const T &reg0, const T &reg1, const T &reg2, const T &reg3, void *ptr)
{
    static_assert(sizeof(T) == 4, "stg128_cs registers must be 4-byte types");
    asm volatile("st.global.cs.v4.b32 [%4], {%0, %1, %2, %3};"
                 :
                 : "r"(*reinterpret_cast<const unsigned *>(&reg0)), "r"(*reinterpret_cast<const unsigned *>(&reg1)),
                   "r"(*reinterpret_cast<const unsigned *>(&reg2)), "r"(*reinterpret_cast<const unsigned *>(&reg3)),
                   "l"(ptr)
                 : "memory");
}
} // namespace ptx
