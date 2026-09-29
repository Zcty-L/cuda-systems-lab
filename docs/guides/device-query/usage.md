# 设备查询工具

`tools/device_query.cu` 参考 CUDAOP 的 `op/device_query.cu`，用于采集实验前的设备配置，通过公共 `cuda_lab::check_cuda` 处理必要查询的错误。

## 构建与运行

在仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target device_query --parallel
./build/tools/device_query
```

程序枚举所有 CUDA 可见设备，遵循 `CUDA_VISIBLE_DEVICES`；输出的设备编号是当前进程的可见编号。例如只查询第一张 GPU 并保存日志：

```bash
CUDA_VISIBLE_DEVICES=0 ./build/tools/device_query > build/device-query.log
```

## 输出口径

- 配置：可见设备数量、驱动支持的 CUDA API 版本和运行时版本；驱动 API 版本不是 NVIDIA 驱动软件包版本。
- 设备：名称、计算能力、SM 架构、SM 数量和 warp 大小。
- 存储：全局显存、L2、共享内存、常量内存和寄存器容量；二进制容量使用 KiB、MiB、GiB，并附字节数。
- 时钟与带宽：通过设备属性 API 查询时钟，避免依赖已移除的结构体时钟字段。带宽按 `2 × 显存时钟(kHz) × 1000 × 总线位宽 / 8 / 10^9` 估算，单位 GB/s，采用 DDR 假设，不代表实测可持续带宽。时钟查询失败时显示 `N/A` 及错误原因，并继续其他查询。
- 驻留：线程、block、warp 和网格限制，以及常用 block 大小对应的驻留上限表。表格只考虑线程、warp 和 block 数限制，不考虑 kernel 的寄存器与共享内存占用，不能替代具体 kernel 的 occupancy 分析。
- 特性：直接展示运行时报告的布尔属性；ECC 表示是否启用。

本工具不根据计算能力粗略推断核心数量、Tensor Core/WGMMA 支持或 FP16/INT8 峰值；这些需要针对具体架构和指令单独核实。Roofline 分析口径见[公共方法](../../benchmarks/methodology/roofline.md)。

正常完成输出 `[SUCCESS]` 并返回 0。枚举成功但设备数为零时输出 `[SKIP]` 并返回 0；必要 CUDA 查询失败时输出 `[ERROR]` 并返回 1。
