# 存储微基准导入与验证

## 来源与范围

从 CUDAOP 提交 `d901ad1` 的 `op/test/` 导入 shared memory、L1、L2、DRAM 的 4 个 bandwidth 与 4 个 latency 程序，放在 `benchmarks/memory/`。原仓库文件保留。此次保持原测量逻辑与参数，仅调整头文件路径、格式和构建集成；依赖的 9 个 PTX wrapper 集中在 `include/cuda_lab/ptx_utils.cuh`。

## 构建与复现

在仓库根目录执行：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --target device_query memory_benchmarks --parallel
./build/tools/device_query > build/memory-device-query.log
ctest --test-dir build -L memory -V > build/memory-tests.log
```

单独运行示例：`./build/benchmarks/memory/l1cache_latency`。8 个 CTest 用例使用 `memory`、`benchmark` 标签，设置串行运行以避免相互干扰，单项超时 120 秒。正常完成输出 `[SUCCESS]`。

## 配置与计时范围

下表记录导入版本对[公共配置](../../../benchmarks/README.md)的覆盖。此轮用于建立导入基线，尚未执行 occupancy/waves 扫描和吞吐平台确认。

| 程序 | 配置 | 预热与计时 |
| --- | --- | --- |
| smem_bandwidth | 1 block × 256 threads；12 KiB shared；每线程 512 次 16 B store | 预热 100 次；测 1 次 kernel 内 store 区间，取各 warp 起止周期的包络 |
| smem_latency | 1 × 16；64 B 自引用地址链 | 预热 100 次；测 1 次 kernel 内 50 次依赖 load |
| l1cache_bandwidth | SM 数 × 64 blocks，每 block 128 threads；32 KiB 工作集；4 路 16 B load，循环 64 次 | 预热 20 次；CUDA event 覆盖 100 次 kernel |
| l1cache_latency | 1 × 4；32 B 自引用地址链，使用 nc load | 预热 100 次；kernel 内额外预热 50 次 load，计时 50 次依赖 load |
| l2cache_bandwidth | 262144 blocks × 128 threads；2 MiB 工作集；每线程 16 次 4 B cg load | 预热 200 次；CUDA event 覆盖 200 次 kernel |
| l2cache_latency | 1 × 32；1408 B 地址链，128 B 步长，cg load | 预热 100 次；计时 10 次依赖 load |
| dram_bandwidth | 128 threads/block；规模从 4 MiB 翻倍到 1 GiB；每线程 16 B；读/写 grid = size/2048，copy 减半 | 每种 kernel 预热 1 次；各用 event 计时 100 次，地址偏移步长 16 MiB；copy 字节数包含读与写 |
| dram_latency | 1 × 32；11264 B 地址链，1024 B 步长，cg load | 预热 1 次，用 max(128 MiB, 2×L2) 缓冲区冲刷缓存后，计时 10 次依赖 load |

CUDA event 区间不包含分配与初始化；周期计时不包含主机传输。延迟按计时区间周期数除以 load 次数计算，未扣除计时和地址运算开销。

## 验证结果

本次运行设备为 NVIDIA GeForce RTX 5070 Ti Laptop GPU，SM 120、46 个 SM、L2 36 MiB、显存约 11.94 GiB；CUDA Runtime 与驱动 API 版本均为 13.2，Release 构建。未锁定 GPU 时钟。8/8 个测试通过，总耗时约 4.78 秒。

| 测量 | 本次结果 |
| --- | --- |
| Shared store 实测 | 127.82 B/cycle，单 block 所在 SM |
| L1 聚合读取 | 14543.01 GB/s |
| L2 聚合读取 | 2335.20 GB/s |
| 1 GiB DRAM 读取 / 写入 / 拷贝 | 345.75 / 398.01 / 533.88 GB/s |
| Shared / L1 / L2 / DRAM 平均访问延迟 | 27.72 / 38.00 / 306.70 / 840.20 cycles/load |

## 结果边界

- 这是单轮导入运行结果，未与原仓库程序做同条件性能对照，也未统计多轮波动，不作为优化收益或硬件峰值结论。
- Shared 带宽原程序存在跨线程重叠写入，其检查仅确认读回值非负；输出的 128 B/cycle 是向上取整推断，9491.46 GB/s 是结合报告时钟和 SM 数的外推，不是全芯片实测。
- L1/L2 带宽按程序请求字节数计算，缓存命中率与实际流量尚未用 profiler 核对。DRAM 冲刷缓冲区不构成全部目标缓存行已被驱逐的证明。
- 原测试的校验强度不同：延迟测试检查地址链结果与正周期数；L1/L2 带宽检查 sink 与有效计时；DRAM 带宽只检查吞吐值有限且为正，没有逐元素读写正确性校验。
- DRAM 带宽最大工作区约 2.55 GiB；本轮沿用全零输入。运行通过只表明当前设备上完成了既有检查。
