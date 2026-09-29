# 存储微基准

Shared memory、L1、L2、DRAM 各有一个 bandwidth 和一个 latency 目标，目标名与 `.cu` 文件名一致。

构建聚合目标：`cmake --build build --target memory_benchmarks --parallel`。

运行前先执行 `./build/tools/device_query`；批量验证使用 `ctest --test-dir build -L memory --output-on-failure`，单项可执行文件位于 `build/benchmarks/memory/`。

参数、计时口径、复现命令和结果边界见[导入报告](../../docs/benchmarks/memory/baseline.md)，公共规范见[性能测量](../README.md)。
