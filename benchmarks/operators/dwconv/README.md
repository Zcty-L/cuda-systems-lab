# DWConv 测速对比

执行链路：`dwconv_full_bench.py` → `dwconv_compare_bench` → `ops/dwconv`。
后端包装、输入与权重准备、CPU 参考和计时均在本目录；暂不保留 DWConv 测试，后续完成 Python 绑定后再添加接口测试。

## 构建与运行

仓库根目录执行，cuDNN、TensorRT 为可选依赖：

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCUDA_LAB_OPS=dwconv -DBUILD_TESTING=OFF
cmake --build build --target dwconv_compare_bench device_query --parallel
./build/tools/device_query
./build/benchmarks/operators/dwconv/dwconv_compare_bench --list-cases
./build/benchmarks/operators/dwconv/dwconv_compare_bench --case 7 --kernel opt --dtype fp16 --check-cpu
conda activate py311
python benchmarks/operators/dwconv/dwconv_full_bench.py --dtype all --include-opt
```

`DWCONV_ENABLE_CUDNN=OFF`、`DWCONV_ENABLE_TENSORRT=OFF` 可关闭对应后端；缺少开发库时自动跳过构建。用例定义在 [compare/cases.h](compare/cases.h)，共 23 个快速用例和 630 个规则网格组合，编号保持不变。单次入口的 `--quick` 仅选择前 23 组；`--help`、`--list-cases` 不初始化 CUDA。

| 选择项 | 含义与范围 |
| --- | --- |
| `baseline`、`baseline-k32/k64/k128`、`opt` | 自研 FP32/FP16；NCHW，FP16 默认 FP32 累加 |
| `prefetch`、`gmem`、`gmem-l2-entry` | FP32 输入预取、weights-L2 实验；NCHW |
| `pytorch`、`pytorch-native` | 抽离的 ATen native CUDA；前者支持 NHWC 输入转换，后者直接 launch，仅 NCHW；均不经过完整 Torch 框架 |
| `cudnn` | Graph + Autotune；`cudnn-autotune`、`cudnn-graph` 是同一入口的别名 |
| `cudnn-heuristic`、`cudnn-separate`、`cudnn-fused` | Graph 启发式、legacy 分离及融合；支持 NCHW/NHWC，FP32 允许 TF32 |
| `tensorrt`、`tensorrt-baseline`、`tensorrt-opt` | TensorRT 原生卷积及两种 Plugin；NCHW |

## 批量配置与结果

默认比较 `cudnn/baseline/pytorch`，`--include-opt` 加入 opt。`--cases` 选择原编号子集，`--kernels` 指定实现，`--layouts` 指定布局；例如：

```bash
python benchmarks/operators/dwconv/dwconv_full_bench.py --cases 7 5 --dtype all \
    --kernels cudnn cudnn-fused cudnn-separate --layouts nchw nhwc \
    --repeats 3 --output build/dwconv-cudnn-compare
python benchmarks/operators/dwconv/dwconv_full_bench.py --cases 7 5 4 401 8 --dtype all \
    --kernels cudnn-heuristic cudnn-autotune --reference cudnn-heuristic \
    --repeats 3 --output build/dwconv-autotune-compare
```

`--repeats` 为独立进程轮数，默认 1，每轮重新准备后端；`--samples` 为一次准备后的采样次数，默认 5。每次准备后默认预热 20 次，每个样本连续执行 100 次，CUDA Event 包围普通 launch，包含提交间隙；输出样本中位数及最小/最大值。数据生成、分配、布局转换、权重打包和库选优在正式计时外；`pytorch` 的 NHWC→NCHW 转换在每次调用内，计入耗时。`--check-cpu` 在计时前校验，默认关闭。

启动每项前查询空闲显存，输入输出估算占用加 2 GiB 不超过空闲容量时才运行；估算不含权重和库 workspace。`--max-io-gib` 可额外设置上限，策略保存在 `memory-policy.json`。

| 产物 | 内容 |
| --- | --- |
| `samples.csv` | 每个进程的状态、采样、准备耗时、algorithm/engine/workspace 等 |
| `summary.csv` | 按 case/精度/实现/布局聚合，全部重复成功才报告耗时中位数 |
| `comparison.csv`、`aggregates.csv` | 横向对比，以及容量范围内规则网格的统计 |
| `config.json`、`device.txt`、`cases.txt` | 配置、二进制校验值、设备及用例列表 |
| `case-<ID>-<dtype>-<kernel>-<layout>-r<repeat>.log` | 命令、原始输出、`DWCONV_RESULT` JSON 和退出状态 |

同配置可续跑：复用成功或明确不支持的结果，失败/OOM 重新尝试；二进制或测量配置改变后使用新的输出目录。默认目录为 `build/dwconv-compare-full-<dtype>/`。
`relative_to_reference = 当前耗时 / 参考耗时`，默认同布局 `cudnn`，可用 `--reference`、`--reference-layout` 调整；`relative_to_cudnn` 为同值兼容字段。失败不生成成功耗时，参考缺失不生成比值。`checked` 表示是否完成 CPU 校验；`workspace_bytes=null` 表示后端未报告，当前 TensorRT 即为此情况。

代码分工：[common.cpp](compare/common.cpp) 管理运行与计时，[data.h](compare/data.h) 管理输入及 CPU 参考，四个后端 `.cpp` 只负责兼容条件、准备与 launch；[runner.py](runner.py) 处理子进程协议，[reporting.py](reporting.py) 汇总 CSV。
绘图使用 [dwconv_plot.py](dwconv_plot.py) 的 `--csv`、`--dtype`、`--layout` 参数。测量规范见[公共说明](../../README.md)，历史报告与结果见[文档索引](../../../docs/README.md)。旧专项 `.cu`、测试程序和两个 cuDNN 专项脚本已退役；历史 Graph 数据与当前 Event 口径分别保留，未重新测量。
