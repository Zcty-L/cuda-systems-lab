# 文档索引

正式文档统一按 `docs/<大类>/<主题>/<文档>.md` 组织，先按用途分类，再按具体主题归档。`docs/` 根目录只保留本索引，不直接放报告或指南。

## 分类与目录

| 大类 | 目录 | 主题示例 | 内容 |
| --- | --- | --- | --- |
| 使用指南 | `guides/` | `build/`、`profiling/` | 构建、运行、调试与分析工具的使用方法 |
| 架构学习 | `architecture/` | `ptx/`、`memory/` | 指令语义、硬件机制与学习实验结论 |
| 基准测试 | `benchmarks/` | `methodology/`、`memory/`、`instructions/` | 公共测量方法与独立微基准报告 |
| 算子优化 | `operators/` | `softmax/`、`vector_add/` | 按算子归档的实现说明、正确性验证、性能基线与优化报告 |

例如，Softmax 的文档组织为：

```text
docs/
├── README.md
└── operators/
    └── softmax/
        ├── arithmetic-intensity.md
        ├── baseline.md
        └── block-online.md
```

## 归档规则

- 新增文档先选择大类，再创建或复用主题目录；目录按实际内容创建，不预建空目录。
- 目录与文件名使用小写英文，多个单词用连字符连接；算子主题目录与 `ops/{name}/` 保持一致。主题目录内的文件名不再重复算子名称。
- 按文档主要用途归档。同一算子的基线和优化测量集中放在 `operators/{name}/`；独立存储或指令微基准报告放在 `benchmarks/`。涉及多个领域时通过链接关联，避免重复存放。
- 按适用范围拆分：公共配置集中在 `benchmarks/README.md`，通用指标与分析方法放 `benchmarks/methodology/`；算子文档只保留特有口径、配置和结果，通过链接引用公共方法。规则只维护一处，`AGENTS.md` 和 README 以简短引用或入口表导航。
- 文档使用简练表述，避免重复定义、背景和结论。
- 默认保持“大类 / 主题 / 文档”三级结构，不为单篇文档继续增加目录层级。主题内容较多时可增加局部 `README.md` 作为导航。
- 新增、移动或重命名文档时，同步更新本索引及根 README、模块 README 中的相关链接，仓库内链接使用相对路径。
- 根 README、模块 README 和各级文档 README 用于导航与局部说明；详细指南、实验报告按上述结构归档。
- 实验报告保留设备与软件环境、输入配置、计时范围、关键结果和复现命令；优化报告同时记录正确性验证以及相同条件下的性能基线与比较结果。

- CSV、TSV、profiler 输出与日志等中间结果统一保存在被 Git 忽略的 `build/` 下；文档仅保留关键数据、结论和复现命令，不链接本地中间文件。

## 文档索引

### 使用指南

- 公共工具：[设备查询](guides/device-query/usage.md)

### 基准测试
- 存储微基准：[导入与验证](benchmarks/memory/baseline.md)

- 公共方法：[Roofline 指标与判断](benchmarks/methodology/roofline.md)

### 算子优化

- Softmax：[导入基线](operators/softmax/baseline.md)、[block 与 online 实现](operators/softmax/block-online.md)、[AI 计算说明](operators/softmax/arithmetic-intensity.md)、[Roofline 分析](operators/softmax/roofline.md)、[行数与并行度](operators/softmax/row-scaling.md)、[Cluster 对照](operators/softmax/cluster.md)
