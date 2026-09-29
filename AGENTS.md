# CUDA Systems Lab 协作规范

## 语言与目录

- 所有回答与项目文档使用简体中文。
- 算子实现放 `ops/{name}/`；性能测量放 `benchmarks/`；指令和架构学习实验放 `studies/`。
- 所有正式报告与指南保存在 `docs/`；根 README、本文件和模块 README 用于导航与局部说明。
- 正式文档按 `docs/<大类>/<主题>/<文档>.md` 分层归档，分类与命名遵循 [docs/README.md](docs/README.md)；新增或移动文档时同步更新索引及引用链接。
- 公共头文件放 `include/cuda_lab/`。

## PTX 与架构实验

- 新增可复用的生产代码 PTX wrapper 时，统一放在 `include/cuda_lab/ptx_utils.cuh`。
- 新增 wrapper 时在注释中写明指令名称、来源链接和用途。
- `studies/` 中可使用本地 raw inline PTX 来验证指令语义；不要把本地 wrapper 当作公共接口。
- SM 专用目标应在 CMake 中声明架构要求，测试输出也应标出设备和 SM。

## 代码与构建

- C++/CUDA 输出统一使用 `std::cout`，禁止 `printf`；Python 输出使用 `logging`。
- C++/CUDA 代码遵循根目录 `.clang-format`；修改后对涉及的文件执行 `clang-format -i <文件>`。
- C++/CUDA 目标统一经 CMake 构建；仅临时验证允许直接运行 `nvcc`，最终目标仍须注册到 CMake。
- Python 脚本可独立执行。需要 Python 解释器时默认使用 `conda activate py311`；缺少依赖包时先询问是否安装。
- 可选依赖按目标启用，避免一个实验的依赖阻止整个仓库配置。

## 验证与测量

- 新算子至少包含正确性测试；基准测试记录配置、设备、计时范围、结果和运行命令。
- 开始 GPU 性能测量或架构相关实验前，运行 `./build/tools/device_query` 确认设备与计算能力；构建方法和输出口径见设备查询指南 (docs/guides/device-query/usage.md)。
- 基准配置遵循 [benchmarks/README.md](benchmarks/README.md)，公共方法与算子报告的划分遵循 [文档规则](docs/README.md)。
- 测试输出包含配置、主要阶段、关键结果、`[SUCCESS]` 标记；不同测试与主要阶段之间留空行。
- 优化前记录正确性和性能基线；同一设备、输入和计时方法下比较结果。算子基线与优化报告统一保存在 `docs/operators/{name}/`。

## 分支与提交

- `main` 是唯一长期集成分支；每项独立改动从最新 `main` 建短期分支，经 PR 合并后清理。
- 分支使用 `feature/<area>-<task>`，首次引入一个领域可使用 `feature/<area>`。示例：`feature/softmax-online`、`feature/ptx-wgmma`、`feature/memory-smem-latency`。
- 不在 `main` 直接修改算子代码；改动前检查工作树与当前分支，避免把其他任务的提交带入。
- 提交信息以 `feat:`、`fix:`、`perf:`、`refactor:`、`docs:` 或 `chore:` 开头。
- 不为每个算子维持永久功能分支。算子由目录表示，分支表示一次具体改动。
- 功能分支完成一次独立改动并合并到 `main` 后应删除；同一算子的下一项任务从最新 `main` 新建分支，不继续复用旧分支。
