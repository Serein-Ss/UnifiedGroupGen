# 当前主线：周期EquiformerV2联合流（2026-11-13）

主模型 flow_mp20_dng / flow_c2db51_dng，性质与CSP分别用对应后缀。共享周期图EquiformerV2，从空图开始，联合几何流、元素DFM与轨道跳转；JointAtomGraph同路径对照用 model=joint_flow_atom_graph。

完整命令、任务特化、批量功能与验证边界见 [JOINT_FLOW_V2.md](docs/JOINT_FLOW_V2.md)。旧选择器保留兼容，下方旧版本相关介绍以新说明为准。

当前代码支持 [完整生成轨迹保存](docs/GENERATION_TRAJECTORIES.md)，以及可选的 BF16＋融合 AdamW 执行配置 `runtime=joint_flow_fast`。
正式训练前需要为相同配置执行数据审计；首次启动步骤、RTX 3060 隔离短计时、重计算的显存/速度折衷及尚未完成项见 [训练工程效率](docs/TRAINING_EFFICIENCY.md)。短测通过不等于完整训练收敛或材料质量已验证。

本轮代码发布范围与验证边界见 [2026-11-13 更新说明](docs/RELEASE_20261113.md)。

# UnifiedGroupGen

以群作用为统一约束基础，从简单随机先验生成完整晶体。现行研究主线是**单一完整原子图网络中的受约束联合生成**：元素、晶格、坐标和轨道配置共同演化，网络内部显式读取群操作并施加类型化约束。三维晶体、二维层材料和严格平面结构沿用已有空间维度／周期维度定义。

默认“条件生成”仅指目标性质条件生成。结构类型、群操作及CSP组分属于硬规范／任务输入。三项并列主任务先按相同架构分别训练独立模型：无性质从头生成、目标性质条件生成、指定组分CSP。未指定群时按对应结构类型的TRAIN群频率外部采样，不训练神经选群头。新增task_*选择器已接入TRAIN经验频率和无群输入入口；同权重联合训练留作后续对照。详见[任务与默认术语](docs/TASK_DEFINITIONS.md)和[独立任务实现](docs/INDEPENDENT_TASKS.md)。

## 统一项目网页

项目的统一线上入口为 [UnifiedGroupGen 对称性实验室](https://unifiedgroupgen-symmetry-lab.serein-zhong.chatgpt.site/#overview)。项目与模型、训练与生成、真实结构交互、数学推导及数据分析都在同一站点。当前仅展示MP-20和C2DB-51；本地源码为web/symmetry-lab，发布需通过既有Sites项目，不能把本地构建当成线上更新。

## 现行理论与研究材料

- [完整数学理论](docs/THEORY.md)：固定群规范、动态占据状态、公共状态硬解码、联合跳转核、教师路径、加权目标和事件感知采样。
- [研究方案](docs/RESEARCH_PLAN.md)：同一项目的架构调整、任务范围、创新定位与验证问题。
- [实现对照](docs/IMPLEMENTATION.md)：目标功能、当前代码和未接入接口逐项对应。
- [联合生成器用法](docs/JOINT_SYMMETRY_V1.md) · [操作编译与网络约束层](docs/OPERATION_COMPILER.md)。
- [来源与近邻工作](docs/REFERENCES.md)；中文论文初稿在原站点的[文章工作区](https://unifiedgroupgen-symmetry-lab.serein-zhong.chatgpt.site/manuscript.html)维护，沿用其访问范围。
- [统一项目网页](https://unifiedgroupgen-symmetry-lab.serein-zhong.chatgpt.site/)：当前模型、分开的训练／生成流程、两库分析与真实TiO₂／MoS₂分支。站点源码独立发布，不是训练代码运行的依赖。

现行方案不再以“先固定完整元素／轨道描述，再生成几何”定义整个模型。DescriptorTransformer／OrbitFlow 及旧检查点、默认配置保留为兼容和对照路径，不因文档修订而替换。

当前非磁性特征、CSPNet／GemNet／EquiformerV2对照，以及同一SnS记录的完整编码说明见[通用晶体特征](docs/GENERIC_FEATURES_AND_BACKBONES.md)。标准配置不创建磁矩编码器，旧检查点按原特征契约兼容加载。

## 安装与运行

    conda env create -f unifiedgroupgen/environment.yml
    conda activate unifiedgroupgen_cuda
    python -m pip install --no-build-isolation -e ./unifiedgroupgen
    python -m unifiedgroupgen.run experiment=operation_joint task=plan
    python -m unifiedgroupgen.joint --help

显式启用的现行联合选择器为 joint_symmetry_mp20、joint_symmetry_c2db51、joint_symmetry_joint；启用操作编译及内部约束层的选择器为 operation_mp20、operation_c2db51、operation_joint。配置存在不表示新模型已完成材料质量训练。旧 generate/evaluate 入口继续服务相应旧检查点；不能混用权重格式。

现有联合入口的固定群示例：

    python -m unifiedgroupgen.joint sample --checkpoint outputs/EXPERIMENT/RUN/model.pt --kind space --group 225 --count 16 --output outputs/EXPERIMENT/RUN/generate/fixed_group/samples.jsonl --exact-group

新增task_*检查点已保存TRAIN经验群频率，可省略群编号进行采样；CSP按计数可行性筛选。旧无先验检查点仍需显式规范。环境与旧入口详见[环境说明](docs/ENVIRONMENT.md)、[实验配置](docs/EXPERIMENT_CONFIGURATION.md)及[生成与评估](docs/EVALUATION_TASKS.md)。

## 代码、数据与结果

- group_constraints：规范、轨道模板、操作编译、硬解码和合法事件。
- pl_modules：共享周期 EquiformerV2、JointAtomGraph 对照、内部约束层、联合目标和训练；保留旧模型。
- pl_data：真实数据适配、当前分支锚点与条件事件路径。
- joint_flow：v2连续几何流、离散元素流、轨道跳转、批量采样和轨迹记录。
- generation：保留v1联合采样；conditional：性质与其他条件工具。
- evaluation：既有评估协议；joint 适配器记录完整尝试、失败和最大群检查。

主要训练数据为 data/dataset/full_verified_v1 下的 MP-20 与 C2DB-51。数据划分、编码来源、缺失标签及文件哈希保留；其他历史数据保留在本地档案，不进入当前网页。C2DB 近对称理想化与层群识别容差见[模型输入数据](data/dataset/README.md)。

历史15套两阶段模型的训练／生成／性质代理结果继续保存，不能用来宣称新联合模型性能。数学、教师路径、梯度、短跑、材料质量和DFT证据分别记录；缺失磁矩保持缺失，不填零训练。

    python -m pytest -q

训练与结果仍按 [OUTPUT_LAYOUT.md](docs/OUTPUT_LAYOUT.md) 保存；运行日志、权重及历史数值不由本轮文档刷新重写。理论更新范围、原文快照和验证清单见 [THEORY_UPDATE_20261112.md](docs/THEORY_UPDATE_20261112.md)。

当前三任务的六个task_*选择器、硬组分采样及有限步数测试用法见[INDEPENDENT_TASKS.md](docs/INDEPENDENT_TASKS.md)。本轮只执行短测，没有完整训练或混合任务训练。
