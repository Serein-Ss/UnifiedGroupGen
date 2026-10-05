# 实现说明与基础来源

## 1. 与前期代码的关系

审计的 CGDiT/newton 提交：`da3e7554016a2c6b13ae3bc820c9c9cce31a1920`。
CrystalFlow main HEAD：`9c25ff0245d787efd87c9d1d797a8968443608cc`。

| 基础能力 | 来源 | 本实现 |
|---|---|---|
| 标量/类别性质、缺失/空条件 | newton `conditional_embedding_utils.py` | PropertyEncoder；独立字典输入；scalers 仅来自训练集 |
| CFG | newton 与 CrystalFlow | 前端 logits 和后端速度都支持；硬条件保留 |
| CSPNet 周期边特征/消息传递 | newton `decoder/cspnet.py`、CrystalFlow | 独立 PyTorch 完整图打包实现，免 PyG/torch-scatter |
| 晶格/周期坐标联合 FM | CrystalFlow `pl_modules/flow.py` | 从独立合法参数构造路径，使用 Heun 积分 |
| 轨道代表与展开 | newton `generation/symmetry.py` / `rl/symmetry_quotient.py` | 从奇异 Wyckoff 参数映射取真实自由度；不为每个代表固定分配 3 维 |
| 晶格对数表示 | newton / CrystalFlow / DiffCSP++ | 从群操作构造不变对称基底和 SPD 指数图，而非晶系编号手写掩码 |
| MP20/MPTS52/C2DB 数据 | 本地前期工作 CSV | 只读复用，明确列映射、坐标与晶胞转换来源 |

这里继承方法基础，采用新的独立代码接口。没有复制整个旧项目、旧环境路径、凭据、RL 作业或无关预测器。旧检查点不是新模型的兼容权重；正式预训练迁移需要显式形状/语义映射及独立验证。

## 2. 数据记录协议

JSONL 每条记录：

```json
{
  "id": "source-material-id",
  "dataset": "mp_20",
  "descriptor": {"kind": "space", "number": 2, "orbits": [{"letter": "i", "atomic_number": 14}]},
  "state": ["k parameters first, then orbit q parameters; actual values are numbers"],
  "properties": {"formation_energy": -1.0, "band_gap": null},
  "provenance": {"coordinate_mode": "fractional", "cell": "pyxtal_conventional"}
}
```

state 的维数由编译器唯一计算，不是固定 3N+6。`offsets` 给出每个轨道在 q 中的位置；`lattice_dof` 给出 k 的长度。零维轨道不占 q，但仍占原子、元素和轨道 token。

性质 canonical names：formation_energy、band_gap、energy_above_hull。C2DB 的 hform/gap/ehull 按配置对应这些名称；计量定义和计算来源仍需分别审核，命名统一不代表能量标签可直接跨库混合。

## 3. 状态和物理约定

- 列向量晶格；pymatgen 行向量晶格只在导入/导出转换。
- SG：三周期分数坐标。
- LG：xy 周期分数坐标，z 为 Å 高度；参考法向度量固定 1。
- PG：二维周期坐标嵌入 z=0，不额外生成层外变量。
- 晶格先验中心 k=0 对应参考 H0，不是零体积晶格。
- 核空间先转为子空间投影，再按固定列顺序正交化，固定晶格参数基底的顺序和符号，避免任意 SVD 基底旋转改变序列化 k 的含义。
- 仿射观测作用于固定提升图中的 z；不在每一步独立 wrap z，以免破坏混合线性约束。

## 4. 数据转换的实际边界

三维提取使用 PyXtal；若其 site.wp 与默认群表不是同一 Hall 设置，明确拒绝，不以相同字母猜测对应关系。常规胞扩大计数保存 source_atoms/expanded_atoms。

正式 MPTS52 配置使用统一 0.01 Å 的识别容差，与原小模型配置的 0.001 Å 分开缓存；原因、原生库失败及容差敏感性在 PRETRAINING_READINESS.md 中记录。新版层群匹配使用平面物理度量，将 xy 分数偏差和 z Å 偏差都转换为 Å 后判断。

旧二维诊断以 lgnum 为指定群，通过操作闭合与轨道匹配接受标签。正式 `layer_group_policy: identify` 从恢复的完整晶胞调用 spglib 原生层群识别/标准化，再提取轨道，保存原标签、识别标签、标准化变换和原操作残差。无层群标签也可识别，失败不回退。正式统一0.1 Å，原胞/常规胞与标准化变换均可追溯。

本地错误 Cartesian CIF 不保存原始晶格矩阵的朝向，仅保留长度角度。为中心化结构尝试的平面朝向受标签约束，但仍是**推断**，记录 `cell_orientation_inferred`。正式数据需要原 ASE cell 与 positions 复核；不得将推断匹配称为原数据已经完全恢复。

容差允许近对称的源结构转换为严格群参数化端点，这改变了细微坐标。源条件、转换和容差均应在论文说明；原始/理想化结构的性能不能混报。原生层群标准化已实现并通过512条诊断，每条记录保存固定标准晶胞中的坐标RMS/最大偏差与二维度量相对变化；全量转换/统计仍未完成。失败清单保留并计入覆盖率。

## 5. 训练与采样

训练 joint CE+FM，离散模型不需要穿过采样步骤反向传播。描述序列使用 padding/causal attention，几何网络将多个完整图打包成批次，图之间不连边。PropertyEncoder 的整体性质 dropout 保证 CFG 的全空性质分支在训练中出现；共享配置的 source_domain 是必填上下文，在两分支都保留。组成条件随机保留/缺省，固定观测随机选择晶格或部分参数。

推理在指定 kind 的候选群中用 Transformer 概率生成描述；没有 p_emp(G) 抽样。给定固定描述时直接走流模型。前端/后端都可不提供性质，实现同一体系的无性质条件生成。

固定组成不包含“选择最佳原胞倍数”的额外搜索；用户给定的整数计数就是当前表示的晶胞计数。整数倍搜索属于后续离散规划扩展。

## 6. 当前验证器能说明什么

可以检查指定操作闭合、晶格残差、轨道正则性、原子重合、最小距离及 spglib 实际 SG/LG 编号。二维识别使用 get_layergroup，失败记为未知；包含指定操作与恰好识别为该群分别统计。80个合法双轨道合成结构均精确回收目标层群。这些检查不包含价态、MLIP、DFT、凸包和声子。

采样及audit CLI使用独立原生识别进程，20秒超时或非零退出均返回未知编号并记录原因，避免spglib原生终止杀死整个批次；不猜测P1。正式held-out损失入口绑定已审计的完整测试文件与对应训练/验证哈希，同时按来源报告标签覆盖与损失，不能把总平均损失当成每个来源都表现良好的证明。

最小距离阈值目前为统一 0.6 Å 诊断值，不是完整元素依赖的化学有效性标准。未来正式评价需要统一半径规则和 SMACT/化学测试，并将成功/失败和求解成本分别报告。

## 7. 接口扩展顺序

优先完成数据来源确认与三维公平基线，再做共享二维、目标条件消融及共同观测任务。非线性曲面、动态群/轨道跳转、磁群枚举和自旋生成训练不是当前实现已完成的能力。

完整坐标的结构条件 O f=c 可以用编译后的 b、A 转为 O A q=c-O b，再组成 A_obs z=b_obs 输入现有 AffineObservation。固定物理晶格先用 encode_metric 转为 k，再固定对应索引。
