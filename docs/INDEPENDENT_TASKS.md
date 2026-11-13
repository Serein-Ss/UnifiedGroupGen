# 三项独立任务：joint_flow_v2
2026-11-13。主模型为共享周期EquiformerV2，从空图开始；JointAtomGraph为同路径对照。每项独立训练、独立优化器和检查点，不混合为一个任务。

| 数据 | 无性质生成 | 性质生成 | 精确组分CSP |
|---|---|---|---|
| MP-20 | flow_mp20_dng | flow_mp20_property | flow_mp20_csp |
| C2DB-51 | flow_c2db51_dng | flow_c2db51_property | flow_c2db51_csp |

新入口默认从操作编译约束并缓存。旧task_*和joint_symmetry_*保留兼容，不代表当前主模型。
DNG和性质任务联合连续几何流、轨道MASK离散流及轨道跳转。限定元素只限制允许集合，可出现任意非空子集。
性质仍使用原编码器及既有直接条件配置，本次没有重设计CFG或交叉注意力。
CSP使用ChemicalControl与可完成性见证，带元素出生/替换、删除返还库存；不独立抽MASK元素，因此DFM元素项为零。冻结配置需准确计数，最终核验CIF。
训练为批图单次共享核心前向；生成用异步时间活动图合批，逐样本随机源和失败账本保留。
新检查点ugg_joint_flow_v2保留原生Lightning续训；旧检查点继续加载旧结构。
完整命令、路径与边界见[JOINT_FLOW_V2.md](JOINT_FLOW_V2.md)，数学见[THEORY.md](THEORY.md)。
短测不是完整训练，不给出材料质量、稳定性或性质命中结论。
