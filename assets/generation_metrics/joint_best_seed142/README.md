# 当前共享模型逐任务标准诊断

2026-10-18。每项256次尝试，使用已核验服务器CIF；全部六项标准评价完成。图中已知失败、未知与通过分开计数，3D密度／体积和2D面积／层厚分别处理。

- MP20：[无性质条件](mp_20_unconditional.png)、[性质条件](mp_20_conditional.png)。
- MPTS52：[无性质条件](mpts_52_unconditional.png)、[性质条件](mpts_52_conditional.png)。
- C2DB：[无性质条件](c2db_51_unconditional.png)、[性质条件](c2db_51_conditional.png)。

数值、定义和失败分析见 [文章第9节](../../../docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md)。可复核标准汇总为 [joint_standard_summary.json](../../paper/tables/joint_standard_summary.json)，逐样本表位于同目录。MPTS52参考指纹4035/4036，COV-R同时保留全参考分母下界；其他两个来源参考指纹完整。

这些是同一检查点、单采样种子的初评，不是公平基线比较或独立训练重复；性质与稳定性没有独立计算值。二维连通性偏差单列于 [补充诊断图](../../paper/figS_layer_connectivity.png)。
