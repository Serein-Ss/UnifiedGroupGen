# 方法来源、近邻工作与创新边界

现行理论核对日期：2026-11-12。文献支撑所列机制，不意味着完成穷尽的首创性证明。主线见[THEORY.md](THEORY.md)和[RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 1 代码与数据来源

- [CGDiT/newton](https://github.com/Serein-Ss/test_CGDiT/tree/newton)：既有审计提交da3e7554016a2c6b13ae3bc820c9c9cce31a1920。性质条件、周期消息传递与前期对称性处理为方法基础。
- [CrystalFlow](https://github.com/ixsluo/CrystalFlow)、[论文](https://arxiv.org/abs/2412.11693)：既有审计提交9c25ff0245d787efd87c9d1d797a8968443608cc。沿用原来源记录，不表示本次重新核对远端HEAD。
- [PyXtal](https://github.com/qzhu2017/PyXtal)：标准群及Wyckoff表；版本以环境与历史报告为准。
- [spglib](https://spglib.readthedocs.io/)：群识别与设置转换。三维SG不冒充LG识别。

新文件及可选官方主干的来源见原SOURCES.json、vendor_manifest.json。直接复制上游代码须保留许可证；CrystalFlow审计版本的MIT版权包括Xiaoshan Luo (2024)和Rui Jiao (2023)。本轮不改代码归属。

## 2 显式对称生成近邻

| 工作与已核对公开版本 | 已有机制 | 比较边界 |
|---|---|---|
| [MCFlow，ICML2026](https://arxiv.org/html/2602.20210)；[作者机构说明](https://www.lgresearch.ai/blog/view?seq=695) | Generator Matching统一离散元素CTMC与连续几何ODE；单模型多任务 | 单条路径预定N，软排序先验，不采用本项目的整轨道增删和硬约束核 |
| [DiffCSP++](https://arxiv.org/abs/2402.03992) | 晶格对数基与Wyckoff约束 | 显式约束不新 |
| [SGFM](https://arxiv.org/abs/2509.23822) | 固定群与轨道的对称先验、条件向量场 | 合法运动及群平均已有基础 |
| [SGEquiDiff](https://arxiv.org/abs/2505.10994) | 分阶段生成、群等变坐标扩散与切向性 | 不能只换“零空间”名称称新 |
| [SLayerGen，2605.08262v1](https://arxiv.org/html/2605.08262v1) | 空间群、层群及部分周期表示 | SG／LG统一已有先例 |
| [SPFlow，Materials Today Physics 62, 102063 (2026)](https://doi.org/10.1016/j.mtphys.2026.102063) | 连续几何与离散元素／位点对称联合流 | 单网络及混合联合流不是空白 |
| [Orbitol，2610.30086v2](https://arxiv.org/html/2610.30086v2) | 固定占据轨道，联合元素、自由坐标、晶格 | 不预固定元素及联合去噪已有先例 |
| [GEODE，2610.01898v1](https://arxiv.org/html/2610.01898v1) | 模板后原子图联合化学与笛卡尔几何 | 原子图联合元素／几何本身不新 |
| [SymmCD，2502.03638v3](https://arxiv.org/html/2502.03638v3) | 联合位点对称、元素、代表坐标、晶格 | 末端合法化与逐事件物理合法需分开 |
| [SymADiT，2605.10115v1](https://arxiv.org/html/2605.10115v1) | 轨道自编码器与潜空间流 | AE训练阶段不同于先固定化学骨架 |
| [UFO-MGen，2609.26547v1](https://arxiv.org/html/2609.26547v1) | 拓扑、化学占据、固定骨架几何流 | 旧方案直接近邻 |
| [DynaCrys，2608.07401v1](https://arxiv.org/html/2608.07401v1) | 群／轨道／元素符号扩散，再几何生成 | 与固定群下轨道事件不同 |
| [SbCD，2608.13457v1](https://arxiv.org/html/2608.13457v1) | 群跳跃与几何扩散 | 不笼统宣称“跳跃＋几何”首次 |

另核对[SPFlow官方采样代码](https://github.com/SuntaoCoder/SPFlow/blob/master/spflow/pl_modules/discrete_flow_w_site_symm.py)：同一decoder循环更新多变量，末端移除虚拟节点并投影／展开轨道。代表节点规模与完整原子数不能混淆；该路径不同于过程中显式整轨道birth／delete目的核。没有把未检查的其他实现断言为不存在。

## 3 概率过程与学习基础

- [Flow Matching](https://arxiv.org/abs/2210.02747)：条件路径与向量场学习。
- [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660)：流形上的生成。
- [Generator Matching](https://arxiv.org/abs/2410.20587)：一般马尔可夫过程的匹配基础。
- [Branching Flows](https://arxiv.org/abs/2511.09465)：混合变量、分支／删除、锚点与变规模。
- [Trajectory Matching](https://arxiv.org/abs/2610.32602)：带依赖的联合及变规模动作。
- [Discrete Flow Matching](https://arxiv.org/abs/2407.15595)：离散状态流。

加权事件目标与条件程序应明确这些归属。当前增量定位于具体晶体规范、类型化约束、整轨道核和几何对应，不将通用理论改称原创。

## 4 其他保留参考

[DiffCSP](https://arxiv.org/abs/2309.04475)、[CrystalFormer](https://arxiv.org/abs/2403.15734)、[WyckoffTransformer](https://arxiv.org/abs/2503.02407)、[WyckoffDiff](https://arxiv.org/abs/2502.06485)、[WyckoffDiff-Adapter](https://arxiv.org/abs/2601.08115)、[CrysVCD](https://arxiv.org/abs/2507.19799)、[SHAFT](https://arxiv.org/abs/2411.04323)、[SCIGEN](https://arxiv.org/abs/2407.04557)、[SymmBFN](https://arxiv.org/abs/2502.03146)作为周期生成、描述与条件方法来源保留。

内部自旋参考[spin-space-group enumeration](https://arxiv.org/abs/2307.10371)，普通群编号不等于完整磁性表示。对称噪声分布不等于每个样本对称；统计性质条件不等于严格等值面。
