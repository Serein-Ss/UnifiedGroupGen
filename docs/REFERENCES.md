# 文献与代码来源

以下链接用于定位方法与创新边界。日期为本项目会话检索/审计日期 2026-10-04；不是穷尽文献清单。具体差异见 RESEARCH_PLAN.md。

## 直接方法来源

- [CGDiT/newton](https://github.com/Serein-Ss/test_CGDiT/tree/newton)，审计提交 da3e7554016a2c6b13ae3bc820c9c9cce31a1920。
- [CrystalFlow](https://github.com/ixsluo/CrystalFlow)，[论文](https://arxiv.org/abs/2412.11693)。审计提交 9c25ff0245d787efd87c9d1d797a8968443608cc；[具体流匹配文件](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/diffcsp/pl_modules/flow.py)。
- [PyXtal](https://github.com/qzhu2017/PyXtal)：本实现读取空间群/层群 Wyckoff 与群操作表，测试环境版本 1.1.1。
- [spglib](https://spglib.readthedocs.io/)：三维 SG 重新识别；没有把其常规三维输出当作 LG。

本项目新文件独立实现相应算法，没有直接复制 CrystalFlow 的完整模块；该上游为 MIT，版权包括 Xiaoshan Luo (2024) 与 Rui Jiao (2023)，参见[上游许可](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/LICENSE)。如后续直接引入上游代码，应保留许可和版权。

## 晶体与层群生成

- [DiffCSP](https://arxiv.org/abs/2309.04475)
- [DiffCSP++](https://arxiv.org/abs/2402.03992)
- [SGEquiDiff](https://arxiv.org/abs/2505.10994)
- [SGFM](https://arxiv.org/abs/2509.23822)
- [SLayerGen](https://arxiv.org/abs/2605.08262)
- [WyckoffDiff](https://arxiv.org/abs/2502.06485)
- [WyckoffDiff-Adapter](https://arxiv.org/abs/2601.08115)
- [WyckoffTransformer](https://arxiv.org/abs/2503.02407)
- [DynaCrys](https://arxiv.org/abs/2608.07401)
- [SymmCD](https://arxiv.org/abs/2502.03638)
- [SymmBFN](https://arxiv.org/abs/2502.03146)
- [CrysVCD](https://arxiv.org/abs/2507.19799)
- [CrystalFormer](https://arxiv.org/abs/2403.15734)
- [SHAFT](https://arxiv.org/abs/2411.04323)
- [SCIGEN](https://arxiv.org/abs/2407.04557)

## 几何生成与自旋表示

- [Equivariant Flows](https://arxiv.org/abs/2006.02425)
- [Riemannian Diffusion](https://arxiv.org/abs/2202.02763)
- [Riemannian Flow Matching](https://arxiv.org/abs/2302.03660)
- [Spin-space-group enumeration](https://arxiv.org/abs/2307.10371)

当前没有把统计条件生成等同于严格物性等值面生成，也没有将表示约束工具等同于已训练的磁性材料生成器。
