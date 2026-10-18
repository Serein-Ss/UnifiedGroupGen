# 论文图表输出

2026-10-18更新。当前文章为 [UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md](../../docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md)，图像已放入对应方法、实例、结果和实验位置。新增图只保存PNG；缺失结果用明确灰色占位，不填模拟数值。

## 当前主文图

| 文件 | 状态与内容 |
|---|---|
| fig01_unified_method.png | 已绘制：统一编译、共享生成、TiO₂和MoS₂真实编码示例的概念图 |
| fig01_material_examples.png | 已绘制：由真实独立状态展开的原子坐标与晶格，配套第8节 |
| fig02_representation.png | 已绘制：102610条准备记录、2280种单轨道类型、48条来源重编码抽查 |
| fig03_shared_generation.png | 已绘制：最新共享检查点六组1536次采样的原始群／几何诊断 |
| standard/03_shared_generation_validity.png | 已绘制：三来源无性质条件标准质量及Wilson区间 |
| standard/03_shared_generation_coverage.png | 已绘制：COV-P/R和有效唯一新颖产率，标明MPTS52参考缺失 |
| standard/03_shared_generation_by_dof.png | 已绘制：坐标自由度分层质量，保留实际样本数 |
| figS_layer_connectivity.png | 已绘制：C2DB真实测试参考与两种生成模式的周期连接秩，显示条件组偏差 |
| fig04_continuous_controls.png | 占位：同描述先验／学习流、结构观测残差、自由部分质量 |
| fig05_property_control.png | 占位：独立性质响应、CFG扫描、有效命中产率 |
| fig06_physical_validation.png | 占位：弛豫、兼容相图参考、物理候选与DFT／声子 |

补充 figS_joint_generated_dof.png 使用本轮真实采样的坐标自由度；figS_joint_sampling_cost.png 使用每组256次尝试的实测平均耗时，含识别及写盘。原始分组表、逐尝试表、测试损失、时间及检查点／归档指纹位于 tables。

可复现来源为 source/material_examples.json 和 source/build_manuscript_figures.py；图1的图像生成内容与核查记录为 source/fig01_spec.md。方法图为PNG位图，未宣称矢量可编辑。

```bash
python scripts/import_joint_archive.py --archive /path/to/UnifiedGroupGen.zip
python assets/paper/source/build_manuscript_figures.py \
  --joint-run artifacts/imported_server/joint_best_seed142 \
  --standard-metrics artifacts/evaluations/joint_best_seed142_metrics/summary.json
```

导入只提取结果和对应最佳检查点，不覆盖代码。绘图脚本读取原始记录复核计数；已被真实结果替换的占位图不会被通用占位流程覆盖。原始0.6 Å几何诊断与下述标准质量评价分别保存。

标准评价已完成六组：1316/1536联合有效，961/1536有效唯一新颖。MP20/MPTS52/C2DB无性质条件联合有效计数为218/256、217/256、220/256。C2DB条件组的周期连接秩2计数仅122/256，参考为1667/1739；正文保留这一失败分析，不能由高SMACT通过数推断单层连接或性质达标。

tables/joint_standard_summary.json 和六份逐尝试CSV保留完整评价分母与原始字节；standard保存汇总表、分层表和PNG。表中无独立计算的性质及稳定性保持null。重画标准图不需重新计算指纹：

```bash
python scripts/build_paper_results.py --manifest assets/paper/source/standard_results_manifest.json --output assets/paper/standard
```

图形作者输入的文本SHA按LF换行规范化，具体定义写入figure_data_manifest.json；导入的标准指标、逐样本表和CIF指纹使用原始字节SHA。

完整图1–6设计、各图应支持的结论和当前缺口见 [论文证据计划](../../docs/PAPER_EVIDENCE_PLAN.md)；命令见 [服务器执行流程](../../docs/PAPER_WORKFLOW.md)。现有数据图、表示检查和模型架构分别在 ../datasets、../representation、../architecture。

## 标准评估后的汇总图

以下图由 scripts/build_paper_results.py 根据已完成标准指标生成，不由原始完成日志推导。

| 输出 | 内容 |
|---|---|
| 03_shared_generation_validity.png | 三来源无条件生成有效性和精确群回收，全部尝试分母 |
| 03_shared_generation_coverage.png | COV-P/R、有效唯一新颖产率，每个采样任务独立显示 |
| 03_shared_generation_by_dof.png | 按坐标自由度分层的联合有效率与数量 |
| 04_continuous_controls.png | 同描述/种子的合法先验、训练流、固定晶格对照 |
| 04_observation_residuals.png | 固定观测残差累计分布 |
| 05_condition_response.png | 独立性质值相对请求目标的中位数/IQR与CFG扫描 |
| 05_condition_valid_yield.png | 联合有效率与有效性质命中产率 |
| 06_physical_screening.png | 有兼容能量参考的弛豫后筛选与SUN产率 |
| 06_relaxation_symmetry.png | 弛豫前后均成功识别的样本中，回收群不变的比例与计数 |

当前已导入并核验本轮共享生成，并完成标准质量补算。连续对照、性质与物理占位保留，待实际实验后原位更新；工程测试夹具不进入论文结果。

## 材料结构显示风格

正文 TiO₂/MoS₂ 数值结构图采用共同的 Crystal Toolkit 风格渲染器：VESTA 配色、均一半径哑光球体、半透明配位多面体、分色显示键、细灰晶胞边框和三色方向轴。三维保留周期像，二维仅面内重复。图1和架构图中的插图为方法示意；数值几何以独立状态展开图为准。更多视图、显示键规则与重建命令见 [结构图说明](../structures/README.md)。
