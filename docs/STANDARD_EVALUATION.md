# 统一表示与生成结构标准评估

本页对应两个独立的 CPU 评估入口。图只保存 PNG；CSV/JSON保存数值与分母。
已有生成任务继续执行，待其完成后再更新服务器代码。`evaluate_joint.py --resume`会检查源码指纹，更新代码后不能直接沿用旧的恢复身份；已完成结果可以由新后处理入口读取。

## 1. 更新与环境

```bash
conda activate cgdit
cd /path/to/UnifiedGroupGen
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python -m pip install --no-deps --no-build-isolation -e '.[test,evaluation]'
python -c "import torch, smact, matminer, matplotlib; from importlib.metadata import version; print({n:version(n) for n in ['torch','smact','matminer','matplotlib']})"
```

若最后一行提示评估依赖缺失，安装后再检查：

```bash
python -m pip install -r requirements-evaluation.txt
```

这里只补评估包，不重新安装 PyTorch。若解析器提出替换已有核心科学计算依赖，先查看解析输出并保留环境记录；现有 `requirements-remote.txt` 和 `constraints-tested.txt` 的历史 torch版本不能直接套用到当前2.2.1环境。

```bash
python -m pytest -q tests/test_standard_metrics.py
```

## 2. 数据能否统一表示

### 2.1 完整执行

```bash
mkdir -p artifacts/evaluations
nohup python -u scripts/verify_representation.py \
  --workers 4 \
  --output artifacts/evaluations/representation \
  --figures assets/representation \
  > artifacts/evaluations/representation.driver.log 2>&1 &
echo $!
```

默认检查联合划分的所有train/val/test记录，合计102610条；同时从原始/恢复后的来源CSV找到相同CIF哈希，重新执行来源编码，逐条与保存状态比较。
这一步不加载训练权重、不使用GPU。完整来源重编码包含群识别和标准化，成本可能明显超过纯代数检查，不能用采样生成的耗时估计。

来源路径由配置确定。MP20/MPTS52优先使用便携包的 `raw_data/<dataset>/{train,val,test}.csv`；C2DB使用已核验恢复目录 `artifacts/source_recovery/c2db_official/`。
历史 `provenance.source_csv` 中的Windows路径不作为服务器文件地址。来源身份使用CIF原始字节SHA256；找不到的条目作为未评估保留，不用近似材料替换。

若想先核对部分来源，可用 `--source-limit 128`：仍检查全部准备记录和全群目录，但每来源只对可复现选出的128条来源CIF重编码；报告明确标记subset。`--source-limit 0`为全量（默认）。
`--no-source-check`只检查代数表示；`--limit N`限制每来源记录数，报告标为partial，不能作为全数据结论。

### 2.2 统计含义

| 输出 | 含义 |
|---|---|
| `records.csv/jsonl` | 每条记录的状态维数、k/q维数、展开原子数、编码求逆误差、晶格正定性、群操作残差、合法路径抽查残差 |
| `catalog.csv` | 230 SG、80 LG、17 PG的所有单轨道类型；轨道基底秩、锚点稳定子约束核维数、群作用切向一致性、晶格基底正交误差 |
| `source_reencoding.csv/jsonl` | 来源CIF独立重编码后的标准胞坐标与晶格对比；失败和未找到来源的记录保留 |
| `summary.json` | 各来源与群类别的总数、通过数及min/p50/p95/p99/max；源码输入的文件指纹和统计范围 |
| `assets/representation/representation.png` | 数据数值误差累积分布及全目录轨道检查计数 |

单条记录的独立参数逆映射误差、群操作残差和路径抽查阈值为1e-7，采用float64。晶格残差按度量范数归一化，正定性单独检查。
来源重新编码在保存的标准胞下允许一个共同原点平移和同元素原子重排：最大坐标误差阈值1e-5 Å，相对晶格度量误差阈值1e-5。
来源容差沿用各条记录的历史设置；来源群识别改变、晶胞约定改变和数值超限均报告失败，需检查具体原因，不自动改为P1或丢弃。

独立参数求逆检查的是**准备好的表示内部**是否一致；来源重新编码检查的是**真实来源→表示**是否可复现。来源原始结构的标准化/理想化误差是第三个量，使用记录中已有的 `source_operation_residual_angstrom`、`std_position_*`、`std_metric_relative_change` 等字段单列。缺少字段的老3D记录标为未提供，不能将其视为原始结构零误差。

核维数通过单轨道锚点的全部稳定子求得，避免为高重数轨道建立巨大的3N密集约束矩阵。不同轨道实例的坐标自由度按直和组合。该检查覆盖目录中的所有单轨道类型，不声称枚举无限的重复多轨道列表。
路径在起点、中点、终点抽查，联合解析参数化与轨道核验证提供依据；数值抽查本身不构成所有连续时刻的独立数学证明。

```bash
tail -n 20 -F artifacts/evaluations/representation.driver.log
python -m json.tool artifacts/evaluations/representation/summary.json
```

来源重编码有原生库调用，工作进程隔离并限时；超时或原生崩溃会中止该次来源重编码并留下已写日志，不允许将未处理行记为通过。按报错处理后再重跑。

## 3. 已完成联合生成的标准指标

确认 `artifacts/evaluations/joint_best_seed142/pipeline.json` 的state为complete，再执行：

```bash
nohup python -u scripts/evaluate_generation.py \
  --run artifacts/evaluations/joint_best_seed142 \
  --checkpoint artifacts/runs/joint_space_layer_large_seed42/model.best.pt \
  --output artifacts/evaluations/joint_best_standard \
  --figures assets/generation_metrics \
  > artifacts/evaluations/joint_best_standard.driver.log 2>&1 &
echo $!
```

无需重新生成CIF。脚本先核对生成文件、实际检查点、训练参考文件及测试参考文件的SHA256，再读取六组输出。
三维的新颖性参考同时包括MP20和MPTS52训练结构，因为共享模型见过两套来源；二维参考包括C2DB训练结构。测试集仅用于覆盖/分布比较，不加入训练参考来改变“是否见过”的定义。
参考指纹在输出目录缓存，文件哈希、指纹代码和主要依赖版本一致时复用，六组评估不会重复计算同一来源测试指纹。

首次计算CrystalNN指纹和训练集结构展开可能较慢，主要使用CPU和内存。结构匹配按约化化学式分组；覆盖距离矩阵分块计算，不一次建立全部生成×测试矩阵。

```bash
tail -n 20 -F artifacts/evaluations/joint_best_standard.driver.log
```

### 3.1 指标口径

| 指标 | 当前实现 |
|---|---|
| `legacy_structural_valid` | 前期CGDiT的0.5 Å原子对距离/0.1 Å³体积门槛；保留原基准口径 |
| `periodic_distance_valid` | 额外检查全部近邻中的周期镜像，包括同一原子的镜像；单原子小晶胞也可失败 |
| `radius_distance_valid` | 额外共价半径比例距离阈值0.6；缺少半径数据为未知 |
| `composition_valid` | SMACT电荷中性/Pauling筛查，保留单元素/金属合金例外；缺少氧化态或搜索预算超限为未知 |
| `joint_valid` | 结构与组成同时有效；开启指纹时，指纹计算失败也使该传统联合指标失败，另存错误原因 |
| `exact_target_group` | 默认使用生成时已保存的独立识别结果；`--reidentify`可对CIF重做隔离SG/LG识别 |
| COV-R/P、AMSD/AMCD | 61维平均CrystalNN-ops与132维Magpie组成指纹；组成标准化常数来自前期CGDiT，随包分发 |
| Wasserstein距离 | 3D密度、体积/原子、元素种类数；2D面积/原子、层厚、元素种类数。生成侧使用联合有效集合，参考侧使用该来源测试集 |
| Unique | 联合有效结构按StructureMatcher去重，以首次出现者作为代表 |
| Novel | 相对指定训练参考结构的匹配；不代表相对所有已知材料数据库的新颖性 |
| `valid_unique_novel_count` | 有效、唯一且训练参考中未出现的独立结构数量及占全部尝试的比例 |
| `periodic_connectivity_rank` | 共价半径1.2倍成键规则下，商图平移循环的最大分量秩；0/1/2/3区分孤立、链、层、三维网络，属于额外诊断 |

默认结构匹配容差为ltol=0.2、stol=0.3、angle_tol=5°、scale=False，与前期稳定性/新颖性代码口径相近；CSP的stol=0.5、ltol=0.3、angle_tol=10°是另一任务的指标，不能混用。
二维匹配消除真空和共同法向平移，使用辅助高晶胞寻找基底/超胞/原子映射，最终按真实二维距离及sqrt(A/N)检查，不能靠增大真空提高匹配率。二维体积密度置为null。
CrystalNN的Voronoi在纯部分周期法向上可出现无限顶点。二维指纹使用h=max(24 Å, 层厚+12 Å)的辅助周期晶胞和大于h的搜索半径闭合Voronoi；至少12 Å的层间隔超过化学距离截断，使法向镜像近邻权重为零。该高度由真实层厚确定，与输入真空无关，作为二维适配协议记录，不将辅助法向周期性当成真实材料约束。

覆盖率保留前期实现的“分别寻找最近结构指纹与最近组成指纹”定义。默认阈值0.4/10来自MP20基准；MPTS52和二维使用该阈值属于明确的协议选择，比较其他方法时须统一，并在二维正式实验中做校准/敏感性检查。
COV-P以全部生成尝试为分母；参考指纹失败时COV-R标为partial，并同时提供以所有参考记录为分母的下界。指纹失败数保留，不悄悄缩小分母。
组内失败、未知、指纹失败、SMACT预算超限分别保存。每项有效率同时提供全部尝试分母和已评估分母。

### 3.2 输出

- `summary.json/csv`：六组主指标。
- `<job>.csv/json`：逐尝试状态、额外几何/化学检查、对称性、匹配和文件指纹。
- `<dataset>_reference.json`：测试参考指纹和几何缓存及身份信息。
- `assets/generation_metrics/<job>.png`：有效性与3D密度/2D面积及元素种类数分布对照。所有新绘图只输出PNG。

`--no-fingerprints`为快速诊断：跳过覆盖率及指纹有效性，不与完整基准的联合有效率混报。
`--samples path/to/samples.jsonl --dataset mp_20`可评估旧样本；此模式参考为用户选择，不绑定实际模型训练文件，报告明确标注 `user_selected_references`。

## 4. 独立性质、稳定性和SUN结果接口

本入口不把请求条件当作输出性质，也不在同一GPU上自动启动额外力场/DFT任务。
提供独立计算的CSV和出处JSON后，可以计算性质命中率、弛豫后近凸包比例和SUN。单位为形成能/凸包能eV/atom、带隙eV。

CSV列：

```text
job,index,formation_energy,band_gap,energy_above_hull,relaxation_success,relaxed_cif
```

每行对应生成任务名和原始尝试index；缺少的性质单元格留空。`relaxed_cif`是相对项目根目录或绝对路径，指向独立弛豫后的CIF。
出处JSON至少包含 `evaluator` 和 `units`；若提供凸包能，还必须包含 `energy_reference`，说明势函数/DFT方法、元素参考、相图来源及兼容修正。提供值的科学正确性仍依赖这些计算流程。

```bash
python scripts/evaluate_generation.py \
  --run artifacts/evaluations/joint_best_seed142 \
  --assessments artifacts/evaluations/independent_properties.csv \
  --assessment-metadata artifacts/evaluations/independent_properties_metadata.json \
  --output artifacts/evaluations/joint_best_with_properties \
  --figures assets/generation_metrics_with_properties
```

默认命中容差：形成能0.1 eV/atom、带隙0.2 eV、凸包能0.05 eV/atom；命中还要求生成结构联合有效，并报告多性质联合命中。目标来自真实生成记录，不另行伪造目标。
若还要复现前期 `wdist_prop`，提供由同一个独立评估器计算的测试参考CSV（列 `dataset,id,formation_energy,band_gap,energy_above_hull`），追加 `--reference-assessments <该CSV>`，并在出处JSON中声明 `reference_evaluator` 等于 `evaluator`。脚本验证参考ID属于测试集，报告实际参考数量和完整/部分覆盖。未提供参考预测时性质分布距离为null，不拿不同计算方法的原始标签强行比较。
稳定性筛查阈值为弛豫后Ehull≤0.1 eV/atom。缺少弛豫成功和独立凸包能时不评稳定性。
SUN要求提供弛豫后的CIF，在这些结构上重新判定有效性、唯一性和相对选定训练参考的新颖性；缺少弛豫结构时SUN为null，不能用弛豫前的去重代替。
本项目训练参考上的SUN与MatterGen完整已知材料参考库上的SUN范围不同，不能直接比较数值。力场预测的稳定性也不能写成DFT认证。

## 5. 回传与后续判断

```bash
tar -czf artifacts/evaluations/representation_and_standard_metrics.tar.gz \
  artifacts/evaluations/representation \
  artifacts/evaluations/joint_best_standard \
  artifacts/evaluations/representation.driver.log \
  artifacts/evaluations/joint_best_standard.driver.log \
  assets/representation \
  assets/generation_metrics
```

先检查表示验证的通过数、来源重编码分母及具体失败；再查看六组的几何/组成/指纹有效率、覆盖和新颖性。质量不佳时按失败类型判断离散描述、连续几何或评估器的问题，不通过删除失败或放宽阈值改写主结果。

指标来源：[前期CGDiT](https://github.com/Serein-Ss/test_CGDiT/tree/newton)、[CDVAE官方指标](https://github.com/txie-93/cdvae/blob/main/scripts/compute_metrics.py)、[CrystalFlow](https://github.com/ixsluo/CrystalFlow)、[MatterGen评估说明](https://github.com/microsoft/mattergen#evaluation)。本仓库不依赖前期仓库的Hydra、Lightning或训练模块。
