# 统一原始结构评估：CSP / 从头生成 / 性质条件

本实现增加独立评估入口，不修改训练、模型、数据、旧评估脚本或已归档结果。
当前协议为 `unified_raw_v2`。输入可以来自任何模型；原生15套模型另有采样适配器。
v1结果不覆盖，v2使用新目录，避免不同分母或失败处理混用。
只计算原始结构和独立代理性质，不进行弛豫、DFT、凸包稳定性或SUN评价。

## 1. 两类任务及信息层级

|任务|生成时允许的信息|比较对象|主要结果|
|---|---|---|---|
|CSP：指定组成|标准晶胞的各元素整数个数；可选目标性质|同一 `case_id` 的测试结构|组成/化学式/N满足率、Match@1/5/20、最佳候选RMSE/RMSD、每案例候选多样性|
|CSP：固定描述诊断|真实群、Wyckoff轨道、轨道元素；可选性质|同一测试结构|主要检查OrbitFlow；不得与仅组成CSP混作同一信息量|
|DNG：从头生成|随机先验；模型所需结构类型/上下文；可选目标性质|完整测试分布，另对训练集查新颖性|有效性、唯一性、新颖性、VUN、覆盖/距离、结构/元素/群分布|
|DNG：指定群|固定群编号，剩余轨道/元素/连续结构由模型产生|逐群评价|目标操作包含率、精确群率、逐群/宏平均有效性；可用 `--per-group 5`|

组成采用硬约束。当前15套模型 `composition_probability=0`，适配器保持组成神经编码为空，
不会在推理时突然启用未经训练的组成MLP。固定描述分支绕过描述Transformer。
性质使用原有CFG：无目标性质任务 `guidance=0`；性质条件任务默认为2。
联合模型仍输入已有的 `source_domain`，所以是**分来源上下文的共享生成**，不是来源无条件生成。
模块测试使用完整测试集、真实前缀的token指标与真实描述下的流匹配误差；不等于生成重建率。

## 2. 通用结构指标

* 目标输出结构数、成功输出结构数、生成成功率；解析与数值检查另列工程诊断。
* 旧几何规则：原子对距离至少0.5 Å且晶胞体积至少0.1 Å³，保留 `legacy_structural_valid`。
* 新周期距离规则：计入原子自身周期镜像，最近周期距离至少0.5 Å。
* 0.6倍共价半径之和仅保留在附加诊断中：当前没有核验到这个系数对应的论文出处，属于项目自定启发式，不作为主要有效性规则。
* SMACT：电荷中性与Pauling筛选，保留单元素/全金属例外；已启动检查但缺氧化态或搜索超预算时标记计算失败，计入该指标失败。
* `basic_valid`：新周期几何与SMACT联合，**不要求指纹成功**。
* `legacy_joint_valid`：旧几何+SMACT+指纹（开启时），与basic分开报告。
* 元素种类、原子数、体积/原子、密度；二维用面积/原子、面密度、层厚，不用带真空的体密度。
* 共价连接图的周期秩、rank-2原子比例是诊断，不直接等于材料稳定性。
* 群操作包含与精确群重新识别分开：允许额外对称性时，包含成立也可能精确群不相等。
  群重识别分别在1e-4/1e-3/1e-2 Å做一次独立原生子进程检查。
* 原生记录额外检查描述组成、状态展开中的轨道退化/重合。该诊断标记来源为独立展开的状态；
  导出CIF另行做组成与群操作检查，不能把状态检查冒充全部CIF认证。

生成成功率=成功输出结构数/目标输出结构数；结构有效率=有效结构数/成功生成结构数。
不再输出“已评价通过率”或 `rate_on_evaluated`，不使用“未知”掩盖计算失败。
已经启动但没有得到结果的检查计为失败，CSV保存错误类型、阶段和原因。
没有提供预测器或显式关闭的检查为 `not_evaluated`（未运行），不能伪装成已计算失败或成功。
解析、有限值、指纹可计算性放入工程诊断，不作为论文主要生成性能指标。
原生生成在导出前拒绝非有限状态/晶格/坐标，写CIF后立即回读并核对组成；失败不标为生成成功。
不使用NaN替换或事后裁剪伪造有效结果；有限精度下仍可能出现数值失败，应保留种子、错误原因定位。
所有失败保留在生成记录中，不重采样凑出“成功的1024个”。

已核验的来源：
[DiffCSP官方结构有效性实现](https://github.com/jiaor17/DiffCSP/blob/main/scripts/eval_utils.py#L201-L209)
使用0.5 Å距离和0.1 Å³体积；
[CDVAE官方实现](https://github.com/txie-93/cdvae/blob/main/scripts/eval_utils.py)保留同类定义。
本项目增加自身周期镜像检查，须与原始原子对矩阵规则区分。

## 3. CSP协议

|匹配协议|ltol|stol|角度|晶格缩放|
|---|---:|---:|---:|---|
|`legacy_ugg_v1`|0.2|0.3|5°|不缩放|
|`diffcsp_v1`|0.3|0.5|10°|3D允许缩放|

图例和新增 `csp_metrics.csv` 同时写出“Strict”/“DiffCSP”及全部参数，避免只给不透明的协议编号。
DiffCSP参数来源：[官方RecEval/RecEvalBatch](https://github.com/jiaor17/DiffCSP/blob/main/scripts/compute_metrics.py#L109-L180)。
这里对齐匹配参数，不将整个新有效性协议冒充官方原样评估。
默认 `--csp-cases 0 --csp-k 20` 覆盖完整测试集；正数case预算仅用于明确标记的调试子集。
每个测试案例保留20个有固定顺序的候选，以前1、前5、前20个分别算Match@1/5/20，失败占据原候选位置。
不能先按匹配分数从20个中挑最好1个再称Match@1。

3D使用StructureMatcher。二维匹配沿用项目的层材料适配：统一辅助真空用于寻找映射，最终以
物理位移和 `sqrt(A/N)` 判断；不对法向逐原子取模，不用真空长度放宽误差。
因此二维的 `diffcsp_v1` 是**二维适配**，不能标为原论文原样协议。

标准化RMSE、物理Å RMSD、晶格长度/角度/每原子体积或面积误差分别输出；
无法可靠映射的量为空，不填零。Match@K以**所有案例**为分母，而不是匹配成功的候选数。
最佳误差仅在对应K以内匹配且有误差的案例上统计，另保留参与计数。
当前表含全候选匹配误差与每案例最佳误差，二者不混用。
候选唯一性在每个案例内计算；按参考坐标自由度分层。

## 4. DNG分布协议

* 唯一性只在basic有效结构中匹配；报告相对目标输出结构数的有效唯一产率。
* 新颖性必须指定训练参考。只对同约化组成的训练结构做结构匹配，使用严格旧匹配参数。
* VUN是有效、唯一、新颖的交集产率；不是SUN，不声称稳定。
* COV同时输出历史独立最近邻公式，以及要求结构和组成匹配**同一个参考邻居**的新公式。
  保留原指纹实现和阈值（结构0.4、组成10）；二维指纹是项目既有的辅助Voronoi适配。
  统一层COV使用basic有效且指纹成功的生成集合，**与旧脚本的筛选集合可能不同**。
  如需旧论文表格的完全原样口径，继续使用 `scripts/evaluate_generation.py`，不能静默混表。
* AMSD/AMCD、结构和组成指纹多样性、结构分布Wasserstein、元素/群/自由度分布JSD。
* CMD标为 `cmd_50_union_range_normalized`：联合范围归一化的50阶中心矩距离，是明确命名的本实现版本，
  不默认等同于其他论文的CMD数值。
* FWD接口只消费独立嵌入与元数据；没有嵌入则not_evaluated，不能用描述网络隐藏层冒充独立Wrenformer结果。
* 模板为群+允许重复的Wyckoff字母列表；模板新颖性对TRAIN和TEST分别标注参考对象。

## 5. 性质评价

性质CSV必须来自独立评估器，并绑定生成ledger的SHA256。FE/BG适配器可读取现有CGDiT冻结预测器，
在CPU构图、预测并缓存同一预测器在完整测试参考上的结果。
缺少凸包能预测器时，`energy_above_hull`目标保留、评价为空，**不由形成能推算**。
MP-20训练的预测器用于C2DB时，必须显式 `--allow-proxy-transfer`，标为探索性跨域预测，不能作为二维性质控制已证实。

等值条件默认容差为FE 0.1 eV/atom、BG 0.2 eV、Eh 0.05 eV/atom，可在request中预先指定。
同时支持 `eq/le/ge/between` 的独立评价。输出单性质误差/命中、联合命中、basic有效联合命中，
以及同一代理下生成/测试参考的Wasserstein距离。输出每个目标组的预测分布；目标输入值不是预测值。
未给性质评估器时仍可完成结构评估，性质状态保持not_evaluated。
预测器保存的测试MAE只是其历史训练域验证信息，不代表在新材料或二维材料上已校准。

## 6. 模型无关输入示例

`samples.jsonl` 每个请求序号记录一行成功或失败，可使用CIF、pymatgen结构字典、原子数组或原生descriptor/state：

```json
{"index":0,"status":"generated","cif_path":"sample_00000.cif","case_id":"test-001","properties_requested":{"band_gap":1.0}}
{"index":1,"status":"failed","case_id":"test-001","reason":"generation failure"}
```

CSP参考JSONL必须有唯一 `id` 和可解码结构，每个候选（包括失败）必须有对应 `case_id`。
通用CSP评估不要求参考结构本来来自UnifiedGroupGen。
若没有dataset字段，传入dataset标签不会丢弃这些通用参考；有dataset字段则按标签筛选。
`request.json`：

```json
{"case_ids":["test-001"],"attempts_per_case":2,"properties":{"band_gap":{"op":"eq","value":1.0,"tolerance":0.2}}}
```

代理 `predictions.csv` 以 `index,formation_energy,band_gap` 为列，启动预测但数值为空时标为计算失败；没有提供该性质的预测器则标为未开展该项评价。
`evaluator.json` 至少包括 `evaluator` 与 `samples_sha256`。
如使用 `--reference-assessments`，参考CSV列为 `id,formation_energy,band_gap`，元数据还要有
`reference_sha256`（参考文件绝对路径到SHA的映射）和 `reference_predictions_sha256`。
独立嵌入JSON字段为 `evaluator,samples_sha256,reference_sha256,generated,reference`，
generated按尝试index索引、reference按id索引，每项为相同维数向量；至少各2个有限向量。

```bash
python scripts/evaluate_structures.py --samples <samples.jsonl> --task csp --kind space \
  --reference-files <test.jsonl> --request <request.json> --output <new-output>
python scripts/evaluate_structures.py --samples <samples.jsonl> --task dng --kind layer \
  --reference-files <test.jsonl> --novelty-files <train.jsonl> --output <new-output>
```

输出 `samples.csv`、CSP的 `cases.csv`、`summary.json`、`manifest.json`，以及PNG：
`metrics.png`、`distributions.png`、有群标签时 `groups.png`、有代理预测时 `properties.png`。
`--resume`只有在输入、代码、协议和输出哈希一致时复用完整评价；损坏结果不覆盖。
原生采样按每尝试独立seed续跑，已生成CIF需逐个通过哈希。
完整JSON行后的不完整尾行不会自动删除；发现则停止并保留证据。
独立指标任务中断可重算该任务；历史reference缓存按数据、kind、代码和依赖版本隔离。

服务器15套批处理与资源边界见 [REMOTE_UNIFIED_EVALUATION.md](REMOTE_UNIFIED_EVALUATION.md)。
