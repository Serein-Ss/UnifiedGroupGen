# 公开预训练性质预测器

核对日期：2026-10-31。本轮接入冻结的公开MEGNet权重，不训练新的预测器，不修改生成器或15套模型。公开模型的可调用性与其在本项目数据上的预测精度是两件事。

## 可用库与本次实现范围

|库|官方公开能力|本项目状态|
|---|---|---|
|MatGL / MEGNet|MP形成能，多保真带隙；MatGL也提供M3GNet等架构|已接入并进行真实权重验证：MEGNet形成能及PBE/GLLB-SC/HSE/SCAN带隙|
|ALIGNN|官方提供mp_e_form_alignn、mp_gappbe_alignn，以及JARVIS形成能、带隙、ehull等性质头|已核对官方模型清单；本轮未实现ALIGNN适配器，不把它标为已验证|
|CHGNet|公开能量、力、应力、磁矩预测和ASE接口|未接入；不把总能量直接标成形成能或凸包能|
|MACE|材料通用势及ASE接口，可计算能量/力等|未接入；常规通用势不等价于现成的带隙性质头|

来源：[MatGL](https://github.com/materialyzeai/matgl)、[MEGNet官方示例](https://github.com/materialyzeai/matgl/blob/v0.9.2/examples/Property%20Predictions%20using%20MEGNet%20or%20M3GNet%20Models.ipynb)、[ALIGNN公开权重清单](https://github.com/usnistgov/alignn/blob/main/alignn/pretrained.py)、[CHGNet](https://github.com/CederGroupHub/chgnet)、[MACE预训练模型](https://mace-docs.readthedocs.io/en/latest/guide/foundation_models.html)。

## 版本和权重

当前基础环境为Python3.10/PyTorch2.7.1。新版MatGL已转向Python3.11以上并改用PyG；不能直接对Python3.10环境执行无版本约束的pip install matgl来复现经典MEGNet模型。本接口选择MatGL0.9.2和DGL1.1.2，使用其官方经典模型API。CPU版DGL不需要匹配当前CUDA图算子ABI；本接口明确在CPU预测。

```bash
conda activate unifiedgroupgen
python -m pip install -e './unifiedgroupgen[pretrained]'
```

环境主清单仍只有unifiedgroupgen/environment.yml，预测依赖是pyproject.toml中的可选extra。此次本地检查把额外库放在intermediate/cache/property_predictors/python，通过测试进程的PYTHONPATH使用，没有升级或替换基础训练环境中的包。独立验证通过后，仅将MatGL0.9.2与DGL1.1.2新增到本机unifiedgroupgen环境；当前可直接调用，pip check通过。PyTorch、NumPy、Lightning版本没有变化。

使用两个权重：

- MEGNet-MP-2018.6.1-Eform，形成能单位eV/atom。
- MEGNet-MP-2019.4.1-BandGap-mfi，带隙单位eV；默认PBE。

官方仓库固定提交为096f512e852bc2a5bd454775d1b890404f559fbc。公开模型文件的SHA256写在common/predictors/public_megnet_models.json。首次运行自动下载到data/scr/pretrained/matgl；以后逐文件校验，发现更改则失败，不静默覆盖。资源JSON只含模型名称、来源和哈希，不含权重。已下载的六个权重/参数文件合计约1.6MB。

原发布model.pt内含Normalizer对象，使用Torch的局部safe_globals允许这一已知类型；不全局关闭安全加载。MatGL原生loader使用strict=False，本适配器随后执行strict=True参数核验。旧model.pt子模型元数据缺少版本键，官方loader会给出版本警告；同包model.json记录MEGNet版本1，当前实现也是版本1，所有参数严格匹配。没有改写原权重或它的元数据。

## 图转换

转换路线为：

```text
UGG独立状态/展开图/JSONL中的结构
  -> 元素 + 分数坐标 + 晶格（明确行列约定）
  -> pymatgen.Structure
  -> 官方Structure2Graph
  -> MEGNet节点类型、周期镜像和距离径向展开
  -> 冻结模型及官方目标反归一化
  -> 形成能/带隙
```

UGG的学习节点向量、轨道嵌入、傅里叶边特征不能直接作为MEGNet输入。预训练模型有自己的元素词表、嵌入和径向特征。模型仅收到物理结构与所选带隙理论级别，不收到生成时要求的目标性质数值。

展开图的lattice是列晶格，默认先转置；其他来源的行晶格必须显式lattice_columns=False。转置后的非连续数组复制为C连续数组，满足官方Cython邻居构建器要求，不改变晶格数值。

二维保留面内晶格和真实面外位移，在辅助晶胞中重新居中。对每个模型的截断半径r，采用H=层厚+2r+2 Å。构图后断言所有边的面外周期镜像编号为0。两个模型各用自己的官方截断半径，不能强行共享邻居图。

目前支持小批量图预测；batch_size只改变推理分批，不改变生成器的训练batch。异常结构及失败样本保留原index，预测列留空并记录prediction_error；不丢弃失败、不把NaN写成有效结果，不裁剪负预测值冒充物理合理。

## Python直接调用当前图

```python
import json
import torch
from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.common.predictors.matgl import MatGLPredictor

root = project_root()
record = json.loads((root/'examples/mp20_walkthrough/record.json').read_text())
compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
expanded_graph = compiled.expand(torch.tensor(record['state'], dtype=torch.float64))
predictor = MatGLPredictor(root/'data/scr/pretrained/matgl')
print(predictor.predict_graph(expanded_graph, kind='space'))
# 对MoS2示例读取examples/c2db_walkthrough/record.json并使用kind='layer'。
```

若已有pymatgen.Structure列表，可以调用predictor.predict_structures(structures, kind='space', batch_size=8)。不需要先写CIF文件。

## 生成结果批量预测与统一评估

以下路径中的SAMPLES需要指向已经生成的某个任务的samples.jsonl。既支持inline CIF，也支持已有外部CIF、pymatgen字典、行晶格数组和descriptor/state记录。

```bash
SAMPLES='/absolute/path/to/task/samples.jsonl'
PRED='outputs/property_predictions/mp20_public_megnet'

python -m unifiedgroupgen.evaluate task=predict_properties device=cpu \
  property_predictor.backend=matgl property_predictor.kind=space property_predictor.dataset=mp_20 \
  property_predictor.samples="$SAMPLES" \
  'property_predictor.reference_files=[data/dataset/full_verified_v1/mp_20/test.jsonl]' \
  property_predictor.reference_cache=intermediate/cache/public_property_references \
  property_predictor.batch_size=8 property_predictor.output="$PRED"

python -m unifiedgroupgen.evaluate device=cpu \
  evaluation.samples="$SAMPLES" evaluation.task=dng evaluation.kind=space evaluation.dataset=mp_20 \
  'evaluation.reference_files=[data/dataset/full_verified_v1/mp_20/test.jsonl]' \
  evaluation.conditional.enabled=true \
  evaluation.assessments="$PRED/predictions.csv" \
  evaluation.assessment_metadata="$PRED/evaluator.json" \
  evaluation.reference_assessments="$PRED/reference_predictions.csv" \
  evaluation.output=outputs/evaluations/mp20_public_megnet
```

CSP任务使用evaluation.task=csp并保留原任务的case_id/reference对应关系及request文件；预测接口对CSP/从头生成通用。性质目标和命中容差仍通过统一评估的request.json提供，不把目标数值输入预测器。

二维任务将property_predictor.kind设为layer、dataset设为c2db_51、参考文件设为data/dataset/full_verified_v1/c2db_51/test.jsonl，并设置property_predictor.allow_proxy_transfer=true。该标志只是确认进行迁移预测，不会提高或认证预测精度。

每个预测任务输出predictions.csv、reference_predictions.csv、evaluator.json。模型身份、单位、带隙理论级别、请求/成功/失败数量和数据哈希写入evaluator.json。evaluate.py读取这些文件后，按原协议输出指标CSV、汇总JSON与PNG，包括性质目标误差、命中比例及同一预测器下的分布距离。参考预测缓存按参考输入、模型、代码和二维处理设置绑定；外部CIF变更也会拒绝复用。

CGDiT backend保留原预测与输出协议；当前需设置property_predictor.backend=cgdit及cgdit_root、predictor_root。新YAML默认选matgl；旧命令只在tests/legacy归档，当前通过evaluate.py调用。

## 解释范围

- 这是对原始几何的冻结性质代理，没有结构弛豫。
- MP来源公开权重可能见过MP-20参考结构；重合未审计，不能把当前参考集误差称为独立泛化精度。
- C2DB精度未验证。即使图在几何上正确适配，也不说明训练域、泛函、能量参考或带隙定义一致。建议后续在来源清楚的标签集上比较误差与排序；本轮未宣称完成该精度验证。
- 本接口不输出energy_above_hull。不能由形成能单值直接推出凸包能；还需要一致的相竞争结构与能量基准，或专门的已验证性质模型。ALIGNN有公开JARVIS ehull头，但本轮尚未接入。
- 多性质目标若含尚未预测的性质，不能依据其他两项的结果宣称所有目标都达标。
- TiO2/MoS2集成检查用真实示例结构验证接口，不是新的生成质量实验，也不是预测精度基准。

显式真实权重检查（需要可选依赖及网络或已有模型缓存）：

```bash
python -m unifiedgroupgen.tests.check_public_predictors \
  --output intermediate/checks/public_predictors/real_weights_check
```

默认离线单测不下载模型。最终本地证据见历史工程记录（已清理），原始临时日志和示例图已清理，需要时重跑对应检查/26-10-31。
