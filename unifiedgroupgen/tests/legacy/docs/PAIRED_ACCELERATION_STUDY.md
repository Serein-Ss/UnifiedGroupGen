# 优化前后配对训练检查

## 目标与范围

在开展15套正式训练之前，检查加速实现是否导致明显的学习或生成质量退化。
本次仅比较 `mp20_all`、`c2db51_all`、`joint_all` 三套代表性多性质CFG模型。
它们不是完全无条件基础模型；无性质采样使用这些CFG模型的空性质分支。

| 数据设置 | TRAIN | VAL | TEST | 实验数 |
|---|---:|---:|---:|---:|
| MP-20 | 27136 | 9047 | 9046 | 优化前、优化后各1个 |
| C2DB-51 | 10143 | 3381 | 3381 | 优化前、优化后各1个 |
| MP-20 + C2DB-51 | 37279 | 12428 | 12427 | 优化前、优化后各1个 |

使用 `source_integrity_v1` 修复数据和完整划分。正式YAML、模型规模、群约束、
性质编码、损失权重和优化器设置均保持不变；实验训练上限通过命令行设为30轮。

## 配对控制

- 优化前源代码：`26183967082bae4142b785e0306a72c1f5a00134`。
- 优化后源代码：`4116dea0e96d7fba39b200ab08ac7979fc486276`。
- 两份源代码均冻结到实验目录，逐文件记录SHA256。
- 每对模型显式加载同一份初始权重；恢复同一份Python、NumPy、Torch CPU/CUDA随机状态。
- 种子42、30轮、batch size 64、BF16、激活检查点、边分块4096、每5轮验证。
- 每轮记录批次顺序及生成批次之后的随机状态SHA256，最终核对两支是否一致。
- LR调度策略相同；若浮点差异触发不同的学习率轨迹，应在历史中保留并检查。
- 顺序为MP前→MP后→C2DB后→C2DB前→联合前→联合后，不并行争用GPU。
- 最新完整轮的模型与优化器/RNG状态可恢复；中断批次轨迹单独保留，不混入完成轮。

RTX 3060使用项目内 `artifacts/acceleration_cuda/Scripts/python.exe`，不修改原cgdit环境。
具体Torch/GPU、数据文件哈希、拟合后的性质标度和训练元素集合写入 `protocol.json`。

## 统一评估

主比较采用各支第30轮模型 `model.pt`，不按不同最佳轮数选择主结果。
六个模型都用同一份冻结优化后推理代码评估，分离训练差异与推理计算顺序差异。
`model.best.pt` 会保留，但不替代主比较模型。

1. 完整TEST上的描述Transformer、流匹配、晶格、坐标损失，保留不支持元素的分母。
2. 每个来源256次无性质目标采样，128次三性质联合条件采样，64个积分步。
3. 联合模型分别请求MP-20和C2DB来源；共3072次生成尝试。
4. 性质目标取本来源TRAIN中形成能排序中位附近、三个性质均有标签的真实记录，
   条件CFG scale=2；具体记录ID、数值预先写入协议，不使用TEST标签选目标。
5. 两支对应任务用相同采样种子142。自由描述可能不同，故后续噪声不能宣称逐结构相同；
   比较的是同等预算下的生成分布，训练的初始化与随机过程另行严格核对。
6. 群操作包含、轨道重数、不重合、晶格度量残差，与独立恢复的精确SG/LG分别报告。
7. 统计几何有效性、SMACT组成有效性、指纹成功、联合有效性、唯一性、新颖性、
   COV及分布距离。2D使用既有面积/层厚、部分周期和指纹适配协议。
8. 指纹参考采用本来源完整TEST；新颖性参考完整同类TRAIN。指纹失败参考仍保留分母和覆盖状态。
9. 失败尝试和群识别未知数保留。有效率图给出95% Wilson区间；这不是多种子置信区间。

本次没有独立性质预测器或DFT，不能根据条件输入宣称性质达标或物理稳定。

## 判断原则

首先核对权重、数据与逐轮随机状态配对是否成立；再查看三类训练/验证曲线和最终TEST损失。
脚本将最终验证损失上升超过5%、联合有效率或精确群率下降超过10个百分点作为排查提示。
这些是预设诊断线，不能据此证明统计非劣。
如果两支生成质量都很低，或样本区间太宽，也不能据此批准15套正式实验。
30轮、单种子的结果用于发现明显退化；最终收敛质量仍需足够预算与重复实验支撑。
脚本不会自动启动15套正式训练。

## 执行与输出

在项目根目录的PowerShell中执行：

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:MKL_THREADING_LAYER='SEQUENTIAL'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
& artifacts/acceleration_cuda/Scripts/python.exe -u scripts/paired_training_study.py
```

默认目录：`artifacts/paired_training_20261023/`。

- `protocol.json`：冻结协议、初始权重和输入哈希。
- `status.json`、`driver.stdout.log`、`driver.stderr.log`：队列状态与总日志。
- `runs/<dataset>/<before|after>/model.status.json`：当前批次。
- `runs/.../model.history.jsonl`：全部训练/验证记录。
- `runs/.../pairing.jsonl`：每轮顺序及随机状态证据。
- `evaluations/`：完整TEST、CIF、逐尝试指标、生成分布PNG。
- `figures/paired_loss_curves.png`：三种数据设置×总/描述/流匹配损失。
- `figures/paired_generation_quality.png`：规则包含、精确群、联合有效率和COV。
- `generation_comparison.csv`、`comparison.json`、`REPORT.md`：完成后自动生成的比较结果。
- `smoke/`：小规模流程检查，明确排除在正式比较之外。

若中断，确认旧进程已退出后重新运行同一命令，已有checkpoint将续训。
正常退出会释放 `pipeline.lock`；异常断电遗留锁应先核实PID再处理。
不覆盖部分生成结果；若采样中断，保留目录并检查原因后再决定补跑方式。
