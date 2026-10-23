# 修复数据上的 15 套主实验与硬约束任务

本方案不增加控制微调。性质编码保持每种性质独立 MLP；群—轨道编译器与图网络候选速度主干保持原方式。旧数据模型保留为历史结果，不用于恢复这批训练。

等价计算加速、4090 同批测速、显存检查与续训选项见 [TRAINING_ACCELERATION.md](TRAINING_ACCELERATION.md)。15 套 YAML 和数据保持不变；`train_main15.py` 支持 `--checkpoint`/`--no-checkpoint`、`--edge-chunk-size` 运行覆盖。

## 1. 训练矩阵

三个数据设置：`mp20`、`c2db51`、`joint`（仅 MP20+C2DB）。每种设置五个后缀：

| 后缀 | 目标性质 | 条件训练 |
| --- | --- | --- |
| `u` | 无 | 无目标性质 CFG |
| `ef` | formation_energy | 单性质 CFG |
| `eg` | band_gap | 单性质 CFG |
| `eh` | energy_above_hull | 单性质 CFG |
| `all` | 上述三项 | 多性质 CFG，整组空分支及非空子集训练 |

所有配置 `composition_probability=0`、`observation_probability=0`。默认 1000 epoch 上限、每 5 epoch 验证、50 个训练 epoch 的早停耐心；不是要求跑满 1000 轮。来源/二维三维任务身份是保留的上下文，`u` 不应解释为连任务身份都未知的混合生成。

独立和联合采用相同来源划分：MP20 27136/9047/9046，C2DB 10143/3381/3381，联合 37279/12428/12427。联合全量审计检查源认证、组成/几何证据、源 CIF 与图指纹候选跨划分重复，失败时禁止训练；没有用三来源联合划分替代。此检查不穷尽原型近重复。

C2DB 16905 条中带隙已知 8686 条，缺失不填零。所有模型保留同一结构池；无带隙记录仅缺失该条件。联合目标性质按来源使用 TRAIN 拟合的均值/标准差；仍保留原单位输入。多性质模型的非空子集从已知目标标签中均匀选择，整组 CFG 丢弃概率 0.1。不同性质及两个网络的编码权重均不共享。

## 2. 服务器更新与准备

已有部署的服务器执行：

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git status --short
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/unpack_training_data.py --bundle datasets/source_integrity_v1
python scripts/prepare_main15.py
python -m pytest -q
python scripts/train_main15.py --plan
```

新服务器先执行一次 `python scripts/unpack_training_data.py` 导入基础数据包及 C2DB 恢复证据，再导入追加的 `source_integrity_v1` 包。已有数据不需要重复转换原始 CIF。无需安装另一份 PyTorch。`prepare_main15.py` 使用现有来源 JSONL，先对每种数据设置完整审计一次；其余性质变体复用完全相同输入的完整性检查，并保存独立配置哈希与门禁。全量检查可能需要数分钟。

`--configs-only` 只生成配置，不是数据验证；`--verify-only` 检查已经完成的门禁。配置生成器不覆盖内容不同的现有实验定义。

## 3. 执行训练

先运行三套基础模型：

```bash
mkdir -p artifacts/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py \
  --models mp20_u c2db51_u joint_u --device cuda \
  > artifacts/runs/main15_base.driver.log 2>&1 &
echo $!
```

基础模型完成并检查结果后，运行全部清单；`--resume` 跳过同一运行目录中已完成的模型，恢复已中断模型，未开始的模型从头训练：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py --resume --device cuda \
  > artifacts/runs/main15_remaining.driver.log 2>&1 &
echo $!
```

也可以首次直接运行完整 15 套：`python -u scripts/train_main15.py --device cuda`。**不要同时启动以上队列。**同 seed 的 runner.lock 防止重复队列，旧锁仅在确认对应父子训练进程均已结束后手动移除。

```bash
tail -F artifacts/runs/main15_base.driver.log
tail -F artifacts/runs/main15_seed42/mp20_u.log
watch -n 5 python -m json.tool artifacts/runs/main15_seed42/mp20_u/model.status.json
nvidia-smi
```

每套训练同时优化描述 Transformer 和流网络，另存 `model.pt`、`model.best.pt`、`model.history.jsonl`、`model.status.json`。不能将旧实验的检查点传给新配置恢复。`--seed` 用于独立训练重复；采样重复不等于独立训练种子。

## 4. 完整测试、无条件/性质采样

模型完成后，先检查评估计划：

```bash
python scripts/evaluate_main15.py \
  --checkpoint artifacts/runs/main15_seed42/joint_all/model.best.pt \
  --output artifacts/evaluations/main15/joint_all_seed142 --plan
```

正式执行：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_main15.py \
  --checkpoint artifacts/runs/main15_seed42/joint_all/model.best.pt \
  --count 4096 --conditional-count 1024 --steps 64 --guidance 2 --device cuda \
  --output artifacts/evaluations/main15/joint_all_seed142 \
  > artifacts/runs/joint_all_evaluation.log 2>&1 &
```

对每个训练来源执行完整测试损失、4096 次无性质采样；性质模型额外使用 TRAIN 的 10/50/90 分位目标分别采样。多性质联合目标使用实际 TRAIN 完整标签元组，避免拼接无数据支持的边缘目标。验证损失的条件输入随训练模型不同，不能单凭其数值直接排名生成质量。`--standard-metrics` 可追加现有 SMACT、指纹、COV、唯一/新颖和分布评估，需要 evaluation 可选依赖；只保存 PNG 图。没有独立性质预测器/DFT 时，属性命中率保持未知。

旧 `evaluate_joint.py` 的三来源域编号不适用于这批两来源模型，使用新入口。每次使用新输出目录，不能覆盖旧采样。正式指标保留所有尝试及失败；结构有效、精确群、性质和物理稳定性分别报告。

## 5. 化学条件：直接解码，不使用组成 CFG

以下 `MODEL` 指实际训练完成的检查点；联合模型必须额外加入 `--properties '{"source_domain":0}'`（MP20）或 `1`（C2DB），并选择相应 `--kind`。单来源模型默认无需 properties。

```bash
MODEL=artifacts/runs/main15_seed42/mp20_u/model.best.pt

# 恰好 Li-Fe-O，每种必须出现，计量比可变
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --chemical-system Li-Fe-O --count 256 --output artifacts/samples/main15/system

# 允许集合（不要求全部出现）与必含集合
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --allowed-elements Li-Fe-Mn-O --required-elements Li-O \
  --count 256 --output artifacts/samples/main15/required

# 所选模型标准胞内精确计数
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --composition Ti2O4 --count 256 --output artifacts/samples/main15/exact

# 经验式，均匀选择指定范围内可行的最简化学式单元数
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --formula TiO2 --formula-units 1 2 3 4 --group 136 \
  --count 256 --output artifacts/samples/main15/formula
```

集合要求、精确计数与群/轨道预算冲突时明确拒绝。模型不支持的元素也明确拒绝。主实验没有组成条件训练，采样时保持组成编码输入为空，组成只通过离散规则约束；不向网络注入未训练的组成特征。局部掩码解码不等同于从原模型的精确条件分布采样，必须评价实际质量。

## 6. 部分结构补全：固定晶格第一版

### A. 完整已知轨道作为前缀，生成未知轨道和坐标

```bash
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --group 136 --scaffold examples/my_partial.cif --composition Ti2O4 \
  --count 20 --output artifacts/samples/main15/scaffold
```

输入 CIF 只包含已知原子，必须已经处于指定群表的标准晶胞、轴和原点，并由完整轨道组成；程序不会从残缺骨架自动猜测群或静默调整坐标。轨道分解形成已知描述前缀，Transformer 生成其余轨道；固定晶格度量、已知轨道自由参数，对未知部分积分。允许未知部分原子数和轨道数变化，但受预算与可选组成规则限制。

### B. 任意部分原子：显式完整描述与对应索引

```bash
python -m unifiedgroupgen.cli sample --checkpoint "$MODEL" --device cuda \
  --descriptor examples/my_descriptor.json --scaffold examples/my_partial.cif \
  --atom-map examples/my_atom_map.json --count 20 \
  --output artifacts/samples/main15/completion
```

映射 JSON 示例 `{"0":0,"1":4}`：左侧是部分 CIF 中原子下标，右侧是完整描述展开后的原子下标。映射必须覆盖所有观测原子，元素一致，不重复。同一轨道使用共同周期展开分支，并检查观测相容性。固定一个原子可能同时确定整个对称轨道，不能声称只约束了该原子而其余同轨道原子仍任意变化。

二维使用 `--kind layer` 和层群编号，CIF 的周期平面必须为 xy、真空轴正交，群坐标的零高度对应 CIF 的 fractional z=0.5；面外高度转换为 Å，不周期包裹。固定的是晶格度量与群坐标中的骨架，CIF 导出仍使用规范取向和居中约定。任意取向/原点的实验片段须先明确转换，暂不支持自动碎片配准、非线性可变晶格骨架控制或电子密度控制。

这些主模型未接受观测补全训练，采样不使用观测特征编码，仅约束先验与速度并回缩。约束保持不代表补全分布准确。

## 7. 自动任务评估

```bash
python scripts/evaluate_tasks.py --checkpoint "$MODEL" --dataset mp_20 \
  --cases 32 --count 20 --steps 64 --device cuda \
  --output artifacts/evaluations/main15/mp20_u_tasks --plan

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_tasks.py \
  --checkpoint "$MODEL" --dataset mp_20 --cases 256 --count 20 --steps 64 --device cuda \
  --output artifacts/evaluations/main15/mp20_u_tasks \
  > artifacts/runs/mp20_u_tasks.log 2>&1 &
```

测试清单通过采样 seed 和 TEST ID 哈希固定，与模型权重无关；同源独立/联合使用相同测试案例。默认六项：`system formula required completion scaffold fixed`。其中 formula 是精确标准胞组成的 CSP，completion 明确给完整描述，scaffold 仅给群与已知首轨道；fixed 是固定完整描述的诊断任务。没有未知坐标自由度的骨架案例保留在 skipped 清单，不计为成功；首轨道掩码是当前可复现协议，不能外推所有掩码的效果。

保存逐任务命令、检查点/测试/生成哈希、所有尝试、失败原因、元素满足率、几何有效率、已知原子最大位移、Match@1、Match@K。骨架保持从生成状态和实际输入 CIF 重新计算，不仅相信生成时的布尔标记。Match@K 包括失败尝试，且化学体系和必含元素设计不以恢复唯一参考结构为唯一目标。补全任务缺失处的参考匹配与已知处保持分别判断，物理性质和稳定性保持未知。已完成采样可用完全相同参数加 `--postprocess-only` 重算任务指标。

通用化学/指纹/COV/VUN 可继续使用 `scripts/evaluate_generation.py --samples ... --dataset ... --reference-config configs/main15/<model>.yaml`，与本任务的精确约束指标分开报告；20 个条件候选不作为可靠整体分布覆盖率估计。

## 8. 本地验证的边界

需要通过：15 配置的完整输入门禁、群/轨道测试、精确计数与必含元素预算测试、三维/二维骨架保持、周期展开冲突、性质缺失和来源归一化、真实 CLI 与失败分母、原有测试。小网络初始化权重只验证程序与约束，不是正式训练或生成性能。正式模型训练和材料质量结论由服务器结果补齐。

2026-10-23 本地验证：完整测试 150 passed、18 skipped（CUDA/BF16 相关用例因本地不可用跳过）。MP20、C2DB 及联合全量门禁通过，划分与第 1 节一致。小型未训练模型实际执行六类任务评估入口成功，仅属软件诊断。真实修复数据的 TiO₂ `mp-2657`（SG136）和 MoS₂ `2MoS2-2`（LG63）固定骨架均保持，分别还剩 1 个未知连续自由度，观测位移与度量残差为数值零；这证明这两个输入的约束实现，没有评价未知部分的化学质量。
