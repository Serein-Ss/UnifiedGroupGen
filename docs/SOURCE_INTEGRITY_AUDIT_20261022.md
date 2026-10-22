# 2026-10-22 全量来源一致性审计与修订版训练

## 1. 结论

全部102,610条源记录均已核对，未用抽样代替全量检查。MP20旧编码48条不一致（2条组成、46条额外几何）；MPTS52为20条（1条组成、19条额外几何）；C2DB为0条。66条修复，2条MPTS52输入在既定公差下不能通过，保留在隔离清单，不进入正式训练。正式版本source_integrity_v1为102,608条：MP20 45,229、MPTS52 40,474、C2DB 16,905。

三个来源的原始CSV、原始编码目录、旧划分、检查点及历史生成输出均保留。新目录为artifacts/full_verified_v1与artifacts/joint_verified_v1。旧配置仍可用于解释历史检查点，正式训练门禁已禁止旧数据目录。

## 2. 核对内容与边界

1. 将元素计数约化为整数比例；允许合法换胞引起总原子数变化，不允许元素比例变化。
2. 将描述与独立状态用真实群表展开，比较每原子周期测度，禁止自由缩放。
3. 三维允许换胞、原点、刚体旋转与同元素置换；原胞约化用明确的小容差，避免比较器默认0.25 Å造成误判。
4. 二维按真实面内周期格和非周期高度比较；先解开真空边界并约化平移原胞，禁止辅助真空剪切、重复和轴交换，最后检查真实笛卡尔位移。
5. 建模验收公差分别为0.001／0.01／0.1 Å，另记录最大位移、每原子测度相对差及数值同一性标记。数值同一性使用位移<1e-4 Å、测度相对差<1e-5。空间群识别symprec不是标准化位移的数学上界；本次验收是显式工程预算。
6. C2DB参考为已通过官方来源恢复核验的CIF。最早有问题的CSV将笛卡尔数值写进分数坐标字段，不能将其按普通分数坐标CIF重新解释。

这些结果是给定公差内的来源结构一致性证据，不证明物理稳定性，也不是所有可能群／轨道组合的数学完备性证明。

## 3. 根因与代码修复

旧路径调用PyXtal.from_seed。其位点搜索失败时会尝试merge到另一种Wyckoff位点；这可能改变重数。两个已确认例子：mp-1281111由Li2CrO2变成Li2CrO3，mp-997160由AgMnO2变成Ag2MnO2。仅检查编译器内部的群闭合无法发现来源已经改变。

新路径先以spglib识别群，显式请求编译器指定Hall设置的完整标准胞，再在同元素原子之间执行逐轨道双射匹配，覆盖全部原子。转换末端强制检查组成与周期结构一致性。每条正式记录绑定源CIF SHA256和描述／状态SHA256；修改状态后认证失效。新转换版本为空间群3、层群8。

## 4. 隔离记录

- mp-621925：重新编码后最大匹配位移约0.0102018 Å。
- mp-1078353：重新编码后最大匹配位移约0.0104721 Å。

二者均超过MPTS52的0.01 Å验收预算。它们不是凭此被判为无效材料，只是当前固定规则下未通过本次严格来源验收。没有逐条放宽容差或静默降到P1。完整源数据保留；quarantine.jsonl记录旧编码及原因。

## 5. 对历史模型的影响

| 模型 | 旧TRAIN受影响 | 旧VAL受影响 | 旧TEST受影响 | 正式处理 |
|---|---:|---:|---:|---|
| M1 MP20 | 26 | 11 | 11 | 修正数据从头重训 |
| M2 MPTS52 | 14 | 0 | 6 | 修正数据从头重训 |
| M3 三来源联合 | 58 | 4 | 6 | 修正数据从头重训 |

M1/M2/M3训练轮次和权重仍是历史事实。它们可以用于架构、采样流程与失效分析，但不再作为修订数据上的正式性能。直接评价历史生成结构的群、距离、SMACT数值仍描述那些候选；依赖旧参考分布的COV、唯一性／新颖性和VUN须在新实验中重新评价。受影响记录虽少，不能据此断言统计影响可忽略。

修订后的联合分组为82,747个组件，TRAIN／VAL／TEST为82,183／10,190／10,235。源间关联仍按材料ID、相同CIF与核验后的图指纹候选保持同分区；这不是穷尽近重复或原型隔离。

## 6. 证据与复现

- assets/source_integrity/summary.json：公差、分母、修复与隔离、历史模型影响。
- assets/source_integrity/all_records.csv.gz：全部102,610条逐记录检查。
- assets/source_integrity/affected_records.csv：68条处理记录。
- assets/source_integrity/01_full_source_audit.png：审计、处理去向及模型影响。
- assets/datasets：基于修订数据重新绘制的PNG与源表。
- scripts/audit_source_integrity.py：逐条检查，--repair只写新目录。
- scripts/finalize_verified_data.py：固化新记录、生成配置、重分组并执行训练前门禁。

复现新审计时使用一个新的--output，避免覆盖证据。现有审计报告保留，不把试运行当作正式结果。

## 7. 服务器更新：现有服务器已有旧完整数据包

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python scripts/unpack_training_data.py --bundle datasets/source_integrity_v1
python scripts/verify_training_data.py
python -m pytest -q
```

全新服务器先运行一次默认的python scripts/unpack_training_data.py导入原始数据与C2DB恢复证据，再导入上述追加数据包。旧包不是修正版训练入口；追加包不覆盖旧运行或旧权重。无需安装另一份PyTorch，也不要使用旧模型的--resume。

## 8. 修订数据的从头训练入口

先验证，再按GPU预算顺序启动；以下命令不会复用旧权重：

```bash
mkdir -p artifacts/runs/source_integrity_v1
CUDA_VISIBLE_DEVICES=0 nohup bash -c '
set -e
python -u -m unifiedgroupgen.cli --seed 42 train --config configs/mp20_verified_v1.yaml --device cuda --output artifacts/runs/source_integrity_v1/mp20/model.pt --validation-interval 5
python -u -m unifiedgroupgen.cli --seed 42 train --config configs/mpts52_verified_v1.yaml --device cuda --output artifacts/runs/source_integrity_v1/mpts52/model.pt --validation-interval 5
python -u -m unifiedgroupgen.cli --seed 42 train --config configs/joint_verified_v1.yaml --device cuda --output artifacts/runs/source_integrity_v1/joint/model.pt --validation-interval 5
' > artifacts/runs/source_integrity_v1/driver.log 2>&1 &
echo $!
tail -F artifacts/runs/source_integrity_v1/driver.log
```

仍是1000轮上限及早停策略，不承诺实际轮数或耗时。当前没有训练source_integrity_v1的新模型。本地仅执行数据修复、验证与绘图。
