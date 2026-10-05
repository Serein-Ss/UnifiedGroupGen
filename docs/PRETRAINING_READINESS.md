# 正式训练前准备与验证

日期：2026-10-05。MP20 large 正在完整数据训练，MPTS52 large 顺序排队。C2DB全部16905条来源恢复、v7完整转换与审计已通过，零拒绝；联合划分保留全部102610条记录，修复后的完整联合审计及实际训练入口核查均已通过。共享模型训练与最终评测尚未完成。

## 1. 已补齐的训练基础

| 内容 | 实现与检查 |
|---|---|
| 真正批处理 | 描述序列 padding + causal attention；几何网络打包为互不连边的多个图；与单结构计算的输出和梯度对照 |
| 图内存预算 | 按展开原子数分桶，限制 `ΣN(N−1)`；超限单结构明确拒绝，不切碎对称轨道 |
| 大模型 | medium/large 六个数据集配置；激活重计算、边消息分块、BF16/FP16；416 原子反向压力测试 |
| 训练诊断 | descriptor、flow、lattice、coordinates 分项损失；梯度裁剪、有限性检查、LR、时间与峰值显存 |
| 保存与续训 | AdamW、ReduceLROnPlateau、last/best 权重、早停；完整 RNG/优化器/AMP 状态；输入 SHA256 和缩放后配置匹配 |
| 可复现验证 | 固定验证随机源，并恢复训练随机源；CPU 连续两轮与一轮+续训逐参数一致 |
| 性质与组成 | 标量/类别编码、缺失标记、全性质 CFG dropout；缩放只使用训练集；性质 CFG 保留硬组成/观测条件 |
| 元素域 | 正式配置从训练集确定元素域；不指定组成时也限制生成元素；违反域的组成明确拒绝 |
| 符号合法性 | 可完成前缀掩码、固定位置占据约束；搜索缓存、最大重数下界与 gcd 必要条件剪枝 |
| 转换与恢复 | 原始 CSV 只读；进度 journal、源哈希、转换配置 manifest；多进程有界提交/超时；尾部中断写入可恢复 |
| 描述顺序 | 按 `(自由度, Wyckoff 字母, 元素, 源序号)` 同步重排轨道和连续参数；不将重复轨道类型去重 |
| 数据入口检查 | ID/CIF 跨划分重合、严格图指纹候选经 StructureMatcher 确认、非有限状态/性质、维数、晶格、预算、来源和覆盖率 |
| 正式训练门槛 | 必须完整源转换、覆盖率≥95%、审计通过、配置/缓存哈希一致；小缓存不能冒充全数据 |
| 共享模型门槛 | 各来源全量审计 + 材料分组划分 + 记录库存 SHA256；保留所有输入，不从旧单库检查点初始化 |
| 计算来源与 CFG | source_domain 为必填类别上下文；CFG 只丢弃性质，保留来源；拒绝给没有训练标签的来源指定性质 |
| 采样统计 | 保存全部成功/失败尝试；几何有效率以请求总数作分母；层群精确识别未知与失败分别报告 |
| 完整测试绑定 | 正式评测只接受已审计的全部测试文件，检查点训练/验证哈希匹配；按来源报告损失、标签覆盖和未支持分母 |
| 原生识别隔离 | 采样/审计 CLI 的 SG/LG 识别在独立进程执行，原生终止或20秒超时记为未知，后续样本继续 |

`minimum_conversion_coverage=0.95` 是预先列出的覆盖率门槛，不保证剩余数据无选择偏差。出现拒绝仍须检查原因及群/元素分布，不能据此将删除后的结果称为原始全基准。

## 2. 本次验证

- 最新 **80 项测试通过**，保留一条 pymatgen 在敏感三维 CIF 中进行坐标舍入的预期警告；cgdit 单环境 JUnit 证据在 `artifacts/pretrain_validation/pytest_cgdit_joint.xml`。新增测试包含无层群标签识别、来源上下文 CFG、二维/三维联合反向、跨来源材料分组、原始 ASE 精确恢复、全量来源/测试门槛、按来源损失与原生识别异常隔离，以及 CIF 原始坐标保存和计算优化的损失/梯度/RNG 一致性。
- 单元/回归测试覆盖所有 230 SG、80 LG、17 PG 的度量与轨道表、代表性群的零空间等价、完整合法路径、观测纤维、批处理、掩码、续训和 CUDA AMP。
- medium：64 train / 32 val 的真实 MP20 两轮短训练通过，验证总损失 44.9764 → 33.3393；这是有限性和流程检查，不是性能结果。
- 小子集缺少验证元素，因此该短程诊断使用预定义的 1–118 词表；正式配置采用训练集元素域。诊断采样仍可能输出罕见或不合理元素。
- 固定 SG2 描述、固定全部晶格参数生成 2/2，通过操作闭合，六个晶格参数保持为 0；最小距离约 0.850 / 1.376 Å，仅通过 0.6 Å 诊断阈值。
- 性质条件未知描述生成 4/4，全部操作闭合，3/4 回收精确目标 SG，2/4 通过距离阈值；没有化学、稳定性或性质命中认证。
- medium/large 的真实 batch 与 large 的 416 原子压力测试见 [容量审计](MODEL_SCALE.md)。
- 已接入 `spglib.get_layergroup`，明确非周期方向为 z，重新识别容差 0.001 Å；对 80 个 LG 各构造两个不同元素的一般位置轨道，80/80 成功识别且精确回收目标。证据为 `layer_identifier_80.json`。这验证合法合成结构的识别链路，不表示真实含噪数据或所有生成样本都能精确回收目标；识别失败计为未知。二维群包含率与精确回收率分开。

验证产物：`artifacts/pretrain_validation/`。早期诊断环境保存在 `environment.json` 和可选 `constraints-tested.txt`。正式运行已在现有 cgdit 单环境验证：torch 2.7.1、numpy 2.2.6、pyxtal 1.1.1、spglib 2.6.0，实际 CUDA 矩阵乘法及反向传播成功。启动前设置 `MKL_THREADING_LAYER=SEQUENTIAL`、`OMP_NUM_THREADS=4`、`MKL_NUM_THREADS=4`，消除重复 OpenMP 初始化冲突；没有采用 `KMP_DUPLICATE_LIB_OK`。这仍不是全新安装环境复现。

当前环境完整元数据保存在 `environment_cgdit_snapshot.json`。Python torch.version.cuda 标签为空，但实际原生构建报告 CUDA11.8/TORCH_VERSION2.7.0、Python版本标签2.7.1，GPU训练确实运行；不能据标签推断为CPU训练。既有 pandas-stubs 缺少 types-pytz（类型标注用途），且 scipy dist-info 缺少Name/Version元数据；这些问题单独记录，不修改正在运行的既有环境，也不将 pip check 称为通过。

正式完整测试入口会核对训练/验证检查点哈希及所有测试文件路径/哈希，拒绝子集或修改后的标签。已经对真实 MP20 best 检查点验证该门槛，证据 `heldout_gate_actual_mp20.json`；没有提前计算尚未收敛模型的测试损失。共享测试报告将列出每个来源的样本数、元素支持覆盖、实际性质标签数与分项损失。

在真实敏感输入 `mp-568324`、0.001 Å 下，隔离识别器实测捕获原生退出码3221226505；随后健康结构正常识别SG2，整个检查约0.97秒。证据 `native_identifier_sensitive.json`。该失败记为未知而非猜测SG1；训练数据的统一0.01 Å协议不改变。隔离可防止评测进程被原生崩溃终止，不证明所有样本可识别。

## 3. 全数据状态与识别约定

### MP20

三个划分全部读取并转换，train / val / test 为 **27,136 / 9,047 / 9,046**，无转换拒绝。中/大配置审计已经通过。训练标签缩放不读取 test。

test 的 `mp-23155`、`mp-568145` 含训练集中未出现的 Ar，作为分布外样本记录，并保留在后续评价分母中。当前指纹与匹配检查没有发现跨划分重复；这不是穷尽的近重复或原型不重叠认证。

### MPTS52

原 0.001 Å 转换在 train row3111 `mp-568324` 触发本机 spglib 原生进程终止，2.6.0 与隔离的 2.7.0 测试均可重现；没有采用 P1 回退。该结构在 0.002 / 0.01 / 0.1 Å 稳定识别 SG140，0.0005 Å 得到 SG42，0.0001 Å 得到 SG1，表明其识别对容差敏感。

正式 medium/large MPTS52 配置统一采用 **0.01 Å**，使用独立目录 `artifacts/full_tol0p01/mpts_52` 从头转换。旧失败 journal 留在 `artifacts/full/mpts_52`，不混入新训练缓存。后续比较必须统一各方法的识别与评价容差；原 CGDiT 数据配置为 0.1 Å，不能直接认为旧缓存与新标签等价。建议列出 0.01/0.1 Å 敏感性实验。

三个划分全部完成，train / val / test 为 **27,380 / 5,000 / 8,096**，无转换拒绝；medium/large 全量审计均通过。最大展开原子数为 208（MP20 为 80），配置上限 416 为保守预算上限，并非数据集所有结构都具有 416 个原子。结果保存在该目录的 `audit_medium.json` 和 `audit_large.json`；不得使用旧 8 条小缓存报告正式就绪。

### C2DB

原 CSV 把 Cartesian Å 坐标写入 fractional 字段，且只保存晶胞长度角度。旧诊断配置依赖朝向推断，train64 接受47/拒绝17、val32 接受20/拒绝12；这些历史结果不用于正式训练。

正式配置改为 `c2db_official_local.yaml`。`recover_c2db.py` 从官方 UID JSON 恢复完整 cell/positions/numbers/PBC/energy，逐条匹配原 CSV，核对阈值 1e-6；输出独立 CSV，原文件只读。原始全量 train/val/test 为 **10,143 / 3,381 / 3,381**。配置要求恢复报告、来源/输出哈希和每条记录全部通过，转换覆盖率为 **1.0**；`source_verified: true` 本身不能替代这些证据。

原数据有 **94** 条 lgnum=-1（52/21/21）。来源恢复后由 `spglib.get_layergroup` 重新识别并标准化，再提取完整 Wyckoff 轨道；识别失败明确拒绝，不给缺失标签赋 LG1。已有标签与重新识别标签均保存。layer converter version=6，正式缓存新建在 `artifacts/full_official_tol0p1/c2db_51`，不能复用旧层群缓存。

已完成 **512** 条来源核对后的转换诊断：在固定 0.01、0.05、0.1 Å 下都接受512条，511条已有标签分别匹配439、492、509条，另一条缺失标签由原生识别器赋值。正式统一采用 **0.1 Å**，与前期 `conf/data/c2db_51.yaml` 的协议一致；这不是逐行放宽容差。需在最终结果报告统一容差和敏感性。

修复解析器后，version6 使用原始 fractional 坐标重跑固定0.1 Å的512条诊断，全部接受，511条已有标签匹配509条，与version5无群标签变化；缺失标签1条成功识别。标准坐标 RMS 理想化偏差中位数约2.94e-5 Å、p95约0.01014 Å、最大0.04570 Å；最大单原子位移最高0.09815 Å。二维度量相对变化 p95约0.001267、最大0.01269。坐标偏差在固定标准晶胞中计算，**没有包含晶格应变对原始 Cartesian 位置的贡献**，度量变化单独报告。当前证据为 `c2db_raw_parser_diagnostic.json`；旧多容差诊断保留在 `c2db_tolerance_diagnostic.json`。512条不等同全数据通过。

原始导出中 `2IIrS2-1` 的 Cartesian y=0.66667968 被 pymatgen 的 fractional 近分数舍入改成2/3，造成约1.3e-5 Å的假来源不匹配。现在对该错误字段模式及恢复后的二维原始 fractional 输入关闭解析器舍入，再让显式对称识别执行标准化。该条实测来源坐标误差6.285e-9 Å，通过原有1e-6 Å阈值，未放宽核对。运行中的旧恢复进程已加载旧解析器，其拒绝计数暂时仍含该条；后续缓存重试使用新解析器。原始CSV保持只读。

全量恢复仍在运行。目前确认 train row1685 的 `4Nb2S3-2` 在官网返回 `No such material!`，UID 查询计数0，旧 unique_id 跳转失败；公开 NIST/JARVIS 的旧 C2DB 副本也没有找到。后来在公开MLIP Arena的16905条历史ASE档案中找到 exact UID/unique_id，数据库row14015完整通过原有核对阈值，Cartesian残差5.00e-8 Å；档案SHA256与发布的LFS哈希一致。已用该档案重启全来源核对，缺失来源问题已解决，仍须等待全部转换与审计。新增 `--ase-db` 入口只按 exact UID/unique_id 查找并核对全部字段，保存数据库哈希和行号；后续来源检查会检测原数据库变化。其余记录继续恢复；不能删除这条记录后宣称完成原始全量数据。

## 当前本机正式作业

`configs/mp20_local.yaml` / `mpts52_local.yaml` 保持 large 的所有网络参数，将 batch 改为 64；图边预算仍为 200,000。对应完整审计 `audit_local.json` 均通过。最大 1000 epochs，验证损失连续 50 轮未改善时早停；没有设置小数据或少轮数替代正式训练。

`scripts/train_local.py` 使用当前 Python 环境顺序运行 MP20 训练 → 全测试集损失 → 无条件/性质条件各 64 次生成 → MPTS52 同样流程。进度与子进程 PID 存在 `artifacts/runs/local_large_seed42/runner.json`，逐批日志为 `mp20_train.stdout.log`，每轮原子保存 last/best 检查点。模型 `*.status.json` 只在最大轮数或早停正常结束后标记 complete；训练结束不表示物理认证完成。测试损失只对词表支持的样本定义，Ar 两条仍保留在总测试集分母并明确列为未支持。

MP20 首个完整训练+验证 epoch 已完成：793.22 s，验证损失 27.8097，峰值 allocated 2.43 GiB，last/best 已保存，训练继续。检查点可读取且两网络全部 tensor 有限，记录在 `artifacts/pretrain_validation/first_formal_epoch.json`。这验证完整数据一轮的执行与保存，不是完整收敛或材料质量证明。

本轮检查已完成 epoch0–11 十二轮，前八轮含验证约793–798 s，优化后四轮752.56 / 747.79 / 751.17 / 773.53 s；最新验证损失23.2686，最佳22.9396；逐轮日志是最新事实来源。不得重复启动仍活跃的 runner。

耗时分析发现同一批数据重复展开晶格，以及先验读取GPU布尔掩码数量产生同步。现已在批量流损失中复用网络展开的晶格，并从编译器的CPU掩码读取固定数量；模型参数、先验、损失和约束不变。CPU/CUDA测试确认随机样本与RNG状态完全一致，物理损失完全一致，状态/参数梯度在1e-8内一致。5组交替BF16完整优化步计时中位数从1.201 s降至0.784 s，部分组反向波动；因与正式训练共享GPU，仅为局部计时，不推断整轮提速。证据 `profile_large_mp20.json`、`performance_paired_mp20.json`。已在epoch7结束的完整检查点处完成一次受控续训，MP20当前也加载新实现；后续MPTS52和共享训练阶段同样加载新实现。

续训前确认已有 runner 和训练子进程已退出，再运行：

```powershell
python scripts/train_local.py --resume
```

runner 锁阻止重复启动；异常断电留下旧锁时，确认记录的 PID 已退出后再清理该锁。不要仅因终端等待超时重复启动。

## 共享二维/三维主模型

`joint_space_layer_large.yaml` 注册 **18,030,806** 参数，使用同一描述 Transformer 与几何流网络。MP20/MPTS52 含 **19,819** 个共有材料ID，其中 **9,985** 个在原划分中跨 train/val/test；共享训练前必须重划分。

`joint.py` 将同源材料ID、相同源CIF和经 StructureMatcher 验证的指纹候选组成连通分量，以种子42按组件分配约80/10/10，保证一个材料的所有记录处于同一划分。训练元素覆盖不足时移动整个组件并记录原因，只使用身份/组成，不读取目标性质。保留每个来源记录及其标签，不平均或合并标签；全库存 SHA256 检测掉行、重复行、坐标或标签修改。预计输入总数102,610，实际划分计数待来源全量通过后生成。

三个来源使用 `source_domain=0/1/2`，由配置映射并仅在加载时附加；原始记录不改写。此上下文区分计算方法/来源，不能据此认为各库形成能已经物理校准。缩放使用新共享训练集；缺失性质仍为缺失。无性质生成也指定来源：如 `--properties '{"source_domain":2}' --kind layer`。条件生成需该来源有相应训练标签。

旧单库检查点可能训练过新共享测试集的材料，**共享主模型从头训练**。当前单库原划分实验不能作为共享新划分下的公平基线；正式比较需让基线使用同一新划分。

`scripts/complete_pipeline.py` 依次等待来源恢复、全量二维转换/审计、共享划分/审计、现有三维作业完成，再进行共享完整训练、完整测试集损失和各来源无条件/条件各64次生成。等待时持有真实 Windows 进程句柄，避免 PID 复用；源恢复退出2只允许进入缓存重试，重试仍有拒绝则停止，不放行训练。恢复已有失败来源尝试时，可使用 `--resume --ase-db <原数据库路径>`，其参数与实时 PID 见运行状态文件。

这些脚本只编排训练与当前结构检查；64次采样用于完整流程复验，正式论文仍需要更大的采样预算与外部物性认证。

本机编排器已启动，状态在 `artifacts/runs/full_pipeline/pipeline.json`。它当前等待 C2DB 全量恢复，没有新增并行 GPU 训练。若原始数据库在下载作业结束前找到，可只补齐缺失UID的缓存并保持 provenance；不要同时向同一恢复目录启动第二个全量导出作业。现有下载结束后，编排器会重新验证缓存；任何拒绝都将阻断共享准备。

## 4. 正式操作命令

从 UnifiedGroupGen 目录，在兼容环境中安装并先跑测试：

```powershell
python -m pip install -e ".[test]"
python -m pytest -q
```

全量转换、审计（这些命令不会训练）：

```powershell
python scripts/prepare_training.py --config configs/mp20_large.yaml --workers 4 --resume
python scripts/prepare_training.py --config configs/mpts52_large.yaml --workers 4 --resume
```

已完成转换时 `--resume` 复用 journal 并重新审计；源文件/转换配置变动须另建缓存，不能混用。不同机器不要同时写同一 journal。

审计通过后，正式主配置和续训分别使用：

```powershell
python -m unifiedgroupgen.cli train --config configs/mp20_large.yaml --device cuda --output artifacts/runs/mp20_large_seed42/model.pt
python -m unifiedgroupgen.cli train --config configs/mpts52_large.yaml --device cuda --output artifacts/runs/mpts52_large_seed42/model.pt
python -m unifiedgroupgen.cli train --config configs/mp20_large.yaml --device cuda --resume artifacts/runs/mp20_large_seed42/model.pt --output artifacts/runs/mp20_large_seed42/model.pt
```

上列单数据集命令作为可复现入口保留；本机实际使用前述 local runner。`--epochs` 是总轮数，不是额外轮数；新训练不能覆盖已有检查点。早停根据 descriptor+flow 验证损失，尚不代表材料有效率最优；最终性能评测仍需要独立生成与外部验证。

切换 medium 做容量/成本对照时需要用 medium 配置重新 preflight，因为审计绑定完整配置；可复用同一份转换缓存。

## 5. 正式研究尚需完成的实验

1. 至少三个种子；相同数据、元素域、常规/原胞约定、容差、采样数量、优化预算与 Relaxation 流程比较。
2. small/medium/large 容量消融；独立参数与投影实现；训练路径/先验合法性消融；前端是否学习群、是否学习元素的消融。
3. 无性质、单性质、多性质、固定组成、固定描述、固定晶格/部分坐标分别测试；组成命中与性质命中分开。
4. 指定群包含率、精确回收 SG/LG、轨道退化、碰撞、组成/化学有效、唯一性、训练集新颖性、多样性；失败均记录。
5. 性质预测器、MLIP Relaxation、DFT/凸包/声子用于结果认证，校准其适用域并统计总计算成本。
6. C2DB 来源确认及数据就绪后再开展二维主实验、跨 SG/LG 的共享模型实验。当前没有训练磁性数据或实现任意非线性性质的严格等值面生成。

模型输入性质是条件，CFG 是分布控制机制；两者都不能保证物理性质严格等于目标。编译器保证包含指定群的合法运动，但不能保证结构恰好只有该群、无碰撞或稳定。

## CUDA 续训修复与优化部署记录

2026-10-05 04:09，备份epoch7完整检查点后受控退出原MP20子进程，原runner按预期记录4294967295退出；C2DB恢复未中断。第一次续训暴露CUDA RNG张量被map_location迁移到GPU，而恢复接口要求CPU ByteTensor，触发TypeError，未写入新检查点。现在恢复前逐个将CUDA RNG状态转回CPU。CPU FP32、CUDA FP32、CUDA BF16三种连续两轮与一轮+续训对照均逐参数一致，当时完整77项测试通过。

再次续训成功，runner PID1785308、MP20 PID1780596、编排器PID1786704；C2DB仍为PID1741264。新runner从epoch8开始，完整27136/9047记录、BF16、batch64、最大1000轮及全部训练设置一致。首次25步损失80.6610107421875，与受控退出前同一批次相同；这只验证对应日志点，并非整个大模型训练轨迹严格相同的证明。证据与检查点哈希在 `artifacts/pretrain_validation/optimization_resume_epoch7/deployment.json`，检查点快照在同目录。旧失败日志保留，判断当前状态须读取最新runner状态和实际PID。

后续恢复到train row5182时，旧进程又拒绝 `1CCr2N3Ta3Br4-1`：旧解析残差4.037e-5 Å，修复后7.073e-9 Å。只读复核原始行5175–5199的25条缓存，新解析全部通过，阈值保持1e-6 Å。证据 `c2db_rejection_5175_5200.json`。当前旧进程的3条拒绝包含两条已复核的解析误判和公开缺失的 `4Nb2S3-2`，完整来源核对仍待后续缓存重试。

## 优化后完整一轮检查

epoch8已完成27136条训练和9047条验证，完整耗时752.56 s；此前八轮中位数796.33 s。单轮观察耗时减少5.50%，其中训练425批574.38 s，验证与其余轮内工作约178.19 s。不是严格重复测速，数据顺序和系统负载会影响时间。新检查点、AdamW全部tensor有限，配置与训练/验证源哈希与epoch7快照一致，CUDA RNG正常保存。证据 `artifacts/pretrain_validation/optimized_full_epoch.json`。本轮验证23.7393、最佳23.5295；训练继续，没有收敛或材料质量认证。

train row5555 `2ClPPbO3-1`也受到旧解析器舍入影响：旧来源残差4.308e-6 Å，修复后6.904e-9 Å。行5550–5574共25条缓存只读复核全部通过1e-6 Å阈值，证据 `c2db_rejection_5550_5575.json`。截至该快照，旧恢复进程4条拒绝中有3条经新解析器通过，另1条仍为公开缺失的 `4Nb2S3-2`；等待完整缓存重试，未将局部复核当成全来源通过。

## 公开历史档案恢复

找到 [MLIP Arena C2DB公开档案](https://huggingface.co/spaces/atomind/mlip-arena/blob/1508879571257075528db9522e845605256f6b74/benchmarks/c2db/c2db.db)，完整16905条，文件70762496 bytes，SHA256 `caf58205692de480e06149ac43a437385f18e14582e7d9a8dab8b3cb5d4bd678`。缺失UID `4Nb2S3-2` exact UID/unique_id对应row14015；species/counts、Cartesian位置、晶胞形状、PBC、energy全部通过既定1e-6阈值，坐标残差5.00e-8 Å。没有替换元素或标签，没有删除原始行。来源证据保存在 `c2db_mlip_arena_source_check.json` 与归档目录的source.json。

已受控退出旧串行网络恢复及等待编排器，原MP20 runner/训练进程保持不变。新恢复PID1795580使用已有官方缓存和 `--ase-db artifacts/source_recovery/c2db_alternatives/mlip_arena_c2db.db`，新编排器PID1799580仍持有实际进程句柄。首个启动尝试因未设置PYTHONPATH在导入阶段退出，已修正启动环境，未处理数据；旧错误日志保留。新恢复前6600条重新核对零拒绝，不能据此前缀称全来源通过。实时日志 `recovery_archive_retry.stdout.log`。公开档案路径是后续审计来源，应保留完整文件和SHA；此前公开站点404/缺失及旧解析误判仍作为历史证据。

C2DB train完整来源恢复已通过：10143/10143，零拒绝，train.recovery.json的源/恢复CSV SHA256重新核对匹配。所有UID/unique_id/formula/PBC及顺序保留，lgnum/natoms/energy/hform/gap/ehull/thickness在1e-12浮点舍入尺度内相同，52条缺失层群标签保留。缺失UID已包含，未删除行。证据 `c2db_full_train_source_verified.json`；val/test和完整native转换/审计仍待完成，不能仅凭train报告放行联合训练。

公开ASE归档全库存只读核查：train/val/test共16905条全部按UID和unique_id一一对应，原子数、PBC一致，原lgnum标签一致（含52/21/21条未知标签）。原CSV的非缺失数值标签与归档完整精度值最大绝对差4.9977e-8，通过现有来源核验使用的1e-6绝对阈值；初始1e-10诊断低于CSV数值精度，其差异计数保留于报告。恢复输出仍复制原CSV标签，不能将归档精度差异当成标签替换。归档SHA256复查未改变。证据 `artifacts/pretrain_validation/c2db_full_archive_inventory.json`。此库存检查不代替逐条晶胞/Cartesian匹配、恢复输出哈希核对或完整层群转换审计。

C2DB val完整来源恢复及输出核对通过：3381/3381、零拒绝，原/恢复CSV SHA256匹配，原始行顺序及全部非CIF字段保留，所有数值标签最大差为0，21条未知层群标签保留。来源坐标匹配最大残差6.4031e-8 Å。证据 `artifacts/pretrain_validation/c2db_full_val_source_verified.json`。test恢复正在进行；尚未生成全分割通过报告，完整native层群转换/审计和联合训练仍未完成。MP20已完成14轮完整训练/验证，第14轮验证损失22.3923更新最佳值；未认证收敛。

## 全来源通过与非理想标准坐标修复（2026-10-05 06:00）

C2DB来源恢复进程1795580正常退出0；train/val/test共16905条、零拒绝。独立复核全部原始非CIF字段、行顺序、原/恢复CSV SHA256、16905个来源缓存SHA256及原ASE数据库SHA256通过，所有数值标签差为0。逐条来源匹配最大Cartesian残差train/val/test分别为7.1213e-8 / 6.4031e-8 / 7.0711e-8 Å。证据 `artifacts/pretrain_validation/c2db_full_source_verified.json`。原始CSV与公开ASE归档保持只读。

首次全量layer converter v6转换中，val/test全部通过，train唯一拒绝为row3268 `2S2Cu3-2`。spglib识别LG72，却返回仍有0.020049 Å群操作残差的standard positions，不能直接当作严格闭合的坐标。converter v7首先保留严格匹配；失败时对同一个已识别群在配置的统一0.1 Å物理容差内拟合Wyckoff轨道，并检查元素保持的双射拟合位移与编译后严格群置换。此分支既不改群号，也不放宽配置或忽略闭合错误。拟合位移、原生标准化位移与晶格变化分别保留，`std_position_total_max_upper_bound_angstrom`仅是固定标准晶胞下位移的保守上界，不包含晶格应变。

真实失败记录修复后仍为LG72，独立识别精确回收LG72，源10原子的非原胞表示转为5原子的标准胞，整条数据保留；额外chart fit最大位移0.011221 Å，源标准化加拟合上界0.065611 Å。超过配置容差的扰动会拒绝。fixture与两项回归测试已加入，完整cgdit测试79项通过（旧77项结果仍保留为历史证据）。诊断/修复证据 `c2db_layer72_rejection.json` / `c2db_layer72_repaired.json`。

旧v6失败数据与audit保留，正式配置切到 `artifacts/full_official_tol0p1_v7/c2db_51`，重新转换全部记录；联合模型转换覆盖率门槛也固定为1.0。旧编排器已退出，新的编排器PID1825376、转换PID1831860，复用已通过的全来源报告，MP20训练不重启。后续完整v7 audit、联合分组划分与全模型训练仍需实际通过，不能用来源核验或单条修复替代。

v7全量转换及audit已实际通过：train10143、val3381、test3381，零拒绝、零审计错误，转换进程1831860正常退出0。审计SHA256 `2b48194ba6e16bfbded96c0a0731e46747a6860058253469305d369005a51584`。编排器进入联合分组进程1823444，全部102610条输入的联合划分和audit尚在运行。

全量二维理想化诊断 `artifacts/pretrain_validation/c2db_full_native_conversion_diagnostic.json`：16811条原有效层群标签中16786条匹配，25条在统一0.1 Å识别容差下改变；94条原缺失标签均识别，原标签保留于来源及provenance。额外chart fit仅1条。原生固定标准晶胞坐标位移最大值的median/p95/max为0.000126 / 0.019271 / 0.112079 Å；晶格度量相对变化p95为0.000883、max为0.041290。识别容差不是对最终位移或晶格应变的严格上界，不能称所有原始坐标完全保留，也不能由data audit证明材料稳定性。

## 联合候选记录查找修复（2026-10-05 06:18）

联合分组保留102610条输入，组件82748，train/val/test为82185/10190/10235；原分组进程1823444自行正常退出0，旧audit报告passed=true。随后复现审计查找错误：两个数据库共享同一材料ID但几何不同，第三条复制第二条几何的验证记录会被旧代码拿第一条几何比较，导致漏检。chart候选现在直接保存先前记录的引用，既保持来源身份，又消除每次查找扫描整个分割。回归测试修复前得到0条预期重复错误，修复后正确得到1条；完整cgdit测试80项通过。

旧报告保存在 `artifacts/pretrain_validation/joint_audit_lookup_handover/audit_legacy.json`，未经修复后审计不能作为共享训练放行依据。联合划分及数据文件没有改写，MP20训练不重启；旧编排器1825376已进入等待三维runner1785308阶段。独立修复后全量审计PID1827736运行，输出目标仍为 `artifacts/full_joint_sourcegrouped/audit.json`，重新通过前该文件passed=false并记录reaudit_required，共享训练入口会拒绝。历史运行、查找修复和新审计状态在handover.json；不能根据旧joint_prepare日志中的通过结论判断新审计完成。

修复后的全量审计进程1827736已终止，正式audit实际passed=true、零errors；随后实际调用check_training_gate通过，验证全部父配置与数据、联合清单与输入、102610条库存恰好保留一次。train/val/test为82185/10190/10235，组件82748。记录库存SHA256 `1d30e1a44a5e26b638506a5218e87254ec7966b246416e8c6a19ddf020d70ce2`，修复后audit SHA256 `ca364e2b13a07695ea07f361247f5bdf2686bfdb9ec7f372660ca1d5aca7bedd`，manifest SHA256 `e642c3ba02c075d8a120b0894d0c6cc6c442a612d389b0a0dd248e4ec7158eb9`。证据 `artifacts/pretrain_validation/joint_full_training_gate_verified.json`。全部数据准备及训练入口已通过，共享模型尚未启动，编排器1825376继续等待实际三维runner1785308完成训练和评价；这不是完整训练、物理性能或穷尽原型去重的完成证明。

## 整数索引优化验证（2026-10-05 06:43）

当前配置单步剖析显示3050次CUDA stream synchronization、576次nonzero；约33% CPU self time在同步等待。随机先验现在使用CPU确定的周期参数整数索引，原子对直接按原顺序构造所有不同原子的有向边，批量repeat_interleave提供已知输出长度。没有改变参数量、随机数调用顺序、边语义、物理损失或训练配置。

完整测试84项通过，一条既有CIF有限精度警告，证据 `artifacts/pretrain_validation/pytest_cgdit_indexing.xml`。新增覆盖无周期自由度先验，以及单原子/空间群/层群原子对的完整性和顺序。独立进程加载优化前源快照，比较空间/层/平面群及特殊位置的输出、状态梯度和模型梯度：CPU FP32、CUDA BF16完全一致，CUDA FP32最大梯度差8.94e-8；原实现自身重复的最大梯度差4.77e-7。全部通过rtol=1e-6、atol=1e-7，不能宣称CUDA FP32逐bit确定。

7组交替完整BF16优化步中位数为原实现0.910294 s、优化后0.830884 s，观察下降8.72%，其中一组反向波动。证据 `artifacts/pretrain_validation/indexing_performance.json`。同GPU同时运行正式训练，此值不代表整轮加速。源码优化保留，当前MP20进程1780596未重启，仍使用此前已加载代码；随后新启动的MPTS52及共享训练进程将加载整数索引版本。完整数据、训练上限、早停、优化器与RNG续训规则保持原配置。正式训练仍未完成。
