# 本次实际验证记录

2026-10-05 验证频率与日志更新：默认第5、10、15……epoch及最后一个epoch做完整验证，其他轮不计算验证损失；每轮仍保存最新检查点。TQDM用于交互式终端，重定向日志每轮一行摘要。CPU/CUDA BF16测试确认在第3轮中断后续训与连续运行权重一致，验证轮数不偏移；早停/学习率耐心仍按训练epoch计算。完整测试为 **108 passed**（PyTorch 2.7.1 CPU/CUDA）；PyTorch 2.2.1 CPU为 **90 passed、18 CUDA项跳过**。小型CPU交互式演示仅验证显示与控制流程，不代表正式训练结果。

日期：2026-10-04。本文件记录执行证据，不代表材料性能复现。

2026-10-05 更新：训练批处理、AMP、恢复、完整数据审计、大模型与压力测试的最新记录见 [正式训练前准备](PRETRAINING_READINESS.md) 和 [容量审计](MODEL_SCALE.md)。以下保留上一轮历史测试，不代表新版最终状态。

服务器发布验证：在独立目录导入仓库数据包，68个文件字节校验通过，三套来源及共享模型的四个训练入口全部通过，联合记录守恒102610条；独立目录完整测试 **88 passed**。JUnit为`artifacts/distribution_check/UnifiedGroupGen/artifacts/distribution_tests.xml`，训练入口证据为同目录`artifacts/portable_training_gate.json`。新增4项测试检查路径迁移、导入幂等性、损坏文件拒绝、路径越界拒绝和完整训练计划。Linux安装/4090训练仍需服务器实测。操作见[服务器训练说明](REMOTE_TRAINING.md)。

PyTorch 2.2兼容更新：训练器支持旧版`torch.cuda.amp.GradScaler`。PyTorch 2.7.1/CUDA测试 **91 passed**，JUnit `artifacts/pretrain_validation/torch22_amp_compat_current.xml`；实际PyTorch 2.2.1 CPU环境 **80 passed、11 skipped**，JUnit `artifacts/pretrain_validation/torch22_compat.xml`。跳过项均需要CUDA，不能据此声称旧版CUDA测试已通过。实际依赖记录在`artifacts/pretrain_validation/torch22_runtime.json`：NumPy 1.26.4、SciPy 1.15.2、PyXtal 1.1.3、pymatgen 2025.4.17、spglib 2.6.0。复用服务器cgdit时安装本项目使用`--no-deps --no-build-isolation`，保留已有PyTorch及其绑定扩展。

最新整数索引版本完整cgdit测试 **84 passed**，JUnit `artifacts/pretrain_validation/pytest_cgdit_indexing.xml`；新增随机先验及边顺序检查、优化前后独立输出/梯度对照与交替性能测量见正式训练前准备文档末节。下面80项及更早记录保留为历史结果。

## 1. 环境

直接运行现有 `cgdit` Conda 环境的数值操作出现重复 `libiomp5md.dll` 的 OpenMP 冲突。没有设置 `KMP_DUPLICATE_LIB_OK` 忽略该错误，没有修改用户现有环境。

本次使用已有 `nequip` 环境的 Python/PyTorch/NumPy/SciPy，在 sys.path 最后补充 `cgdit` 环境的缺少纯 Python 依赖，完成测试。没有新安装依赖成功；下载尝试已停止。

实际版本：Python 3.10.18，PyTorch 2.6.0+cu118，NumPy 1.26.4，SciPy 1.15.3，pandas 2.3.3，spglib 2.6.0，PyXtal 1.1.1。测试和小训练使用 CPU。

可复查本次测试命令（从父工作目录）：

```powershell
& D:/app/anaconda3/envs/nequip/python.exe -c "import sys,runpy; sys.path.insert(0,'UnifiedGroupGen/src'); sys.path.append('D:/app/anaconda3/envs/cgdit/Lib/site-packages'); sys.argv=['pytest','UnifiedGroupGen/tests','-q']; runpy.run_module('pytest',run_name='__main__')"
```

正式训练应使用依赖一致的干净环境；本次混合读取依赖的方法只作为已执行验证的来源记录。

## 2. 数学与模型测试

最终结果：**26 passed in 3.63s**。

- 全部 230 SG、80 LG、17 PG 的相容晶格正定与操作不变。
- 所有上述群的 Wyckoff 参数真实秩和周期参数整数展开。
- 14 个一般/特殊/非对称型实例的 im(A)=ker(C)，同时验证维数和投影矩阵。
- 重复一般轨道合法，重复零维位置被拒绝。
- 非对称型空间群完整训练路径及速度关系合法。
- 六方非正交度量最小二乘投影与仿射共同约束。
- 层高度不 wrapping，平面 z=0。
- 共线单位磁矩的表示约束及无解情形。
- 描述前缀可完成性、组成计数、模型反向梯度。
- 双阶段性质条件/空条件分支、固定晶格全过程观测保持。
- 缺失性质与零值不同、训练集 scalers。
- 真实 MP20/MPTS52/C2DB 示例的读取与群一致展开。

这是代数与小规模数值验证，不证明完整训练收敛、物性控制或材料稳定性。

## 3. 真实数据转换抽查

| 数据集/划分 | 原始读取 | 接受 | 拒绝 |
|---|---:|---:|---:|
| MP20 train | 8 | 8 | 0 |
| MP20 val | 4 | 4 | 0 |
| MPTS52 train | 8 | 8 | 0 |
| MPTS52 val | 4 | 4 | 0 |
| C2DB train | 8 | 7 | 1 |
| C2DB val | 8 | 6 | 2 |

缓存和失败原因位于 `artifacts/<dataset>/*.jsonl`、`*.rejected.jsonl`、`*.report.json`。

C2DB 配置显式将错误 CIF 字段解释为笛卡尔坐标；源 thickness 用于交叉校验；近对称匹配容差为 0.01（xy 分数坐标与 z Å 的混合检查单位，正式评价应统一为物理距离）。匹配后展开是严格群合法结构。不同坐标单位下的混合容差是当前数据转换限制，不应称为完整物理误差标准。

抽查拒绝：train `1Br2Zr2S3-1` (LG10)；val `1CoNi2Se4-1` (LG13)、`1SiCd2O4-1` (LG22)。它们不能在当前尝试的设置中匹配指定层群，未降级为 P1、未修改标签。源 ASE 晶胞与位置确认、全数据覆盖率和 RMS 理想化偏差尚未完成。

## 4. 三个数据集各一轮小训练

| 数据 | train/val 数 | 最后一轮 train loss | val loss |
|---|---|---:|---:|
| MP20 | 8 / 4 | 39.341377 | 20.534292 |
| MPTS52 | 8 / 4 | 60.231355 | 20.738234 |
| C2DB | 7 / 6 | 16.331112 | 21.948205 |

配置为 hidden=128、3 层描述 Transformer、4 层消息网络，batch_size=2，CPU 一轮。损失只用于确认有限前向/反向和保存检查点。不同数据的损失不构成性能比较。

检查点：`artifacts/smoke/mp20.pt`、`mpts52.pt`、`c2db.pt`，默认不提交 Git。它们是诊断权重，不是已训练充分的模型。

## 5. 采样/导出验证

一轮诊断权重、Heun 四步：

- MP20 性质条件未知描述采样 2/2 输出：目标 SG23、SG229；操作闭合，spglib 回收同群；两个样本的距离有效性均失败。
- 固定 P-1 描述及全部六个晶格参数，1/1 输出：六个参数保持零，实际 SG2；最小距离约 1.250 Å，超过当前 0.6 Å 诊断阈值。
- C2DB 性质条件未知描述采样 2/2 输出：LG23、LG14；指定操作闭合；两个样本距离检查均失败；没有声称精确层群识别成功。

CIF 与全部尝试记录位于 `artifacts/smoke/{inverse_mp20,fixed_mp20,inverse_c2db}/`。

这些采样只证明描述→编译→流→检查→CIF 的流程可执行。前端甚至仍会采样不适合常见材料任务的重元素，因为仅训练一轮且可用元素为 1–118。符号合法性不会排除这类选择。没有化学/稳定性/目标性质认证，不能把五个输出称为五个有效材料。

## 6. 静态与范围检查

- Python 源码 compileall 通过。
- 原仓库 tracked diff 为空，未修改其模型、配置或数据。
- 新代码与文档全部在 `UnifiedGroupGen`；参考下载位于父目录既有忽略路径 `tmp`。
- 用户原未跟踪 `WORKSPACE_AND_RESEARCH_OVERVIEW.md` 和 `assets/` 保持原状。

## 7. 未完成且不能由当前测试替代的工作

正式全数据训练完成、结构/原型去重划分、全部数据坐标来源确认、元素依赖化学筛选、MLIP/DFT/凸包/声子、性质命中率、公平基线与成本比较、真正的非线性条件曲面生成及磁性训练。

## 本机正式运行更新（2026-10-05）

cgdit 单环境最新 **80 项测试通过**，一条预期三维 CIF 有限精度舍入警告；JUnit 为 `artifacts/pretrain_validation/pytest_cgdit_joint.xml`。源码、脚本与测试 compileall 通过。本轮检查时MP20 large 全数据训练已完成十六轮训练/验证，最新验证损失23.5149，最佳22.3923，MPTS52 large 在顺序队列中；尚未完成收敛训练。runner、完整测试集损失和生成评价入口见 `PRETRAINING_READINESS.md`。

二维原始坐标解析现在关闭隐式近分数舍入：真实 `2IIrS2-1` 的来源残差由错误的约1.3e-5 Å恢复到6.285e-9 Å，核对阈值仍为1e-6 Å。layer converter升为6；固定0.1 Å的512条重新转换全部通过，标签与旧诊断相同。流损失复用已计算晶格、先验使用CPU固定掩码数量；新增CPU/CUDA检查验证损失、状态/参数梯度与RNG等价。局部速度测量不能替代正式训练时间，当前MP20已从epoch7检查点受控续训并加载新实现。

层群独立识别已接入 spglib.get_layergroup。在各 LG 的两个异种一般轨道合成测试中，80/80 精确回收目标，证据 `artifacts/pretrain_validation/layer_identifier_80.json`。旧采样的识别结论保留为当时的记录，不事后赋予新结果。

原点搜索统一物理容差后，恢复后的 LG10 样本可在 0.01 Å 下匹配；LG64 例子约 0.01235 Å 的残差显示标签存在容差敏感性。C2DB 官方 JSON 可恢复缺失晶胞，24 条抽查全部与原 CSV 核对，全量恢复仍在运行；二维正式训练仍需来源、全量转换及审计通过。

新版原生层群识别/标准化在512条已恢复记录上全部完成转换，含一条原标签缺失记录；0.1 Å 下511条已有标签匹配509条。固定标准晶胞坐标偏差与二维度量变化单独保存，全量尚未通过。原 CSV 总计94条缺失层群标签，不再要求预先给定 lgnum；原生识别失败仍拒绝。

共享模型增加来源条件，CFG 的空性质分支仍保留来源。单元测试检查跨空间群/层群的联合前向、反向及来源梯度有限，缺失来源/无标签性质条件会拒绝。跨MP20/MPTS52共有19,819个材料ID，其中9,985个处于不同原划分；新共享划分按组件保持所有输入库存，修改标签即使重新计算文件哈希也无法通过 lineage 校验。全量共享划分尚未生成，不能把小测试当成全量泄漏检查通过。

已新增原始ASE数据库恢复入口，通过 exact UID/unique_id、元素/计数、绝对Cartesian坐标、cell长度角度、PBC和能量检查，保存原数据库SHA256。当前官网下载失败 `4Nb2S3-2` 明确为该站点不存在，不能直接用化学式匹配替换。全量恢复及最终共享训练编排器已启动，持有现有进程句柄；来源失败则缓存重试，仍失败时停止后续步骤。已有三维训练独立继续运行。

正式测试输入绑定已经对实际MP20 best检查点通过验证；新增回归测试拒绝不完整测试文件、修改标签和不匹配训练检查点，并核对按来源损失与总体损失的一致性。没有在模型收敛前读取测试损失以调参。

原生识别隔离已经在真实敏感MPTS52输入上复现3221226505退出；父进程继续，后续健康结构回收SG2。模拟超时/崩溃的单元测试确认未知结果仍保留在采样分母，SG/LG正常识别与直接调用一致。隔离用于采样和audit CLI，正式数据转换已有多进程保护；程序生存不等于结构有效。

完整采样CLI复验使用历史诊断权重、固定SG2描述及六个晶格观测，CPU/Heun四步，1/1生成并导出CIF，隔离识别回收SG2且无识别错误，最小距离1.11775 Å。证据 `artifacts/pretrain_validation/isolated_sampling_cli/`。这里只验证CLI→流→识别→导出的连接，没有使用正在训练的模型，未认证化学、稳定性或目标性质。

## 2026-10-17：停止训练后的联合评估入口

- 新增 `scripts/evaluate_joint.py`，直接评价已保存最佳检查点，执行完整审计测试集和三来源各自无条件/条件生成，记录检查点、代码、数据和输出指纹。默认六组各256次尝试，失败与识别未知保留在分母。
- 任务级恢复会核验完成任务的 JSON/CIF 文件；未完成任务保留原目录并在下一 attempt 重新运行。采样逐条落盘，提供状态和简洁进度日志。
- 实际端到端 CPU 测试发现并修复 float32 层结构进入 pymatgen 部分周期距离内核时的 `Buffer dtype mismatch`。在结构导出/评估边界转换为 float64，不改变网络权重或训练状态。增加 float32 SG/LG 回归测试。
- 最终完整回归：`python -m pytest -q -rs` → **96 passed, 18 skipped, 1 warning，85.81秒**。18项因当前兼容环境没有 CUDA/BF16 而跳过；警告为已有 CIF 分数坐标舍入提示。
- 端到端测试使用小型固定零速度网络，实际执行三来源测试及六组采样、原生群识别和 CIF 导出；验证完成任务复用以及篡改输出拒绝。该测试验证执行链路，不代表已训练模型生成质量。
- 从用户提供的 `../tmp/UnifiedGroupGen.zip` 在 CPU 中加载真实共享 `model.best.pt`，两个网络均通过 strict state_dict 加载。epoch 索引194（显示第195轮），描述模型3,718,346参数、流模型14,312,460参数，总计18,030,806。证据保存在本地忽略目录 `artifacts/evaluations/local_checkpoint_compatibility.json`。
- 本轮没有重训、没有运行完整真实联合测试集、没有得到新的物理或性质验证结果。4090正式评估步骤见 [JOINT_EVALUATION.md](JOINT_EVALUATION.md)。

## 2026-10-17：统一表示统计与标准生成评估

- 全量准备记录：MP20 45229/45229、MPTS52 40476/40476、C2DB 16905/16905通过；合计102610条。最大独立参数逆映射误差1.5242e-12，最大群操作及三点路径抽查残差2.6808e-14 Å；全部记录的锚点轨道基底秩等于稳定子约束核维数，切向作用残差均为0。证据见 [assets/representation](../assets/representation/README.md)。
- 目录范围：230 SG/1731单轨道类型、80 LG/477类型、17 PG/72类型全部通过，共2280种。来源CIF的独立重编码为可复现的每来源16条抽查，48/48通过；这是来源抽查，不是102610条原始CIF全部重构通过。
- 新标准评估涵盖前期0.5 Å/SMACT联合有效性、额外周期自身镜像与半径距离、CrystalNN/Magpie覆盖、Wasserstein几何分布、唯一性和相对实际训练参考的新颖性；失败与未知保留分母。
- 二维指纹的真实材料检查复现Voronoi无限顶点，已用固定规则辅助晶胞闭合，同时拒绝具有非零化学权重的法向镜像近邻。验证不同真空、共同法向平移、面内超胞及真实C2DB记录；二维面积/层厚与3D密度分别处理。
- 三套数据各一条真实记录完成几何、SMACT、61/132维指纹及自匹配检查。C2DB例 `2CdCrO3-1` 的周期连接秩为2；这些是接口检查，不是生成质量结果。证据在忽略目录 `artifacts/evaluations/standard_metrics_validation/real_records.json`。
- 独立性质CSV与出处JSON接口支持目标误差、有效命中率、同评估器的性质Wasserstein距离、弛豫后稳定性和SUN。SUN需要真实弛豫CIF重新匹配；没有外部计算时对应值为null，不从输入条件推断结果。
- 最终完整回归：**105 passed, 18 skipped, 8 warnings，113.79秒**。18项为本地CUDA/BF16不可用；提示涉及CIF舍入及指纹的氧化态/半径回退。安装包构建成功，确认包含132维组成标准化资源。本轮没有新训练或新生成的物理质量实验。
- 所有新增图与后续既有绘图入口仅导出PNG；已有PDF/SVG保留。服务器操作见 [STANDARD_EVALUATION.md](STANDARD_EVALUATION.md)。
- 对服务器实际使用的 `--run` 入口另做定向复验：检查点/生成清单绑定模式成功，修改训练参考文件后被拒绝；单项测试1 passed，25.17秒。

## 2026-10-17：论文实验、独立计算与PNG汇总流程

- 新增按TRAIN标签分位数准备目标/CFG扫描、固定描述的合法先验/训练流/固定晶格对照、采样多种子与同分组划分的分来源训练配置；计划绑定实际训练文件SHA，生成任务逐项校验描述/观测文件和预算。未启动新正式训练。
- 真实全训练集的任务准备检查：每来源4096次无条件生成、单种子142，共3任务/12288次；单性质分位数及CFG=0/1/2扫描、每任务128次，共66任务/8448次。这些是计划预算，不是已生成结构数。
- 三份同划分配置成功从各自TRAIN标签拟合缩放并构建网络：MP20 36289条、18,030,806参数；MPTS52 32464条、17,700,054参数；C2DB 13432条、18,030,806参数。MPTS52省略完全无标签的带隙编码器，主干宽度/层数、群和元素词表保持一致，故不能称三个模型参数严格相等。
- 真实MP20分来源配置完成全数据preflight，passed=true、errors=0；仍验证完整联合父数据与分组谱系，并显式核验允许的来源子集。MPTS52/C2DB在本轮做了配置、标签拟合、模型构建检查，没有完成新的正式训练或全量子集preflight。
- 完整回归：**113 passed, 18 skipped, 8 warnings，191.11秒**。跳过项为本地CUDA/BF16不可用。新增测试覆盖实际CPU单来源campaign、配对先验与零速度流、固定晶格残差、完成任务复用与篡改拒绝、缺失标签配置、外部基线失败分母、汇总重复/表格篡改拒绝、ASE计算与断点恢复。
- ASE桥接在工程夹具中验证3D/2D、元素参考形成能/相图计算接口；总能量、形成能、带隙与凸包能区分处理。输入/弛豫CIF、计算器代码/显式权重文件、任务和测试参考指纹可核查；未提供独立值时保持null。工程EMT/Lennard-Jones计算不构成真实材料稳定性或条件达标证据。
- 九类PNG输出完成工程渲染和代表性图片视觉检查；夹具位于本地忽略目录 artifacts/evaluations/paper_workflow_validation，不上传为论文结果。服务器正式联合样本尚未同步回本地，assets/paper只提供索引与按真实结果生成的程序。
- 另做外部基线CIF导入 → 独立Lennard-Jones单点计算 → 生成/参考CSV与出处哈希绑定 → 标准指标 → 论文PNG的串联检查：2次工程尝试中1次导入成功、1次保留失败，联合有效计数1/2，未计算的稳定性保持null，汇总生成3张PNG。证据在 artifacts/evaluations/paper_pipeline_smoke_20261017；该夹具只验证接口连接，不用于论文材料结果。
- 图1–6的证据、结论边界和缺口见 [PAPER_EVIDENCE_PLAN.md](PAPER_EVIDENCE_PLAN.md)，执行步骤见 [PAPER_WORKFLOW.md](PAPER_WORKFLOW.md)。
