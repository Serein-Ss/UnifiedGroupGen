# 单网络联合生成器：当前实现与运行契约

现行理论版本：2026-11-12，完整定义与证明见[THEORY.md](THEORY.md)，项目目标见[RESEARCH_PLAN.md](RESEARCH_PLAN.md)。本文说明当前已落地的条件联合内核，不再将其当作整个项目必须由用户指定群的限制。

当前默认支持无群输入的目标采用按结构类型统计的TRAIN频率外部初始化，不训练神经选群头。新task_*检查点已接入TRAIN频率和无群输入入口；旧无先验检查点仍需spec。“条件生成”仅指目标性质；群和维度为硬规范，组分为CSP任务硬输入。固定计量CSP已有独立训练与余额受限核；动态观测及完整性质CFG尚需对应路径支持，见[TASK_DEFINITIONS.md](TASK_DEFINITIONS.md)。

这是显式启用的单网络路径：在固定群规范下，用完整原子图共享网络联合生成元素、晶格、坐标和轨道配置，并支持轨道事件改变原子数。原有 DescriptorTransformer、OrbitFlow、检查点格式、生成入口和默认实验保留兼容与对照，本轮理论更新不替换运行默认值。

本次落地包含四个能力层：统一操作与约束编译、共享原子图上的离散/连续联合生成、可更新轨道与变原子数跳转、真实数据训练及采样验收。它们在同一条新模型路径中衔接，磁性分支按任务要求只补实现与数据契约。

## 实现位置

| 层 | 实现 |
| --- | --- |
| 群作用、轨道模板、公共连续状态解码 | `unifiedgroupgen/group_constraints/unified.py` |
| 一般非共线磁矩、轴矢量、时间反演和连续自旋生成元 | `unifiedgroupgen/group_constraints/fields.py` |
| 状态、合法动作、联合跳转核 | `unifiedgroupgen/group_constraints/state.py` |
| 现有数据适配及实际条件路径模拟 | `unifiedgroupgen/pl_data/joint_path.py` |
| 数据缓存和批处理 | `unifiedgroupgen/pl_data/joint_datamodule.py` |
| 单一完整原子图主干 | `unifiedgroupgen/pl_modules/joint_graph.py` |
| 生成器匹配损失 | `unifiedgroupgen/pl_modules/objectives/joint_generator.py` |
| Lightning、检查点和续训 | `unifiedgroupgen/pl_modules/joint_training.py` |
| 事件感知 Heun 采样 | `unifiedgroupgen/generation/joint.py` |
| 操作残差、最大群恢复及独立原生进程 | `unifiedgroupgen/group_constraints/identification.py` |
| 新命令适配器 | `python -m unifiedgroupgen.joint` |

## 群与内部表示契约

`SymmetrySpec` 包含周期掩码、几何维数、空间仿射操作、内部表示、可选连续自旋生成元和轨道模板。空间操作使用适配坐标：周期方向为分数坐标，非周期方向为真实长度。空间群、层群和平面群的内置目录沿用已有 PyXtal 1.1.3 数据及编码约定，覆盖 230/80/17 类。

层群 z 不取模；平面群 z 恒为零。杆群和点群通过显式操作与模板接入，不声称内置全部杆群或点群目录。周期操作必须保持周期子空间，并是整数幺模晶格自同构；非周期子空间使用欧氏等距操作。

每个模板的整个仿射坐标图都检查群作用一致性，不仅检查一个样本点。晶格在允许的对称度量子空间中指数参数化，生成正定度量。公共坐标保留法向残差，解码后满足当前占据模板。

`Operation.spin` 是完整的实磁矩表示，已经包含时间反演时不再乘一次符号。未显式提供 spin 时，磁空间群使用 `epsilon * det(Q) * Q`，其中 Q 从当前晶格转换到笛卡尔坐标。一般磁矩通过所有操作及连续自旋生成元的公共零空间投影。

`SymmetrySpec.magnetic_database(uni_number, templates=...)` 读取 spglib 的磁空间群操作。未给模板时仅构建空间上去重的一般轨道；磁性特殊位置必须提供与操作设置一致的模板。自旋空间群接受显式空间/自旋操作及连续 spin-only 生成元，没有捏造一个覆盖全部自旋群编号的数据库接口。

当前内部变量是实的经典局域磁矩。非共格磁序、量子自旋子和任意非线性序参量不在本版保证范围内。共享网络在规范坐标框架中计算；严格样本约束来自编译器，不另行声称网络具有独立连续 SO(3)×SO(3) 等变性。

## 状态、动作和概率过程

状态包含当前占据轨道、离散元素（MASK=0）、公共晶格/坐标/可选磁矩状态、事件计数和有限事件预算。群规范固定；骨架更新改变轨道占据、原子等价关系和允许子空间。

空原子图加全局 token 是合法初始态，END 必须非空且没有 MASK。最后一个预算槽专门用于 FAIL；教师程序必须在它之前完成，采样器不会在预算边界强迫选择 END。

- `birth`：选择任一合法模板，从固定高斯核新增一个完整轨道。
- `split`：复制一个有自由度的轨道的公共连续状态，并向新增副本添加固定扰动；原原子不变。
- `delete`：删除一个完整轨道，允许中间态回到空图。
- `element`：同步修改轨道内所有原子的元素。
- `swap`：协同交换两个轨道的元素；不自动宣称保持重数不同轨道的化学计量。
- `replace`：联合替换轨道模板和元素。重数不变时保留公共连续状态并更换解码器；重数改变时新块使用同一出生核，其他原子保留公共状态。
- `end`：离散编辑停止，连续生成继续。
- `fail`：无合法完成方式或预算边界的显式失败。

零自由度轨道禁止重复占据同一个模板。有自由度轨道可以重复占据。每个动作均通过相同的 `T_a` 和随机核 `kappa_a` 用于训练和推理；不在采样时额外挑选“最优出生位置”。所有原子的对应和维度改变由跳转核完成。

隐藏特征不作为持久状态跨事件缓存；每次跳转后由更新的完整原子图、化学标签、公共几何和操作关系重新计算。因此无需为不同原子数的旧隐藏张量定义额外的插值规则。

候选轨道除重数、自由度及偏移统计外，还对完整仿射基 `A`、周期偏移的正余弦和非周期偏移逐点编码后聚合。空间群 47 的 `y/w` 等轨道具有相同重数、自由度和偏移，但自由方向不同；仿射基输入使其评分可以区分，且评分差可对晶格度量反向传播。

生成器为连续漂移与完整跳转测度之和。动作的离散标记包括受影响轨道、新模板和新元素，不只包含一个模糊的“修改骨架”类别。

教师程序从真实终点提取目标轨道，随机安排出生、元素赋值和 END，并插入可恢复的错误元素、临时轨道、删除、替换及成对交换。重复一般轨道可以通过 split 到达。程序不预先固定随机出生坐标或事件时刻。长度 M 的程序在进度 l 的下一事件速率为 `(M-l)/(1-t)`，事件时间可由排序后的均匀随机数精确采样。

事件间朝当前分支锚点做连续桥，跨事件使用真实维度变换。训练样本通过模拟到 t 得到，不把不同原子数的两个向量直接线性插值。最终保留原子的锚点对应真实终点；临时原子在终点前删除。

模型输出连续锚点、剩余事件均值 mu 和所有合法联合动作的归一化概率 pi。连续速度为 `(anchor-Y)/(1-t)`，动作速率为 `mu*pi/(1-t)`。损失为锚点 MSE 加 `mu-k*log(mu)-k*log(pi(next))`，k 是教师剩余事件数。k 不能从动作交叉熵中删掉。

活跃教师至少还有END。理论最优mu是剩余事件数的条件均值；代码采用1+(budget-step-1)*sigmoid(v)。乘子非零时，有限权重仅能极限逼近边界1，不能声称精确表达全部理想均值。预算避免无限事件，末槽保留FAIL；有限epsilon、步长和学习误差仍可导致未完成。检查点按原实现保留全局计数与RNG，并处理Lightning本地循环计数；旧格式不更改。

条件生成器平均在适当良定性条件下给出目标边缘路径；不声称复现教师轨迹或材料真实形成路径。事件时钟和一般生成器匹配分别借鉴 Trajectory Matching、Generator Matching；分支/删除及锚点借鉴 Branching Flows。

## MP-20 与 C2DB

直接读取当前 `data/dataset/full_verified_v1/mp_20` 和 `c2db_51` 的验证编码，不重写、重新标准化或更改原始划分。旧局部状态转换成公共坐标和公共对称矩阵状态。常规晶胞展开后的 MP-20 可能超过 20 原子，因此默认上限沿用 192，而不是按数据集名称截为 20。

本次逐条端点适配检查覆盖 62,134 条记录：MP-20 的 train/val/test 为 27,136/9,047/9,046，C2DB 为 10,143/3,381/3,381；六份输入文件哈希不变。训练集最大轨道数分别为 20 和 51，最短完整教师程序最多 103 个事件，小于默认预算 256。

元素词表仅从 TRAIN 拟合。两套数据的全部训练与验证目标均在该支持集中；MP-20 测试集的 `mp-23155` 和 `mp-568145` 含训练词表未见的 Ar（18）。它们的几何端点可适配，但不是该词表下受支持的元素监督目标；保留为显式 OOD 记录，不偷偷丢弃，也不使用测试信息扩展词表。逐项记录见 `intermediate/checks/joint_symmetry_v1/path_capacity.json`。

磁矩缺失保持 `None`，不填零、不计算磁矩损失。新检查点记录 `magnetic_training`；无对齐标签训练的检查点不能通过命令行冒充磁性生成模型。

磁性标签需附显式 `symmetry`、`magnetic_moments` 和 `moment_order: compiled`。一般显式操作记录还可以提供 `public_state`，包含 `letters/elements/metric/coordinates`，通过同一数据接口训练。对齐和群约束不满足时拒绝记录。

## 使用

使用项目已有 Python 环境，不需要替换旧依赖环境：

本机 Windows 使用 `D:\app\anaconda3\envs\unifiedgroupgen_cuda\python.exe`。运行前在当前 PowerShell 进程设置 `$env:PYTHONUTF8='1'`；本次发现 SMACT 3.2.0 在默认 GBK 下读取其 UTF-8 资源会失败，UTF-8 启动模式下复测旧标准指标。此设置不修改系统环境变量或旧评估算法。以下命令中的 `python` 指这个已有环境的解释器。

```bash
# 三套显式启用的新 Hydra 选择器；plan 不训练。
python -m unifiedgroupgen.run experiment=joint_symmetry_joint task=plan device=cpu

# 真实数据小规模训练；输出只放在 intermediate。
python -m unifiedgroupgen.joint smoke --records 2 --steps 4 --device cpu
python -m unifiedgroupgen.joint smoke --records 2 --steps 4 --device cuda:0 --output intermediate/checks/joint_symmetry_v1/cuda/model.pt

# 正式训练仍经过现有完整数据审计门禁。
python -m unifiedgroupgen.joint audit --experiment joint_symmetry_mp20
python -m unifiedgroupgen.run experiment=joint_symmetry_mp20 logging.mode=disabled

# 其他选择器：joint_symmetry_c2db51、joint_symmetry_joint。
# 新检查点使用新采样入口；不会送入旧 proposal/flow 加载器。
python -m unifiedgroupgen.joint sample --checkpoint outputs/EXPERIMENT/RUN/model.pt --kind space --group 225 --count 16 --output outputs/EXPERIMENT/RUN/generate/fixed_group/samples.jsonl --exact-group
```

新原生 `last.ckpt` 包含单个 model、优化器、循环、RNG、配置及数据哈希；新 `.pt` 是推理导出，格式标记为 `ugg_joint_symmetry_v1`。正式入口可以用原有 `resume` 和 `initialize_from` 参数，但只接受对应新架构检查点。旧格式及其加载函数不改变。

新加载器通过 Lightning 的 `state_dict/load_state_dict` 协议保存 epoch 和已消费批次游标；同时修正本机 Lightning 在 `max_steps` 中途停止时推进的 epoch 计数。epoch 边界和中途保存两种续训均用全部参数与连续训练的逐项完全一致来验证，限定当前单进程、有限批次、`num_workers=0` 路径。

已有评估账本仍绑定其生成时的源码哈希，切换代码版本会按原规则拒绝跨版本复用。本次不放宽这一校验；历史账本应配合相应冻结源码使用。同一固定源码内的旧生成、评估及断点复用另做回归。

采样器联合推进连续状态和累计事件风险，在 Heun 局部密集输出中定位事件；已接受步不会跨事件。有限步长和终端 epsilon 是数值近似，记录在输出中。预算耗尽或未 END 的尝试保留失败原因，不强行投影成成功样本。

每次尝试写一条 JSONL，保留状态、动作及原子数变化、约束残差和可选最大群识别。`--exact-group` 检查最大群，不只检查群包含关系；拒绝的尝试仍保留在分母。空间/层/平面群识别在隔离原生进程中运行。显式其他群族若没有最大群识别器会返回未知，不伪造成功。

接受条件同时包含最小距离检查，计入非零周期晶格平移对应的自身镜像，避免单原子小晶胞遗漏距离判定。

普通空间/层/平面结构同时写入内联 CIF、SHA256 和旧评估接口使用的 `generated/failed/rejected` 状态，并做组成回读检查。完整公共连续状态使用独立字段，不冒充旧的局部 k/q 状态。已有样本账本和检查点不被新任务覆盖。

## 验证边界

非磁性验证覆盖全部 327 类目录的度量、坐标及元素约束，层/平面边界、显式杆/点作用、轨道核、真实数据桥终点、共享主干梯度、事件采样、最大群识别和新检查点续训。实际 CPU/CUDA 短跑只验证执行，不证明生成质量、稳定性或充分训练。

另有 256 条分层抽取的真实训练结构、88 个实际出现的群的完整条件路径检查，逐事件核验 4,255 次正常动作全部落在推理支持集中，并核验桥终点；包括 birth、split、delete、element、replace、swap 和 END。它是教师路径及跳转核验证，不是学习模型的生成质量评估。最终证据汇总在 `intermediate/checks/joint_symmetry_v1/validation_report.json`，有效本机短跑权重目录为 `release_cpu`、`release_cuda`、`release_hydra`。

按本次任务优先级，磁矩、磁空间群和自旋表示实现补齐但不开展磁性训练与新磁性测试，也不把旧共线扩展示例当作完整磁性模型证据。数据无标签不能证明这些任务已经学会。

当前采用三项独立任务选择器，不混合任务训练，详见[INDEPENDENT_TASKS.md](INDEPENDENT_TASKS.md)。
