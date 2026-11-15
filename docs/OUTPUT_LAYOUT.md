# 模型训练、生成与评估的输出规范

本页以当前 `joint_flow_v2` 为主，保留旧两阶段模型和 `joint_symmetry_v1` 的兼容边界。
规范适用于新增运行，不迁移、重命名、删除或改写已有数据、权重和结果。
本轮仅在本地实现和验证，不修改已经冻结的服务器任务。

## 目录对应关系

```text
outputs/<实验名>/<运行名>/
├── model.status.json
├── model.history.jsonl
├── best.ckpt
├── last.ckpt
├── model.pt
├── generate/
│   └── <生成任务名>/
│       ├── samples.jsonl
│       ├── summary.json
│       ├── sampling.status.json
│       └── trajectories/             # 可选
│           └── sample_000000.jsonl
└── evaluate/
    └── <相同生成任务名>/
        ├── samples.csv
        ├── summary.json
        ├── metrics.status.json
        ├── manifest.json
        ├── cases.csv                 # CSP 可选
        ├── csp_metrics.csv           # CSP 可选
        ├── *.png                    # 按任务绘制
        └── properties/              # 性质代理评估可选
```

训练根目录使用 `.env` 的 `HYDRA_JOBS`，缺省为项目根目录下的 `outputs`。
新训练默认以实际开始时间生成运行名；显式 `output` 和续训位置保持原行为。
生成默认跟随检查点所在运行目录；评估跟随标准的 `generate/<任务>/samples.jsonl`。
文件在相应阶段执行后出现，并非创建目录时一次性生成。

## 新联合模型训练

| 文件 | 内容与保存时机 |
|---|---|
| `model.status.json` | 初始化、训练、验证、保存及结束状态，JSON 更新采用临时文件原子替换 |
| `model.history.jsonl` | 每轮结束追加一行，含 epoch、global_step、训练/验证分项损失；启用时还有学习率和早停状态 |
| `model.steps.jsonl` | 每个训练/验证 batch 追加结构化记录：时间、epoch、batch、global_step、loss，以及训练学习率；续训追加 start 标记 |
| `best.ckpt` | 按验证监控指标选择的原生 Lightning 检查点 |
| `last.ckpt` | 原生续训检查点，含优化器、调度器、早停、循环及必要随机状态，按现有 epoch 策略及正常结束时保存 |
| `model.pt` | 正常结束或早停完成后的最终权重导出，含配置、运行参数、数据哈希、epoch、global_step，不含完整优化器恢复状态 |

新模型续训使用 `last.ckpt`，生成/评估通常选择 `best.ckpt`。
不能把新模型 `model.pt` 当成完整续训检查点。
检查点格式仍为 `ugg_joint_flow_v2`；本次不改变参数、损失、随机路径和检查点恢复契约。

状态的 `state` 可为 `initializing`、`running`、`complete`、`failed`、`interrupted`。
`phase` 可依次出现：

```text
building_model → preparing_targets → trainer_setup → waiting_for_first_batch
→ training ↔ validation → epoch_end → saving → finished
```

`preparing_targets` 包含 CPU 上全量训练/验证目标缓存及约束准备，不表示已经进行 GPU 优化。
首批完成前 `global_step`、`epoch` 可以为 null；阶段标记不是逐记录进度或定时心跳。
训练 batch 0 和后续每25步更新当前批次 loss、epoch、global_step；验证及轮次结束也更新阶段。
`epoch=0` 表示第一轮，`global_step>0` 才说明已完成训练更新。
可捕获异常或中断写明异常类型、原因及最后阶段，并继续抛出原异常。
强制终止、主机掉电或磁盘写入失败时不能保证终态写盘，残留 `running` 不能证明进程存活。

没有 `resume` 时，已有的新模型权重、检查点、历史或状态产物阻止覆盖运行。
显式恢复仍追加历史并更新同一运行状态及检查点，不自动复制或删除旧结果。
训练不会自动启动结构生成、材料质量评估或 DFT。
W&B 继续使用 `WANDB_DIR`；新联合入口不强制创建 `train.log`，终端/调度输出由调用方保留。

交互终端使用 Lightning 原生进度条原地刷新，只显示 loss、验证 loss、学习率、计数、速度和 ETA。非交互重定向（包括 Slurm `.out`）不写回车或 ANSI 刷新码，仅保留简洁的轮次摘要和原有异常信息；普通文件本身不能提供原地动画。旧两阶段入口使用相同显示和步骤记录，原 `train.log` 保留。

训练终端固定显示两行：第一行 `Epoch 54/1000 (bs=8)` 汇总，第二行 `Train`。只有启动验证时，才在第三行显示 `Validate`；验证期间上面两行保持显示，验证结束后第三行收起。训练、验证、生成、评估共用固定宽度的标题、条形和计数列，后续字段统一以 ` | ` 分隔。两条进度条的列相同，但 `loss` 和三项分量分别来自训练、验证各自的当前 batch；验证行不复用训练 loss。顶部汇总的 `loss`/`val` 分别表示训练和验证均值。显示这些指标不改变损失计算或验证聚合。

`bs` 在顶部 Epoch 计数后用括号显示，是当前运行配置的 batch size 上限，实际批次数量可能受原子/边预算或最后一批影响。新联合模型显示当前阶段当前 batch 的 `geom`（连续几何）、`elem`（元素）、`jump`（轨道跳转）原始损失分项，权重仍由原目标函数处理，不在显示层重新计算。旧架构未单独定义的分项显示 `-`，不伪造拆分值。未启动或未安排验证时不创建第三行，也不显示等待或 skipped 占位行；完成的验证指标仍保存到原日志和 Epoch 汇总。单独运行验证时仍显示带 Epoch/bs 的验证标题。

每轮结束在顶部原地更新汇总的 `loss`、`val`、`best`（包含本轮在内的最佳验证监控值）和 `early=等待计数/耐心值`，未启用早停时为 `off`。不另在进度条下方追加一条 Epoch 摘要；重定向日志仍保留每轮文本摘要。最佳值同时参考检查点已有最佳值和本轮指标，避免 ModelCheckpoint 在摘要之后执行时显示上一轮最佳值；读取这些信息不会修改检查点或早停回调状态。`model.steps.jsonl` 追加配置/实际 batch size、训练和验证 batch 分项，以及轮次平均分项、最佳值和早停状态；原 `model.history.jsonl` 保持原样。

顶部的 `step | loss | val | best | early | lr` 列在训练中、验证中和轮末均保留固定位置；本轮均值尚未汇总时 `loss/val` 显示 `-`，轮末再填入数值，不挤动后面的最佳值和早停计数。

Train 与 Validate 使用独立的单调阶段时钟。进入验证时暂停 Train，验证期间其 time、速度和 ETA 保持不变；若轮内验证结束后继续训练，再恢复 Train 的时钟。训练批次全部完成或轮次结束后保持冻结，下一轮重新计时。Validate 的计数、time、平均速度及 ETA 只使用当前验证阶段（多 dataloader 时按当前 loader）的时间，不计训练耗时。速度按已处理 batch 数/本阶段有效耗时计算，ETA 仅估计本阶段剩余 batch。此处为主机墙钟阶段时间，不是 CUDA kernel profiling。

Train/Validate 的标题、计数、速度及数值列采用较紧凑的固定宽度，保持两条进度条的竖线同列，Epoch 汇总仍独立放在上方。JSONL 详细数值不受显示精度或列宽影响。每个 batch 的 `phase_seconds` 和轮末的 `train_seconds`/`validation_seconds` 记录本轮累计阶段耗时；多次轮内验证累计到 validation_seconds，非交互日志也记录这些信息。

新联合生成在标准任务目录追加 `sampling.progress.jsonl`，包含已提交连续/跳转采样步和每个结构的完成记录；自定义 ledger 文件名使用前缀隔离。旧生成入口也记录样本或批次进度。评估的参考结构及生成结构处理写入 `metrics.progress.jsonl`，可选性质预测写入 `prediction.progress.jsonl`。进度文件不保存完整原子轨迹，轨迹仍由独立的 trajectory 设置控制。

这些 JSONL 记录使用缓冲写入，至少每秒（有新记录时）、阶段结束和可捕获异常时刷新；强制杀进程可能丢失缓冲区尾部。显示与记录不调用随机数、不改优化器、采样器或检查点参数。已运行的冻结服务器任务不会自动切换到新版显示。

已有任务可以用新版的只读查看器展示总体 epoch 进度、阶段、global_step、loss、验证 loss 和状态更新时间间隔：`python -m unifiedgroupgen.common.progress <DNG的model.status.json> <CSP的model.status.json>`。它每两秒读取文件，Ctrl-C 只退出查看器，不修改文件、不停止训练。旧状态文件没有 batch 总数时只展示总体 epoch 进度，不猜测轮内百分比或 ETA。查看器也可直接作为单文件运行，不需要替换原冻结训练代码。

## 新联合模型生成

以下是用法示例，本次更新没有执行这些模型命令：

```bash
python -m unifiedgroupgen.joint sample --checkpoint outputs/EXPERIMENT/RUN/best.ckpt --kind space --seed 142 --task-name space_dng_seed142 --count 256
```

省略 `--output` 时，v2 输出到 `<检查点所在目录>/generate/<task-name>/samples.jsonl`。
省略 `--task-name` 时使用 `<kind>_<generation_task>_seed<seed>`，没有 kind 时使用 `mixed`。
不同条件、预算和重跑应使用不同任务名；重复名称不会自动续写。任务名不能逃逸运行目录。
显式 `--output` 的位置和文件名保持有效，并优先于任务名推导。

| 账本文件名 | 同目录摘要 | 同目录状态 |
|---|---|---|
| `samples.jsonl` | `summary.json` | `sampling.status.json` |
| 显式 `custom.jsonl` | `custom.jsonl.summary.json` | `custom.jsonl.sampling.status.json` |

不同显式文件名使用各自的配套文件，避免同目录串写。账本、对应摘要或状态任一存在时拒绝开始。
已存在的所选轨迹也会保留并报错。预检查失败不覆盖旧状态。

账本每行保存一次已完成的尝试，包括生成、拒绝和失败；成功结构保留内嵌 CIF 和 `cif_sha256`。
不默认逐候选另写 CIF，仍可用原工具选取成功记录单独导出。
样本字段、轨迹帧、随机抽样和模型计算不因配套文件而改变。

摘要和状态使用 `ugg_joint_flow_sampling_v1`，保留旧终端返回字段，并补充：

- `requested`：请求分母；旧字段 `attempts` 继续表示请求数。
- `attempted`：实际写入账本的完整尝试数；`unattempted=requested-attempted`。
- `outcomes`：所有记录状态计数；`generated`、`rejected`、`failed` 是常用分项。
- `accepted`：接受数，包括仅适合公共状态导出的 `generated_state`。
- `checkpoint`、`generation_task`、`dataset`、`seed`、`steps`、`batch_size`、`request`：来源和请求。
- `started`、`updated`：Unix时间戳；`seconds`：含结构检查和写盘的过程耗时，不含入口之前的检查点加载。
- `samples`、`summary`、`status_file`：实际路径；开启轨迹时增加目录和已完成轨迹数。

开始时写初始摘要/状态，每批账本 flush 后原子更新状态；结束时同步更新摘要和状态。
`complete` 表示全部请求尝试已处理，不代表全部结构生成成功或物理有效。
可捕获异常标为 `failed`，KeyboardInterrupt/SystemExit 标为 `interrupted`，保存原因后重新抛出。
保留已写入账本和轨迹，不伪造未完成 batch 的结果；未处理数量单独计入 `unattempted`。
参数校验、模型加载失败可能发生在输出创建前，因此不保证有新的状态文件。
强制终止和存储故障不能保证终态写入；中断任务不自动续写，重跑应使用新输出位置。
轨迹由 `--trajectory-dir` 和可选 `--trajectory-indices` 启用，详见 [轨迹规范](GENERATION_TRAJECTORIES.md)。

## 评估

```bash
python -m unifiedgroupgen.evaluate evaluation.samples=outputs/EXPERIMENT/RUN/generate/space_dng_seed142/samples.jsonl evaluation.task=dng evaluation.kind=space
```

标准账本默认对应 `<运行目录>/evaluate/space_dng_seed142`。
非标准目录的显式账本需给出 `evaluation.output`，不猜测其模型归属。
已有完成标记 `manifest.json` 的目录拒绝重算覆盖；保留旧入口的未完成任务处理规则。

`samples.csv` 保存逐尝试指标，`summary.json` 保存任务协议、分母及汇总；
CSP 可增加 `cases.csv`、`csp_metrics.csv`，图和性质预测按配置产生。
`metrics.status.json` 保留输入/输出路径、pid、开始/更新时间、耗时及状态；
可捕获异常/中断分别为 `failed`/`interrupted`，完成后为 `complete`。
`manifest.json` 保存输入身份、指标源码/版本和输出哈希，不能用状态文件代替结果校验。
参考结构缓存默认归 `intermediate/cache`，代理性质不等于 DFT、松弛或稳定性结论。

## 旧入口与历史结果

旧两阶段模型、`generate.py` 和旧检查点保持原有接口。
两阶段的 `model.best.pt`、`model.pt` 可以是原生 Lightning 格式；历史 `.pt` 也可能是旧权重包，
应由加载器识别，不能只按扩展名判断恢复能力。非原生历史权重使用 `initialize_from`，不冒充原生续训。
`joint_symmetry_v1` 采样继续要求显式 `--output`，不新增v2生成摘要/状态。

历史15套模型、日期目录、`baseline256_*`、`diagnostic256_*`、CIF、指标和导入凭据全部保持原位。
历史日期目录可能为导入时间，新默认目录使用实际开始时间。本次不执行迁移/清理，也不重新计算历史指标。
独立短测/工程验证归 `intermediate/checks`，跨模型批次元数据归 `intermediate/evaluation_campaigns`；
模型结构和逐任务指标仍归模型目录。展示材料归 `assets`，网页构建归 `web/symmetry-lab`。
脚本支持显式输出位置；以上按入口区分自动/可选产物，不表示每次运行都会生成所有文件。

本轮本地输出契约、兼容性测试及既有文件保护核对见[验证记录](validation/output_contract_local_20261113.json)。
测试使用临时文件、模拟采样和 FakeTrainer，没有真实训练、参数更新、GPU 推理、GitHub 推送或服务器部署。
真实 Lightning 端到端训练/恢复与材料生成质量不属于这份输出契约验证的结论。
