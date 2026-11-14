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
