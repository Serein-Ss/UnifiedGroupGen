# W&B 续训日志与历史曲线补录

## 当前任务与后续任务

`cc7b29de` 的独立恢复入口直接调用 `fit_joint`，没有传入 logger，因此这条入口启动的续训没有 W&B 指标流。原训练入口使用 W&B 离线模式。两种入口都保留每轮 `model.history.jsonl`；补录以该文件为证据，不从权重推测损失。

修复后的恢复入口为后续新启动的任务创建并传入 W&B logger，成功或失败时结束日志会话。`prepare` 默认记录 `logging.mode=offline`、`logging.project=UnifiedGroupGen`，可显式选择 `--wandb-mode online` 或 `disabled`、`--wandb-project`。旧清单未包含 logging 时也使用 offline；升级代码不会改写旧清单、旧冻结源码或正在运行的任务。

联合检查点新增可选的 `wandb_id` 和 `wandb_run_directory` 日志元数据。旧检查点缺少这些字段仍可加载，模型参数键、训练配置、优化器和早停规则不变。同一输出目录的后续在线恢复可复用保存的日志身份；新输出目录或离线会话使用新 ID，避免依赖旧 SDK 的离线 run 恢复行为。原两阶段检查点的 logger 身份读取保持原有行为。

## 补录流程

脚本为 `scripts/wandb_history.py`。服务器操作应在独立的新代码工作树进行，逐条执行；不要在活跃工作树执行 pull、checkout，或替换 campaign 的 source。以下是命令格式说明，不表示已经执行上传。

1. `python -B scripts/wandb_history.py plan --campaign <当前续训campaign.json>`：仅只读检查。根据清单找到原训练和续训历史，不加载模型、权重或 W&B SDK，不创建目录、不访问调度器。输出每个模型的首末 epoch/global_step、去重数、缺失轮次、读取字节数和 SHA256。
2. `python -B scripts/wandb_history.py prepare --campaign <同上> --output <新代码目录/intermediate/wandb_backfill/唯一名称> --project UnifiedGroupGen`：保存独立快照。目标目录必须尚不存在，并与原/续训输出和两个冻结 campaign 目录分离。快照包含 `snapshot.json`、`inputs/` 下的原始完整行前缀，以及每个模型合并后的 `<模型名>.jsonl`。不复制或改写 `.ckpt`、`.pt`、`.env` 或完整源码。
3. 审查 snapshot.json 的模型、数据范围、文件哈希和目标项目，明确要上传的 W&B 账号或团队。随后 `python -B scripts/wandb_history.py publish --snapshot <快照目录> --mode online --entity <账号或团队>`。脚本先校验快照哈希，再上传到明确指定的账号/项目；凭证通过现有 W&B 登录机制提供，不写入快照或源码。

也可使用默认的 `--mode offline`，仅在快照目录写 W&B 离线记录；它不会自动产生网页曲线。离线与在线发布不能对同一个快照互相重试：需要保留离线记录并另做新快照，或在后续明确操作时用 W&B 官方 sync 上传该离线 run。脚本本身不调用 sync 或登录。

## 曲线内容与安全约定

- 每个模型创建单独的 **history-backfill** run，与原训练 run 分离，按续训 campaign 分组。记录总损失、已存在的几何/元素/跳转分项、当轮学习率和已存在的早停指标。横轴 `epoch_number=epoch+1`，同时保留原始从零开始的 epoch 及 global_step。每行以严格递增的 global_step 提交，数据不平滑、不插值。
- 优先读取 `train/loss_epoch` 等整轮均值；`*_step` 的最后一个 batch 指标不冒充整轮均值。没有保存的分项不会补零，CSP 已记录的元素零损失正常保留。无法补回未保存的逐 batch 曲线，也不能从这些日志证明生成性能已经收敛。
- 合并曲线只保留原训练中不晚于恢复检查点 global_step 的记录；原始快照仍保存读取到的完整行前缀。续训文件已复制的原历史去重。相同 epoch 的已选指标或 step 不一致时拒绝补录，避免静默覆盖。缺失轮次会报告，不造数据填补。
- 对仍在写入的历史做一次有界读取，只接受最后一个完整换行之前的记录。末尾未完成行会报告忽略字节数，下次新快照可包含完成后的行；中间完整行损坏、非有限损失或 step 逆序会报错。源历史后来追加不影响已准备快照。
- 每个模型有独立的发布回执 `<模型名>.receipt.json`，记录 mode、entity、run ID、行数和在线 URL。已成功的同目标发布会跳过；失败、进程中断或变更目标时拒绝重复写入，保留证据并要求新快照。在线使用新 ID 与 `resume=never`，不覆盖原 W&B run。发布不提供原训练 run ID 输入。
- W&B 补录关闭代码保存、Git 元数据采集和系统资源统计，仅上传显式构造的指标及有限溯源信息，不上传模型、训练集或本机环境文件。补录进程与 GPU 训练独立；没有自动刷新、循环上传、取消、重启或重新提交任务的行为。

## 验证边界

测试禁止实际 Trainer.fit 和优化器更新，使用合成 JSONL、模拟上传和独立 W&B 离线 smoke 验证；在线服务连通、账号权限及服务器实际历史合并，需要在服务器只读 plan 后逐条确认。本次不改变运行中任务的早停、学习率、batch size 或显示。

发布副本通过 111 项回归；W&B 0.22.3 实际写入两个独立的合成历史离线 run；用户已贴出的 12 条服务器历史行通过整轮均值与记录精度映射检查。记录学习率的 float32 指标与原优化器的 float64 值存在正常舍入差，补录保留日志已有精度。详见[发布验证记录](validation/WANDB_REPAIR_20261115.json)。未执行云端上传或真实训练。

参考：[W&B 离线模式](https://github.com/wandb/docs/blob/main/models/ref/cli/wandb-offline.mdx)、[W&B 初始化参数](https://docs.wandb.ai/models/ref/python/functions/init)、[W&B 指标记录](https://docs.wandb.ai/models/ref/python/experiments/run#method-runlog)。
