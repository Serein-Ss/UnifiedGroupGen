# 修复后独立续训（2026-11-14）

本次发布范围：周期镜像超限时精确分块搜索、无边图上下文及混合批次修复；新模型训练/生成/评估输出规范；以及单独的安全恢复入口。修复测试见[本地验证](validation/PERIODIC_EDGELESS_REPAIR_20261114.md)，输出规范见[OUTPUT_LAYOUT.md](OUTPUT_LAYOUT.md)。不包含此前未提交的网页测试修改、论文草稿或参考代码存档。

## 恢复约定

原运行是 `jointflow_mp20_fp32_v2`，冻结版本 `a5079802`。原目录的 `source/`、`campaign.json`、原配置、审计、校准、日志、`best.ckpt`、`last.ckpt` 和训练结果全部保留。本地代码推送与服务器源码更新都不会自动改变旧冻结源码。

服务器应创建新的代码工作树（例如 `UnifiedGroupGen_resume_fp32`），只把数据目录链接到现有数据，保持 `outputs/` 为新工作树下的新目录。原运行目录通过绝对路径传给恢复入口，不在旧工作树直接 `pull` 或运行 `--retry`。先确认旧作业已结束；本工具不会取消任何作业，也不会终止其他进程。

恢复默认仅选择 `full_mp20_eqv2_dng`、`full_mp20_eqv2_csp`。以下四个阶段分开执行，不把检查、准备和提交合并为一个命令：

1. `python -B submit_files/joint_flow5090_resume.py check --original <原campaign.json绝对路径>`：仅只读检查，输出 JSON，不创建恢复目录、不提交或训练。核对原冻结原始配置哈希、原校准版本身份、拟合后配置、全部审计数据文件、训练/验证哈希、batch size、edge budget、FP32执行参数；对原生模型、优化器及调度器进行CPU严格加载，并检查CPU随机状态。必须看到 `passed: true`。
2. `python -B submit_files/joint_flow5090_resume.py prepare --original <同上> --name <新的恢复名称>`：重复检查后，冻结当前已提交修复源码，只在新工作树创建恢复输出，复制并迁移检查点副本，记录前后哈希与源码继承关系。既有恢复路径会被拒绝覆盖，不允许使用原运行名称。
3. 检查新目录的 `campaign.json`、两个模型的检查点、callback保存路径和保留的历史。确认新 `revision` 与 `original_revision` 分别指向修复版本和原版本，原始 `last.ckpt` 哈希保持不变，待恢复的 epoch/global_step 与原快照一致。此时仍未训练或提交。
4. `python -B submit_files/joint_flow5090_resume.py submit --campaign <新campaign.json绝对路径>`：检查原始清单、审计、profile、原始和副本检查点未变化后，仅提交这两个恢复作业；已记录作业不会重复提交。工作节点从修复后的冻结目录运行并再次核对，再开始实际续训。

实际操作在聊天中逐条给命令，以检查输出作为下一步依据。`check` 的CPU检查不是5090实际恢复已成功的证明；CUDA随机状态、GPU分配和真正的原生恢复只有工作节点实际运行后才能确认。

## 保留训练状态与输出

`prepare` 只在副本的 `ModelCheckpoint` callback 中迁移 `dirpath`、最佳/最后检查点路径和top-k字典键，防止Lightning重置最佳模型历史或删除旧目录权重。模型张量、优化器、调度器、早停、RNG、训练循环、待回放学习率指标、配置和数据哈希保持原内容。原文件不重写。

复制 `best.ckpt` 和 `last.ckpt`，保留不晚于快照 `global_step` 的历史记录。没有复制旧状态文件冒充新进度。恢复从保存的epoch边界开始，原来丢失且未保存的更新不会恢复；1000 epochs 是包含已完成epoch的总上限，不是额外再训练1000轮。

新输出布局为 `新工作树/outputs/<模型名>/<恢复名称>/`，包括 `best.ckpt`、`last.ckpt`、`model.history.jsonl`、`model.status.json`，正常结束后另存 `model.pt`。新日志、兼容性记录、修复代码快照和作业号位于 `新工作树/intermediate/hpc_runs/<恢复名称>/`。

保留原校准选出的DNG batch 8和CSP batch 4及原edge budget。新恢复清单明确它们继承自旧版本，不能把旧显存/吞吐记录当作修复版本重新测得的结果；不会为续训重新调整batch size或重置优化器。原数据审计使用原始未拟合配置核对，实际训练使用检查点内的拟合后配置，避免混淆这两种配置哈希。

旧冻结源码依然只读用于核对配置及加载经过原审计的数据；模型与训练逻辑来自新冻结源码。训练日志使用调度输出和新的JSON历史/状态，不恢复原W&B会话。本工具禁用隐式失败重试；若新恢复作业再次失败，应重新检查其最新快照，再建立下一次独立恢复，保留每次恢复证据。

## 本地验证与边界

验证只执行CPU前向、梯度、状态加载、临时副本迁移和模拟调度命令，没有启动本地真实训练或服务器任务。新增恢复测试显式禁止 `Trainer.fit` 和AdamW参数更新；worker调用仅以mock验证数据入口、拟合配置和新输出路径。原修复与输出规范测试继续回归。发布验证汇总见[恢复发布测试报告](validation/repaired_resume_release_20261114.json)。服务器两份正式权重没有下载到本地，实际兼容性由服务器 `check` 再核对。
