# 保留训练状态的离线30轮早停续训

本入口支持从原始 FP32 campaign 或后续已停止的 repaired-resume campaign 继续恢复。原始目录、每次冻结源码、配置、数据审计、校准、权重和结果都保留；新运行使用独立代码、输出目录和 W&B 离线记录，不取消或重启任何已有作业。

## 只允许明确的早停策略变化

`--early-stopping-patience 30` 是显式的单字段策略迁移，不修改全局默认训练 YAML。代码先按来源检查点的旧配置、运行参数、数据哈希和原校准记录校验，再在内存或新磁盘副本中同步修改：

1. `joint_state.config.early_stopping.patience`；
2. 存在时的 `hyper_parameters.config.early_stopping.patience`；
3. `EarlyStopping` callback 状态中的 `patience`。

同时迁移新副本的 ModelCheckpoint 保存路径。其余字段保持原值，包括 `wait_count`、`best_score`、`stopped_epoch`、min_delta、模型参数、优化器、学习率调度器及其平台计数、随机状态、循环计数、epoch/global_step 和待回放的学习率指标。`last.ckpt` 和历史 `best.ckpt` 各自迁移，最佳权重仍是其原来对应的版本。

只改 YAML 不足以做到这点：原入口会严格核对检查点配置，Lightning 还会从 callback 状态恢复旧耐心值。现在通过有记录的副本迁移使三处配置一致；旧检查点和冻结定义不重写，其他未经声明的配置变化仍被拒绝。

例如无改善计数为12和1时，迁移到30后分别再等待18次和29次无有效改善的验证；有效改善仍清零。每轮验证一次，监控 val/loss，改善阈值沿用来源配置。没有新增学习率门槛或清零规则。

## 服务器逐条执行

先确认来源作业已停止、last.ckpt 可加载并记录 SHA256。源码更新在独立工作树进行；已有数据只作链接，不移动。

- **check**：`python -B submit_files/joint_flow5090_resume.py check --original <已停止的续训campaign.json> --early-stopping-patience 30`。只读检查来源，并对内存中的迁移副本做严格加载；不创建目录、不训练、不提交。输出来源和目标耐心值、保留的计数/最佳值及源文件哈希。
- **prepare**：`python -B submit_files/joint_flow5090_resume.py prepare --original <同上> --name <新运行名> --wandb-mode offline --early-stopping-patience 30`。在新目录冻结已提交源码、保存迁移后的 best/last 副本，并复制不晚于该检查点 step 的历史。已有目标会拒绝覆盖。此时仍不提交或训练。
- **核对**：检查新 campaign 的 `early_stopping_patience=30`、`logging.mode=offline`，副本三处耐心值均为30，源文件哈希不变，保存路径全部位于新运行目录；模型/优化器/调度器/RNG和已有计数不变。
- **submit**：核对通过后再单独提交。不能把 check/prepare/submit 拼成一次命令，也不能向旧冻结 source 中覆盖新代码。本次本地验证不执行 submit 或训练。

## 链式审计与曲线

后一次恢复清单用 SHA256 链接前一次清单，并记录本次早停迁移。检查程序沿链查找最初的数据审计、拟合配置和校准身份；源权重取自最近一次运行。原始拟合配置依次应用清单中已声明的耐心值变化后，必须与最近检查点配置一致。worker 仍用最初签名的数据定义读数据，再用当前迁移后的训练配置续训，不放宽通用检查点加载条件。

W&B 离线接入沿用已修复的 logger；历史曲线脚本也支持多次恢复的清单链，按每次恢复的边界去重和分段。原始历史行和原 run 不回写；离线运行不会自动产生在线网页曲线。

## 验证边界

CPU 测试禁止 Trainer.fit 和优化器更新。验证包括多次恢复、原目录哈希不变、检查点各字段精确保留、原生模型/优化器/调度器加载、RNG不变、Lightning 实际 callback 恢复30轮设置并在剩余18/29次检查后停止，以及两个 worker 的离线 logger 接线。GPU恢复与服务器实际文件仍须在服务器 check 和后续实际运行后确认。

实际暂存源码副本的50项回归全部通过，详见[发布验证记录](validation/OFFLINE30_CHAINED_RESUME_20261115.json)。原生加载使用小型 CPU 模型；不冒充服务器完整 EquiformerV2 权重或 GPU 续训已验证。
