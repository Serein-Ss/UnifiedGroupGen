# 进度回调与 Lightning 日志挂接冲突修复

## 根因与最小修复

TrainingProgress 曾用 `self.log` 保存 StepLog。Lightning 2.5 的 `_CallbackConnector._attach_model_logging_functions` 会给每个 callback 绑定模型的 `log`、`log_dict` 方法，从而把记录器覆盖成函数。真实任务在 `on_train_start` 的 `self.log.write` 失败，异常钩子又因同一问题失败；服务器日志显示模型状态恢复成功，但尚未执行第一个训练 batch。

记录器改为 `_step_log`，所有文件写入和关闭使用该属性；`log/log_dict` 保留给 Lightning。训练/验证分阶段时钟、进度布局、JSONL 格式和追加规则保持原逻辑。没有改动模型参数、优化器、调度器、早停规则、数据集、W&B 模式或检查点格式。callback 仍返回空 state_dict，旧检查点不依赖该记录器属性名。

## 回归方式

不调用 Trainer.fit，也不调用 training_step、backward 或 optimizer.step。测试构造真实 CPU Trainer 和 LightningModule，调用真实的模型 callback 接入、日志绑定和 `_call_callback_hooks` 调度方法；不以手工赋值或 fake Trainer 代替这些挂接行为。计数使用合成的恢复状态，batch 输出仅为测试回调的常数。

旧实现已在该测试中精确复现服务器的 `AttributeError: 'function' object has no attribute 'write'`。修复后覆盖：

- 非交互与 TTY 两种模式的启动、训练 batch、验证 batch、轮末和完成钩子；
- Lightning 原生异常包装器的收尾，确认原始异常对象继续向上传播，不被日志错误遮蔽；
- 失败后另建 callback 时追加日志并保留既有行；
- 文件流关闭，Lightning log/log_dict 仍绑定模型，callback 的空状态、模型参数、优化器状态、恢复计数与 RNG 保持不变。

其他回归继续核对阶段计时、W&B 离线接线、30轮早停及既有计数、链式恢复和输出规范。真实 GPU 续训是否成功，仍须在更新服务器后观察，不能用这些不训练的钩子测试替代。

实际待发布源码副本的128项回归全部通过，其中5项使用真实 Lightning 连接器及钩子调度；修复前已复现原异常。详见[发布验证记录](validation/PROGRESS_CALLBACK_BINDING_FIX_20261115.json)。

## 安全续训约定

更新只作用于新的代码工作树；失败运行的 source、campaign、日志、离线 W&B run、best/last 权重和结果保留。不要在旧冻结 source 内原地打补丁，也不要删除 submitted.json 来重试同一运行。

先从新代码只读检查失败运行的 campaign，确认 last.ckpt 可恢复且哈希未变。随后用新运行名称准备独立副本，继续显式设置 `--wandb-mode offline --early-stopping-patience 30`，保留来源的最佳值、wait_count 和所有训练状态。核对新保存路径后再单独提交。旧失败的 W&B run 没有新增训练 loss；新任务创建独立离线会话，不覆盖它。
