# 当前15套模型

权重目录为outputs/<实验名称>/<日期>/。每套只保留model.best.pt、model.pt、model.history.jsonl、model.status.json、train.log，生成结构和指标分别在同级generate/<任务>与evaluate/<任务>。最优与末轮权重是同一次训练的检查点，不重复计为模型。

历史15套的日期仍为导入时间，不是训练开始时间。当前模型清单在assets/results/main15，最新小型核验在docs/validation，旧导入归档与过期日志已删除；模型权重不改写。新训练使用W&B记录配置与运行参数，不再创建provenance等旁路配置目录。

见[结果目录规则](OUTPUT_LAYOUT.md)。
