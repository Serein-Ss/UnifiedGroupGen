# 当前训练与恢复

唯一正式入口为 `python -m unifiedgroupgen.run` 或 `ugg`。Hydra组合configs/run.yaml、experiment和runtime；启动时run.py先解析项目根目录、.env、数据位置与输出位置，然后核验数据并构建模型。task=plan只展示配置与路径，不训练或创建运行目录。

```bash
python -m unifiedgroupgen.run experiment=joint_u task=plan
python -m unifiedgroupgen.run experiment=joint_u device=cuda wandb.mode=offline
python -m unifiedgroupgen.run experiment=joint_u device=cuda resume=/absolute/path/model.pt
python -m unifiedgroupgen.run experiment=joint_u device=cuda initialize_from=/absolute/path/model.best.pt
```

resume交给Trainer.fit(ckpt_path=...)恢复模型、优化器、学习率调度器、循环和回调状态。initialize_from严格加载两个网络的权重后开始新运行；两者不能同时使用。既有手写训练的.pt仍可生成、评估和权重初始化，但缺少Lightning循环状态，不能直接作为原生完整续训断点。

描述Transformer与OrbitFlow沿用同一个AdamW和原有损失；模型结构及群约束不变。默认参考长度3 Å、坐标损失除以9，每5轮及最终轮验证。MP20参数256/400000、C2DB参数768/200000、联合参数256/400000分别为batch/边预算；这是4090配置，本机3060小测试应降低batch。15套原始配置不改字节。

## 使用框架已有功能

run.py使用原生ModelCheckpoint直接保存model.best.pt与model.pt；文件是可恢复的Lightning检查点，加载器也兼容已有历史权重。EarlyStopping、LearningRateMonitor、混合精度和梯度裁剪由Lightning管理。EarlyStopping轮数按验证间隔换算，ReduceLROnPlateau按验证频率更新。

当前安装的Lightning 2.5.5没有默认写入全局Python/NumPy/Torch/CUDA随机状态。run.py在检查点钩子中调用Lightning/Fabric的随机状态收集和恢复工具，保留本项目续训所需的状态，没有重写随机数算法。另有一项经回归复现的2.5版本时序问题：原生epoch检查点保存先于本轮Plateau更新；钩子只记录尚未执行的那次更新，恢复后调用原生scheduler.step一次补齐。学习率衰减后的连续训练与中断续训已做参数和调度器状态一致性测试。

这些钩子针对当前2.5系列版本。跨版本、硬件、算子实现或不确定性GPU算法不承诺逐位复现；框架工具升级时应重跑恢复测试。

## 路径与日志

.env中的HYDRA_JOBS指定输出根目录，WANDB_DIR指定W&B日志位置，USE_WANDB_LOGGING=1在线、0离线；logging.mode=disabled只保留五个本地训练文件。在线模式需要用户自己的W&B登录，模型不自动上传W&B。

默认目录为HYDRA_JOBS/实验名/真实开始时间，只保留model.pt、model.best.pt、model.history.jsonl、model.status.json、train.log。配置与监督记录交给WANDB_DIR下的W&B；不再写run.json、.hydra、CSV日志目录或profiler文件。生成与指标见[结果目录规则](OUTPUT_LAYOUT.md)。

训练和验证秒数、损失、梯度裁剪比例及显存峰值写入history/W&B；保存耗时交给原生SimpleProfiler记录，不伪造每轮保存秒数，history中的checkpoint_seconds因此为null并注明来源。

Trainer.test必须给出独立held-out集合。训练损失或冒烟测试不代表材料生成质量。现有source_domain上下文、性质配置、群约束和15套历史权重保持原样。本轮只进行了本地回归，没有连接服务器或启动15套正式训练。
