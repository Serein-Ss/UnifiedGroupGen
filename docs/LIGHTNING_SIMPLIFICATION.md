# Lightning训练管理精简

本次只整理基础设施与目录，不改网络权重布局、群约束构造或损失数学。15套已有模型、日志和数据配置原字节保留。

|原模块|新处理|
|---|---|
|common/checkpoints.py|正式训练直接使用run.py中组装的原生ModelCheckpoint；通用SHA移至common/source.py|
|common/paths.py、settings.py|合并到轻量common/configuration.py；run.py启动时解析，生成与评价共享纯配置函数|
|common/rng.py|移除活动模块；run.py调用seed_everything与Fabric随机状态工具|
|common/logging.py|原生CSVLogger/WandbLogger在run.py组装|
|pl_modules/module.py、trainer.py、callbacks.py、runtime.py|训练适配、优化器、Trainer、回调和恢复全部归入run.py|
|pl_data/loading.py、batching.py|合并为dataset.py，提供CrystalDataset和原边预算分批|
|pl_data/splits.py、joint.py、audit.py|合并到prepare.py，分别通过source/joint/audit模式调用|
|pl_data/bundle.py、recovery.py|离线数据工具移至scripts/data_tools|

## 保留的领域职责

Lightning提供训练循环、设备/精度、日志、检查点和回调，不提供CIF标准化、群—轨道编码、训练数据身份审计或变长原子图的边预算采样。pl_data因此保留dataset、datamodule、encoding、prepare四个功能文件。

DiffCSP++的pl_data文件数量少，是因为大量预处理位于common/data_utils.py；不能据此认为PL已经实现了材料数据处理。本项目采用相似的职责分离，不照搬其扩散模型。

run.py中的UnifiedGroupGenModule是训练适配器；真正的DescriptorTransformer、OrbitFlow、backbones和objectives仍在pl_modules。新原生.ckpt和已有.pt均可通过factory.load_model_checkpoint读取。旧.pt可以生成、评价和初始化权重，不能恢复其不存在的Lightning循环状态。

当前PL 2.5.5的全局随机状态与Plateau检查点时序需要很少量钩子，详见[训练文档](LIGHTNING_TRAINING.md)。不再保留活动的手写保存器、RNG算法或独立Trainer模块。

## 验证证据

最终结果索引：历史工程记录（已清理）。原始日志、测试临时权重与配对梯度哈希放在intermediate/checks/lightning_simplification；运行前源码备份放在已删除的历史备份；测试用旧实现仅归档到tests/legacy/training_before_native。

核验包括：新旧真实模型损失与梯度、原生恢复及学习率衰减、GPU BF16、离线W&B、原生检查点实际采样、全量回归、15份数据门禁、15套旧权重严格加载、150个既有文件哈希和15份配置字节保全。测试结果以JSON中的实际完成状态为准，不把冒烟测试当作正式生成质量评价。

参考：[DiffCSP++训练入口](https://github.com/jiaor17/DiffCSP-PP/blob/main/diffcsp/run.py)、[其数据预处理](https://github.com/jiaor17/DiffCSP-PP/blob/main/diffcsp/common/data_utils.py)、[Lightning原生检查点](https://lightning.ai/docs/pytorch/stable/common/checkpointing_basic.html)。框架具体行为按本机安装的2.5.5源码和回归结果核验。
