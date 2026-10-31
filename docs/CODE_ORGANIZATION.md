# 当前代码组织

正式训练入口为 unifiedgroupgen/run.py。数据读取、模型定义、生成和评估各自可独立调用。15套既有配置、数据和检查点不因本轮整理而重写。

|目录|职责|
|---|---|
|common|轻量配置与路径解析、结构显示、源码与文件哈希；不管理Trainer或模型保存|
|group_constraints|群—轨道编译、独立状态、确定性展开、合法切空间和观测约束|
|conditional|性质编码、组成硬约束、部分结构补全|
|pl_data|dataset.py：数据读取、缓存和边预算分批；datamodule.py：LightningDataModule与DataLoader；encoding.py：CIF编解码；prepare.py：源数据/联合划分与审计入口|
|pl_modules|descriptor.py、orbitflow.py：两个模型；backbones：可替换骨架；objectives：损失；factory.py：模型构建与新旧权重读取|
|generation|仅内存采样：描述解码、合法规则、连续积分、轨道编译与几何生成；无配置/检查点/文件/任务调度|
|evaluation|只有common、csp、dng、conditional、utils五个指标模块及指纹常数，无命令入口或调度|
|tests|运行前检查、单元与回归测试、历史参考工具；旧训练基础设施归档在legacy/training_before_native|
|run.py|训练配置、Lightning Trainer、日志和训练续训|
|generate.py|Hydra配置、请求解析、检查点、任务选择、批量执行、结果保存、进度和已有候选核验|
|evaluate.py|已有结构的完整指标计算、可选性质预测、CSV/JSON/PNG输出|
|scripts/evaluate_models.py|多个模型的生成和指标任务组织、已完成任务核验|

pl_data仅有4个功能文件，另加 __init__.py。离线ZIP打包/解包和外部C2DB源数据恢复工具移至 scripts/data_tools，不进入训练热路径。PL不能替代CIF标准化、群—轨道编码、数据身份核验和按边预算分批，因此保留这些材料领域代码。

common/configuration.py保留一份共享的路径/YAML/.env解析函数：run.py启动时调用它，独立生成和评估也调用它。将这些函数全塞进run.py会让只读数据或评估导入训练框架。每个入口只创建自身任务的输出；训练日志装配保留在run.py。

新训练直接使用Lightning的ModelCheckpoint、EarlyStopping、LearningRateMonitor、CSVLogger和WandbLogger，不另写模型保存器。详情见 [训练与恢复](LIGHTNING_TRAINING.md)。

参考 [DiffCSP++的入口](https://github.com/jiaor17/DiffCSP-PP/blob/main/diffcsp/run.py) 的职责划分；它的pl_data文件较少，但CIF预处理另放在common/data_utils.py。本项目没有复制其扩散算法。


生成算法接口为`generation/structure.py`的`sample_orbits`和`sample_geometry`，分别返回编译后的合法空间、展开后的连续生成结果。`descriptor.py`、`flow.py`、`rules.py`保留原算法。原`generation/pipeline.py`、`tasks.py`、`requests.py`的工程职责移至`generate.py`，旧实现仅在tests/legacy/generation_before_entry供回归比较。

此次只改变职责和文件位置，不改变采样公式或既有输出协议：通用请求路径仍保存逐结构CIF及JSONL，批量实验路径仍保存内嵌CIF的JSONL。如何启动任务、选择条件和设置输出全部由configs/generate.yaml、configs/generation/default.yaml及generate.py决定。
