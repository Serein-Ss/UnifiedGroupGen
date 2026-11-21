# 原始输入与来源材料

此目录名为 `src`，保存原始数据集及来源核验材料，不是Python源码目录。

- `mp_20/`、`mpts_52/`：源CSV及说明，由父项目复制；未移动或改写父项目数据。
- `c2db_51/`：数据库恢复后的当前CSV及README，已覆盖项目内旧的错误坐标导出版本。来源、处理记录和最终文件哈希集中在该README。

这些是来源数据。由CSV构建的群—轨道终点Tensor及周期图缓存位于 `../dataset/<数据集>/*.pt`。
不要将模型权重、训练日志或生成结果放入本目录。

历史分发压缩包bundles及中间source_recovery目录已按要求删除；当前数据准备直接读取各数据集目录，模型输入位于data/dataset。

外部性质评估器已单独保存在项目根目录的 `pretrained/property_predictors/matgl/`，不再放在本数据目录中。

## 当前仓库中的正式数据

当前配置从 `data/src/mp_20` 和 `data/src/c2db_51` 读取下列 CSV。划分保持原样。

| 数据集 | train | val | test | 来源 |
| --- | ---: | ---: | ---: | --- |
| MP-20 | 27,136 | 9,047 | 9,046 | Materials Project 的 MP-20 发布划分，CSV 字节与仓库旧版一致 |
| C2DB-51 | 10,143 | 3,381 | 3,381 | 沿用既有划分，CIF 结构按 UID 从 C2DB 官方接口或固定数据库快照恢复，性质标签沿用原 CSV |

处理入口为项目根目录下的 `data/process.py`。它按 `configs/sources/` 中的配置从 CSV 构建 `data/dataset/` 下的 Tensor 缓存和同名配置记录。历史 MPTS-52 数据仍位于仓库的 `data/scr/mpts_52`。
