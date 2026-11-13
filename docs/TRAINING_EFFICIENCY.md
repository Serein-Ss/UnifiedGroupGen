# 当前联合流的工程效率与启动检查（2026-11-13）

本页针对 joint_flow_v2。默认 FP32 与原有权重格式保留；不把短测当作完整训练收敛或材料质量证明。

## 本轮接通的能力

- 周期邻居从逐原子循环改为稳定双键排序，保持边顺序、周期自像及等距壳层；缓存有限个 CPU 平移网格。
- 联合训练的 precision 配置实际交给 Lightning；可选 BF16/FP16 混合精度，权重与关键几何仍保持 FP32。
- 约束解码、晶格映射及坐标切空间求解退出 autocast；跳转对数损失采用 FP32。
- 可选 CUDA fused AdamW。新增 runtime=joint_flow_fast：BF16＋融合优化器＋降低日志写入频率。
- model.checkpoint_layers=true 在 v2 EquiformerV2 中接通整个原生核心的激活重计算（不是改写原生层）。用于内存折衷，默认关闭；仅训练时启用。
- 保留已实现的 packed 周期图训练、异步批采样、CPU 终点缓存及按预算分批；轨迹记录默认关闭，可仅记录指定样本。

## 本机隔离短计时

RTX 3060；正式4层、128通道、lmax=4、mmax=2、有效8头；每个数据集4条真实TRAIN记录、固定教师中间态；2次热身、5次计时，CUDA同步中位数。包含前向、反向、梯度裁剪和优化器更新。不是完整epoch耗时，不含CPU教师构造、验证和磁盘I/O；小样本结果不能推广成所有数据和显卡的加速保证。

| 数据 | 模式 | 秒/优化步 | 峰值已分配显存 MiB |
|---|---|---:|---:|
| mp_20 | fp32 | 0.6046 | 3048.1 |
| mp_20 | bf16_fused | 0.5276 | 1974.7 |
| mp_20 | bf16_fused_checkpoint | 0.7416 | 1970.3 |
| c2db_51 | fp32 | 0.2950 | 794.4 |
| c2db_51 | bf16_fused | 0.2940 | 671.8 |
| c2db_51 | bf16_fused_checkpoint | 0.3956 | 726.8 |

BF16融合模式在该短测中使mp_20步吞吐提高约15%、峰值显存降低约35%；c2db_51耗时基本持平、显存降低约15%。重计算不一定进一步省显存，且会增加耗时，不能作为默认提速开关。FP16入口由Lightning原生梯度缩放支持，本轮主要验证的是BF16；不声称完成所有精度/硬件组合的训练质量验证。原始记录由 scripts/benchmark_joint_flow_efficiency.py --reference-model 生成。

## 正式训练前必须做的数据审计

新选择器首次运行需要匹配其完整配置的数据审计；检查器有意拒绝不存在、失败或配置不一致的审计。不能只凭 task=plan 成功宣称正式入口可运行。

```powershell
python -m unifiedgroupgen.joint audit --experiment flow_mp20_dng
python -m unifiedgroupgen.run experiment=flow_mp20_dng runtime=joint_flow_fast device=cuda:0
```

C2DB替换为 flow_c2db51_dng；CSP/性质任务使用相应后缀，分别审计。审计只读取数据并写报告，不训练。runtime=joint_flow_fast 是执行设置，保持数据/模型定义哈希不变。恢复训练时保持原检查点的配置和执行设置，不在已有运行中途更换精度或优化器。

当前新的 flow_*_dng 明确默认 chemical_set_probability=0，进行纯无条件训练；旧 task_* 兼容选择器仍保留原来的0.5。若要独立训练限定元素集合任务，在审计和训练两处一致覆盖为1：

```powershell
python -m unifiedgroupgen.joint audit --experiment flow_mp20_dng --override train.chemical_set_probability=1
python -m unifiedgroupgen.run experiment=flow_mp20_dng train.chemical_set_probability=1 runtime=joint_flow_fast device=cuda:0
```

该覆盖会重写同选择器本地审计报告；切换回默认配置需重新审计。要长期保留两套报告，可在两处同时覆盖独立 experiment_name。不要绕过数据审计或将TEST的域外元素从评估分母删除。

## 尚未完成的工作与边界

CPU教师路径仍逐记录构造；并未实现全GPU数据流水线、全局最优批量搜索、分布式训练或稳定可用的torch.compile整模型编译。原两阶段模型的edge_chunk_size不能据此认定已在新核心逐边分块。未通过缩短生成步数、截断邻居壳层或放宽对称约束换取加速。完整训练、收敛质量、长期BF16稳定性与DFT仍须后续验证。

## 重复测量

```powershell
python scripts/benchmark_joint_flow_efficiency.py --reference-model --steps 5 --warmup 2 --records 4 --output intermediate/checks/my_efficiency.json
```

避免与其他GPU任务同时计时，使用新的输出文件。默认不加 --reference-model 时是小网络工程诊断。软件依据：[PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)、[AdamW](https://docs.pytorch.org/docs/2.7/generated/torch.optim.AdamW.html)、[Lightning混合精度](https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html)。
