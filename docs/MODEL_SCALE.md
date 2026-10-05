# 模型容量审计与正式配置

审计日期：2026-10-05。下列数字是参数注册/构造计数，不是根据文件大小估算。参考模型没有在本次工作中训练或进行数值复现。

## 1. UnifiedGroupGen 的三个容量档位

| MP20 配置 | 描述 Transformer | 几何流网络 | 总参数 |
|---|---:|---:|---:|
| small，原配置 | 766,074 | 631,564 | 1,397,638 |
| medium | 3,599,482 | 3,172,876 | 6,772,358 |
| large | 3,599,482 | 14,046,220 | 17,645,702 |

MPTS52 的 medium/large 总参数分别为 6,782,598 / 17,655,942；C2DB 为 6,701,031 / 17,574,375。差异主要来自群词表及描述序列最大长度，几何网络规模一致。完整计数见 `artifacts/pretrain_validation/capacity.json`。

正式共享配置 `joint_space_layer_large.yaml` 的描述前端为 **3,718,346**，连续后端为 **14,312,460**，总计 **18,030,806**。增加部分来自 SG/LG 共同词表、最大序列长度、三个性质以及计算来源嵌入。计算来源始终保留在 CFG 两个分支中。只有这一套前端与后端，不为每个群训练独立网络；前面独立 MP20/MPTS52 的正式模型用于单数据集参考。

| 组件 | small | medium | large |
|---|---|---|---|
| Transformer hidden / 层数 / heads | 128 / 3 / 4 | 256 / 4 / 8 | 256 / 4 / 8 |
| 几何 hidden / 层数 / Fourier 频数 | 128 / 4 / 8 | 256 / 6 / 16 | 512 / 6 / 128 |
| 精度 | FP32 | BF16 | BF16 |
| 激活重计算 | 关闭 | 开启 | 开启 |

BF16 要求支持该格式的 CUDA 设备；FP16 同样支持且使用 GradScaler。CPU 诊断需 FP32。配置变更后必须重新做数据审计。当前 backbone 是周期边特征的 CSPNet 风格消息网络；网络给出候选速度，编译器将其转为真实轨道自由度与合法晶格参数速度。没有把网络名称称为 DiT，也没有宣称裸网络严格 SO(3) 等变。

## 2. 与三个参考工作的比较

| 参考对象 | 参数 | 计数边界 |
|---|---:|---|
| CrystalFlow CSPNet | 15,449,600 | 官方 512 / 6 层 / 128 频数；运行时 time latent=256；包含注册的条件适配器 |
| CrystalFlow，无 cemb 时参与计算的模块 | 12,294,656 | 条件适配器在 `cemb=None` 分支未调用 |
| CrysVCD 几何 CSPNet | 12,425,728 | 512 / 6 层 / 128 频数；time+property latent=512；不含可另载入的 guide decoder |
| CrysVCD ionic 前端 | 669,880 | GPT2 128 / 3 层 / 4 heads；含元素、计数、电子组态与输出头；不含可选性质嵌入 |
| CrysVCD alloy 前端 | 629,840 | 相同 GPT2 尺寸，不同元素词表 |
| 本地 CGDiT CSPNet | 12,335,716 | 512 / 6 层 / 128 频数；latent=256；包含元素输出头 |
| 本地 CGDiT，形成能+带隙条件核心 | 12,468,836 | CSPNet + 两个标量性质 MLP + null embeddings；不含可选多模态、RL 或预测器 |

参考源码：

- [CrystalFlow decoder 配置](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/conf/model/decoder/cspnet.yaml)、[CSPNet](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/diffcsp/pl_modules/cspnet.py)、[运行时 latent](https://github.com/ixsluo/CrystalFlow/blob/9c25ff0245d787efd87c9d1d797a8968443608cc/diffcsp/pl_modules/diffusion.py)。
- [CrysVCD 几何初始化](https://github.com/vipandyc/CrysVCD/blob/main/CRYGEN_model/diffusion.py)、[GPT2 初始化](https://github.com/vipandyc/CrysVCD/blob/main/formula_gen/GPT2_chem.py)、[前端配置](https://github.com/vipandyc/CrysVCD/blob/main/formula_gen/config/GPT2_chem_ionic.yaml)、[论文](https://arxiv.org/abs/2507.19799)。
- 本地 `cgdit/pl_modules/decoder/cspnet.py`、`conditional_embedding_utils.py` 与 `conf/model/experiments/exp_mp20_fe_bg.yaml`。

审计脚本 `scripts/parameter_comparison.py` 保留构造函数进行几何模块计数，不导入 PyG/scatter 或执行 forward。GPT2 数字按官方构造定义解析计数：每个标准 block 为 `12h²+13h`，另加词表/位置嵌入、末端 LN、自定义组成嵌入和独立输出头。参考文件 SHA256 和计数在 `artifacts/pretrain_validation/reference_parameters.json`；下载来源记录在 `artifacts/reference/manifest.json`。

CrysVCD 默认单几何模型加一个前端约为 1,306–1,310 万参数，再加所用性质嵌入。代码允许额外加载 `guide_decoder`，此时还需计入第二个几何网络；这不能混入“单模型”数字。CFG 在 UnifiedGroupGen 中复用同一套权重，不增加一套模型参数。

## 3. 原模型是否太小

原描述 Transformer 与 CrysVCD 前端属于相同规模；原几何网络只有约 63 万参数，比上述几何后端小约 19–20 倍。它适合验证约束构造与数据链路，不能直接作为容量公平的主结果配置。

正式容量对照建议以 large 为主、medium 为成本消融，small 保留作实现诊断。large 几何后端已有 1,405 万参数，处于参考模型同一量级，不需要仅因“统一群”继续盲目增大。增加 hidden、层数或频数后的效果仍须通过相同划分、容差、训练预算、种子和采样预算验证；参数更多不证明创新或性能更好。

## 4. 实测成本

设备 RTX 3060 12GB，PyTorch 2.6.0+cu118，BF16，AdamW，激活重计算开启。

| 检查 | batch / 展开原子数 | 每优化步 | 峰值 CUDA allocated |
|---|---|---:|---:|
| medium，8 个真实 MP20 结构 | 8 / 16–28 | 0.263 s | 145.4 MiB |
| large，同一批结构 | 8 / 16–28 | 0.313 s | 357.9 MiB |
| large，合成高重数压力测试 | 1 / 416 | 1.217 s | 2.03 GiB |

416 原子例子是 SG225 的两个 192l 轨道加一个 32f 轨道，检验完整图、编译、混合精度和反向传播，不是合理材料。该次冷启动步骤约 13.16 s，包含首次符号掩码搜索与 CUDA 初始化。

加入合法性搜索缓存和必要条件剪枝后的复测分别保存为 `benchmark_large_final.json` 与 `benchmark_large_416_final.json`：真实 MP20 batch 的优化步约 0.207 s，冷步骤约 0.818 s，显存相同。测量会受同时进行的 CPU 数据转换影响，因此保留所有测量而不只选择最快一次。

416 原子复测为每步 1.185 s，冷步骤 1.771 s，峰值仍为 2.03 GiB；符号搜索优化显著减少该例子的首次掩码计算成本，但不能据此推断全数据训练同比例加速。

这些短步测量不能等同正式整轮速度。重复使用同一批数据会复用合法性缓存；全数据首轮和不同原子数分布会增加开销。`epochs: 1000` 是上限，早停 patience=50；总收敛时间尚不能确定。实际日志逐轮记录时间、显存、梯度、LR 和分项损失。

正式环境补充：现有 conda cgdit，torch 2.7.1、RTX 3060、BF16，large 的真实 MP20 batch32/64 短步测试分别为 0.565 / 1.100 s，峰值 allocated 456 / 686 MiB。证据是 `benchmark_cgdit_batch32.json` / `benchmark_cgdit_batch64.json`。这些与早期不同环境的 batch8 结果分开记录。正式 local 配置采用 batch64，并已启动全数据训练；实际整轮时间与收敛轮数以运行日志为准。

首个完整 MP20 epoch 实测：27,136 train / 9,047 val，425 个训练 batch，训练部分 603.65 s、包含完整验证共 793.22 s（13.22 min），峰值 allocated 2,612,113,408 bytes（2.43 GiB）。首轮验证总损失 27.8097，last/best 检查点均已保存。训练仍在继续，首轮损失不能作为收敛结果。按该轮速度线性估算，100 轮约 22 小时、1000 轮约 9.2 天，仅代表 MP20 本机成本外推；后续缓存、负载、早停和 MPTS52 原子分布会改变耗时。

本轮检查时已完成的前十二轮（epoch 0–11）验证总损失为 27.8097、27.5788、25.6582、26.6212、24.6168、23.7175、23.9533、23.5295、23.7393、22.9396、23.9867、23.2686；前八轮含验证约 793–798 s，优化后四轮752.56 / 747.79 / 751.17 / 773.53 s。当前最佳为第十轮，后续训练仍在运行；单轮波动不能证明收敛。共享模型尚未开始训练，其成本不能直接沿用单 MP20 数字。

计算优化没有增加或删减参数：批量物理损失复用已展开的晶格，先验从CPU固定掩码取得周期自由度数量。5组交替BF16优化步中位数为原实现1.201 s、优化实现0.784 s；共享GPU负载下部分组反向波动，该局部测量不代表整轮训练成本降低34.7%。证据为 `performance_paired_mp20.json`，等价性覆盖CPU/CUDA的损失、梯度和RNG。MP20已从epoch7完整检查点续训并加载优化；首次25步损失与旧运行完全相同，耗时42.32 s与旧43.58 s接近，优化后首次完整训练+验证752.56 s，相比旧八轮中位数796.33 s，单轮观察耗时减少约5.50%；轮次数据顺序和系统负载不同，未作严格重复因果测速。

复测（从 UnifiedGroupGen 目录）：

```powershell
python scripts/benchmark.py --config configs/mp20_large.yaml --records artifacts/full/mp_20/train.jsonl --batch-size 8 --steps 3 --output artifacts/bench_large.json
python scripts/benchmark.py --config configs/mpts52_large.yaml --descriptor examples/stress_416.json --batch-size 1 --steps 3 --output artifacts/bench_416.json
python -m unifiedgroupgen.cli parameters --config configs/mp20_large.yaml
```

优化后完整epoch8的模型与AdamW状态全部tensor有限，CUDA RNG存为CPU uint8，配置与完整训练/验证SHA256均与受控续训前一致。完整证据 `artifacts/pretrain_validation/optimized_full_epoch.json`。当前共九轮，最佳23.5295，尚未收敛，不提前计算测试损失。

优化后连续两轮的整轮中位数750.18 s，相比旧八轮796.33 s，观察耗时减少约5.80%，并非严格因果测速。epoch9 last/best检查点逐tensor一致且权重有限，最佳验证22.9396；仍在完整训练。证据 `artifacts/pretrain_validation/optimized_epoch_series.json`。

后续常量GPU张量复用候选已撤回：原型5组中位数显示局部改善，但正式候选在相同并行负载协议下方向相反，不能据此断言提速。生产tensor转换恢复原实现，训练未重启；候选一致性测试曾通过，当时最终源码完整77项测试再次通过。候选仅留在诊断脚本，证据 `constant_cache_decision.json`。此前晶格复用与先验CPU固定计数优化仍保持。
