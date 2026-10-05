# 训练速度优化与远程续训

## 本次修改

- 性质、组成与 CFG 掩码批量编码，避免逐样本读取 CUDA 标量来决定 Python 分支。
- 固定独立参数的仿射约束用索引赋值与零速度掩码实现；一般线性组合仍用通用伪逆。
- 晶格矩阵指数、Cholesky 和具有物理度量的坐标速度投影批量计算。
- 二维晶格嵌入三维时保留单位法向；坐标投影的填充维度用独立单位方程隔离，不改变有效自由度。
- 编译后的群轨道常量按设备、类型缓存；Wyckoff 数组缓存。编译器容量从 4,096 增至 32,768。
- 合法 token 掩码缓存满时淘汰最早项，不再清空全部缓存；容量从 2,048 增至 32,768。
- 训练状态与描述 token 在 CPU 组装，再集中传输；状态有限性验证仍保留。

完整数据、模型参数、损失定义、性质缺失规则、CFG 条件与约束目标保留。
浮点运算顺序和训练随机数的消费方式改变，不保证与旧代码产生逐位相同的训练轨迹。
优化版内部的连续训练与断点续训一致性另有测试。
扩大缓存会增加主存和显存占用；完整一轮的 `peak_cuda_bytes` 比单批短测更有参考价值。

## 本机证据与边界

RTX 3060、PyTorch 2.7.1、BF16、同一批 64 个 MP20 结构，保持 activation checkpointing=true、edge_chunk_size=4096。
旧版与优化版交替执行 8 组训练步，包含前向、反向、梯度裁剪和 AdamW：

| 项目 | 旧版 | 优化版 |
|---|---:|---:|
| 每步中位数 | 1.069 秒 | 0.369 秒 |
| 相对速度 | 1 | 2.90 |

另一次单步剖析：CUDA 流同步次数从之前的 2,657 降至 253；优化后没有固定约束伪逆调用，矩阵指数、Cholesky、线性求解各批量调用一次。
这些都是本机短测，测量时另有已有训练进程，不是独占 GPU 的整轮测量，也不是远程 4090 的速度保证。
训练参数仍为 proposal=3,599,482，flow=14,046,220，总计 17,645,702。
高精度几何/梯度、CPU/CUDA、CFG 和旧检查点恢复均有测试；本机 PyTorch 2.2.1 CPU 环境另测兼容性，CUDA 项需在服务器确认。
完整测试：PyTorch 2.7.1 CPU/CUDA 为 104 passed；PyTorch 2.2.1 CPU 为 87 passed、17 CUDA 项跳过。
与旧发布代码直接比较混合空间群/层群/平面群的双精度前向：速度输出最大绝对差 3.89e-16，参数梯度最大绝对差 2.66e-15（评估模式，关闭随机条件丢弃）。
不能据此保证 1,000 epochs 回到 20 小时；应以更新后服务器完整一轮的日志为准。

## 1. 停止旧进程，再更新

运行中的 Python 进程不会自动加载更新后的函数。先查看本项目进程：

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
ps -ww -eo pid,ppid,args | grep -E 'scripts/train_remote.py|scripts/train_local.py|unifiedgroupgen.cli.*train'
```

确认刚保存过完整 epoch 后，使用实际输出的 PID 停止本项目的编排入口、参考入口和训练子进程。
例如 `kill -TERM 编排PID 参考入口PID 训练PID`，必须替换为当前 PID；不要使用旧日志中的 PID，也不要停止其他项目。
尚未保存的当前 epoch 会丢失，续训从最近完整 epoch 的 `model.pt` 恢复。
`tail -f` 的 Ctrl+C 只停止看日志，不停止后台训练。

再次用上述 `ps` 确认本项目训练进程全部退出后，清除旧锁并更新：

```bash
rm -f artifacts/runs/remote_pipeline/pipeline.lock artifacts/runs/local_large_seed42/runner.lock
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -m pytest -q
```

继续使用服务器 cgdit 环境，不需要更换 PyTorch 或重新解包数据。

## 2. 在空闲 GPU 上比较两个设置

同一随机选取的 64 个真实结构，固定种子、两步预热、十步计时：

```bash
python scripts/benchmark.py --config configs/mp20_local.yaml --records artifacts/full/mp_20/train.jsonl --batch-size 64 --shuffle-records --steps 10 --output artifacts/runs/benchmark_default.json --profile artifacts/runs/benchmark_profile.json
python scripts/benchmark.py --config configs/mp20_local.yaml --records artifacts/full/mp_20/train.jsonl --batch-size 64 --shuffle-records --steps 10 --no-checkpoint --edge-chunk-size 32768 --output artifacts/runs/benchmark_fast.json
```

比较 `median_seconds_per_step`、`records_per_second` 和 `peak_cuda_bytes`。
关闭 activation checkpointing、增大边块并不保证更快；采用实际更快且显存有余量的设置。
短测没有覆盖每轮最大图。对 MPTS52 和共享模型应分别测其真实记录，并检查最大结构/高边数批次。
默认设置如更快，直接用下一节的普通续训命令。
默认四线程；可用 `--threads 1` 对同一批次另测线程调度开销，不需要修改环境依赖。
脚本兼容 PyTorch 2.2 的 GradScaler 接口；不运行完整训练，也不写训练检查点。

## 3. 从现有检查点恢复

只使用实现优化、保留原计算设置：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage all --resume >> artifacts/runs/remote_driver.log 2>&1 &
```

若上面的快速设置实际更好，可直接续训：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage all --resume --no-checkpoint --edge-chunk-size 32768 >> artifacts/runs/remote_driver.log 2>&1 &
```

两条启动命令只能选择一条。此处 `--checkpoint` 指激活检查点重计算，不是训练权重检查点。
这些运行选项会写入日志和训练检查点的 `runtime_options`，不改原 YAML、模型权重形状或数据哈希。
下次不传运行选项时，恢复检查点保存的运行设置；如要退回原计算设置，显式使用 `--checkpoint --edge-chunk-size 4096`。
不放宽数据、性质归一化或模型配置的原有恢复检查。

## 4. 用整轮确认结果

```bash
tail -F artifacts/runs/local_large_seed42/mp20_train.stdout.log
```

确认 `training_start` 的 `start_epoch` 接续已有检查点，`runtime_options` 符合选择；参数量和数据记录数一致。
观察一个完整 epoch 的 `seconds`、验证损失、梯度范数及 `peak_cuda_bytes`。
如仍明显偏慢，使用 `benchmark_profile.json` 定位剩余开销，同时检查 `nvidia-smi` 和 CPU 占用；不通过删训练样本或跳过验证制造提速。
