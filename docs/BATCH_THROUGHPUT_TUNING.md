# 4090 批量与边预算测速

目标是提高真实训练吞吐量，不设显存占用必须达到的比例。不启动 main15 正式训练。

## 执行顺序

1. 等 `artifacts/loss_balance_4090_seed42/status.json` 的 `state` 为 `complete`；保留这次实验的 batch size 64 和冻结源代码。
2. 在 GPU 0 空闲时运行 `scripts/benchmark_batch_capacity.py --run`。脚本会检查上述状态和 GPU 计算进程，不停止任何进程。
3. 下载脚本生成的 ZIP，分析吞吐量、显存和边预算，选出三种数据设置的候选运行参数。
4. 再分别开展 5 轮试跑。验证计时、损失、梯度和完整轮次峰值后，才决定正式训练参数。

## 测速内容

- MP-20 / 联合：batch size 64、96、128、160、192、256。
- C2DB：64、128、256、512、768、1024。
- 固定 4 个 CPU 线程、bf16（沿用配置）、关闭 activation checkpointing、edge chunk 32768；保持架构和学习率不变。
- 每种设置固定随机抽取最多 4096 条真实 TRAIN 记录。`--full-pass` 在预热之后将这些记录全部处理一次，避免不同批量只测到不同规模的前几个批次。包括前向、反向、梯度裁剪和优化器更新，不保存权重。
- 每个候选另从完整 TRAIN 构造批次，测试边数最大的 6 批。它检查高边数容量，但不保证覆盖所有可能的显存极值；完整 5 轮试跑仍然必要。
- 先测 edge budget 200000。出现多个不足请求大小的批次（尾批最多一个），或不足请求大小的批次达到边预算的 90% 时，对该候选追加 400000 预算；只对通过原预算压力测试的候选追加。
- OOM 会记录并继续其他候选；其他错误停止流程，避免把数据、依赖或数值错误误判成容量不足。
- 没有命令启动 main15。新脚本默认仅打印计划，执行必须显式提供 `--run`。

原始审计配置保持不变，边预算由运行时参数覆盖。正常训练和 checkpoint 的 `runtime_options` 记录并恢复边预算；不跳过数据门禁。

## 服务器命令

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0

UGG_TUNE_DIR="artifacts/benchmarks/batch_throughput_$(date +%Y%m%d_%H%M%S)"
python scripts/benchmark_batch_capacity.py --output "$UGG_TUNE_DIR"
nohup python -u scripts/benchmark_batch_capacity.py \
  --output "$UGG_TUNE_DIR" --run > "${UGG_TUNE_DIR}.driver.log" 2>&1 &
echo "PID: $!"
echo "日志: ${UGG_TUNE_DIR}.driver.log"
echo "完成后的结果包: ${UGG_TUNE_DIR}.zip"
```

父目录 `artifacts/benchmarks` 若尚不存在，先创建。完成标志为 `[COMPLETE] benchmarks only` 和 `[AUDIT PACKAGE]`。

`summary.csv` / `summary.json` 会随每个完成的候选更新，记录请求批量、实际平均批量、样本/秒、秒/步、压力批次最大边数、边预算限制及两种批次的峰值分配/保留显存。每个子任务还保存原始 JSON、日志和实际命令。ZIP 不含固定采样池的训练记录，不含模型权重。

## 如何选择

1. 排除随机吞吐测试或高边数压力测试 OOM 的候选。
2. 优先比较固定采样池的 `samples_per_second`，不以每步时间或显存占用排序。
3. 若多个候选吞吐量接近，保留较低显存峰值的候选，并对前两名用另一随机池/重复测试确认差异是否超过波动。
4. 对 `edge_limited=true` 的候选同时检查预算 200000/400000 下的实际批量、吞吐量和压力显存；请求批量增加不等于有效批量增加。
5. 使用候选批量跑满 5 轮，确认完整数据加载、验证、保存的总耗时。批量改变会改变每轮优化器更新次数，因此不能保证损失轨迹或最终质量完全一致，也不自动线性放大学习率。

## 5 轮试跑入口（参数待测速后选择）

`scripts/train_timing_pilot.py` 支持 `--batch-size` 和 `--edge-budget`，只运行所选的一个模型，固定 5 轮，第 5 轮验证。先用 `--plan` 核对命令；输出只能在 `artifacts/runtime_pilots` 下的新目录。

每个试跑保留 `model.history.jsonl`、`timing.json`、`timing.csv` 和 `pilot.json`。计时区分训练、验证、保存和其他耗时；保留第一轮冷启动，另报告第 2–4 轮训练中位数。描述、晶格、坐标损失保留在完整 history 中。当前损失配对评估未完成前，不在此文档预选正式损失配置或正式批量。

## 本地验证（2026-10-24）

Torch 2.2.1+cpu / PyXtal 1.1.3 隔离环境中，相关 14 项测试通过。包含一次真实 CPU 前向、反向与更新的完整采样池测速，运行时边预算保存/恢复后权重逐项一致，5 轮计时入口参数转发与验证周期检查，以及测速控制流程的模拟运行和归档检查。未启动本地或远程正式训练，未获得这批新候选在 4090 上的吞吐量或最终生成质量结论。
