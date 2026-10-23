# 15 套实验的训练加速与 4090 测速

## 保持的实验定义

本次只改变等价计算和运行调度。15 份 `configs/main15/*.yaml`、来源数据/划分、性质编码、CFG 丢弃与子集规则、损失权重、模型参数键和形状均保持不变。全原子有向全连接图、780 维边特征和 6 层流网络保留；没有改用截断图、缩小网络或增加控制训练。

1. 消息层同一权重按输入维切分：`W[h_i,h_j,e]+b = W_s h_i + W_t h_j + W_e e+b`。节点投影每层每节点一次，边块仍控制显存。
2. 坐标速度投影仍使用原物理度量。利用轨道间块对角结构，将完整 q 方程改为每轨道至多 3 维方程，批量求解，去掉每批 NumPy 大数组和全 q 填充。
3. Cholesky/solve 使用 `_ex` 返回状态，合并检查，失败仍抛错；几何保持 FP32/FP64。没有把群约束计算降为 BF16。
4. 缓存每条记录的描述、编译图和 CPU 端点；端点维数/有限性启动时检查一次，批次顺序与 RNG 消费保留。描述 token 和合法掩码共同缓存。
5. 无组成/无观测时跳过无用的数组构造，保留编码器偏置及其梯度。训练四种损失和梯度范数留在设备累计，每 25 批集中读取。逐批非有限损失与梯度检查保留。
6. 最新/最优权重仍按原策略保存。历史新增 `checkpoint_seconds` 与 `wall_seconds`；`seconds` 继续表示保存权重前的轮耗时，兼容旧绘图。`wall_seconds` 含权重保存耗时，不含运行前数据准备。

这些变换数学上等价，但改变了浮点累加和 BF16 舍入顺序，不保证旧版与新版逐位相同的训练轨迹。新版内部的连续训练/断点续训另测。旧数据上的 M1/M2/M3 不可作为修复数据上 15 套实验的恢复起点；只有同配置、同划分哈希的新实验检查点才能续训。

## 1. 更新和检查

先检查本项目进程；运行中的进程不会加载新代码：

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
ps -ww -eo pid,ppid,etime,args | grep -E 'train_main15.py|unifiedgroupgen.cli.*train'
```

如仍在训练，等一次完整 epoch 保存后，用实际 PID 停止队列及训练子进程，再重复检查。停止看日志不会停止训练。只有确认本队列父子进程均退出，才可移除遗留的 `artifacts/runs/main15_seed42/runner.lock`；不要删除训练权重或运行目录。

```bash
git status --short
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -c "import torch; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python scripts/prepare_main15.py --verify-only
python -m pytest -q
```

继续使用现有 cgdit/PyTorch，无需升级或另装 CUDA。若新数据尚未准备，先按 `MAIN15_TRAINING_AND_TASKS.md` 导入 `source_integrity_v1` 并完整运行一次 `prepare_main15.py`。

服务器全套测试应执行 CUDA 和 BF16 分支；本地 CPU 跳过这些分支不能替代服务器确认。

## 2. 同批次比较，GPU 上不要同时运行训练

测速现在默认读取配置中的真实 TRAIN 数据，执行相同数据门禁、来源上下文注入、TRAIN 元素域和性质缩放。组成/观测概率、LR、损失权重、梯度裁剪、AMP 都读取配置。`--records` 仅替换计时池，性质缩放仍来自配置中的完整 TRAIN，避免用测试数据拟合。CPU 诊断明确标为 FP32。

`--implementation reference` 使用原来的消息拼接和完整 q 投影；其余数据管线与优化版共享。因此对比只衡量这两项内核，不代表完整旧发布版与新版的总提速。

```bash
mkdir -p artifacts/benchmarks/main15
for model in mp20_all c2db51_all joint_all; do
  python scripts/benchmark.py --config configs/main15/${model}.yaml --device cuda \
    --selection random --batches 4 --warmup 4 --steps 20 \
    --implementation reference \
    --output artifacts/benchmarks/main15/${model}_reference.json || break

  python scripts/benchmark.py --config configs/main15/${model}.yaml --device cuda \
    --selection random --batches 4 --warmup 4 --steps 20 \
    --output artifacts/benchmarks/main15/${model}_optimized.json \
    --profile artifacts/benchmarks/main15/${model}_profile.json || break

  python scripts/benchmark.py --config configs/main15/${model}.yaml --device cuda \
    --selection random --batches 4 --warmup 4 --steps 20 \
    --no-checkpoint --edge-chunk-size 32768 \
    --output artifacts/benchmarks/main15/${model}_fast.json || break
done
```

三个命令默认同 seed，`record_ids`、`atoms`、数据/配置哈希应相同。为了减小温度/后台波动，最好再倒序执行一遍。`--atoms` 或 `--descriptor` 才是显式合成资源测试，不用于代表真实性质条件训练速度。

比较：

- `median_seconds_per_step`：越低越好；同时看所有 `step_seconds`。
- `records_per_second`：不同实际批大小时优先看它，越高越好。
- `peak_cuda_bytes` / `peak_cuda_reserved_bytes`：包含预热及计时批次的峰值分配/保留显存。
- `composition_probability=0`、`observation_probability=0`、`data_revision=source_integrity_v1`。
- `properties` 中目标性质、TRAIN 缩放与联合模型来源上下文；`allowed_elements`、划分哈希。

单批基准未覆盖整个 epoch。额外对候选运行设置检查高边数批次：

```bash
for model in mp20_all c2db51_all joint_all; do
  python scripts/benchmark.py --config configs/main15/${model}.yaml --device cuda \
    --selection largest --batches 4 --warmup 4 --steps 8 \
    --no-checkpoint --edge-chunk-size 32768 \
    --output artifacts/benchmarks/main15/${model}_largest.json || break
done
```

24 GB 卡建议保留数 GB 显存余量。若 OOM 或提速不明显，先测试 `--edge-chunk-size 16384`，或保留默认 `--checkpoint --edge-chunk-size 4096`。测速失败不会保存或覆盖正式权重。不要因短测结果自动更改 15 套配置或 batch size。

## 3. 开始或继续同一批 15 套模型

保留默认运行设置：

```bash
mkdir -p artifacts/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py --resume --device cuda \
  >> artifacts/runs/main15_accelerated.driver.log 2>&1 &
echo $!
```

只有三种数据的候选设置均通过、且实际更快，才选择下面这一条替代：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py --resume --device cuda \
  --no-checkpoint --edge-chunk-size 32768 \
  >> artifacts/runs/main15_accelerated.driver.log 2>&1 &
echo $!
```

两条启动命令选一条。`--no-checkpoint` 只关闭激活重计算，仍保存每轮权重。`--resume` 跳过已完成模型、恢复已中断模型，并从头开始未训练模型。仅想启动部分模型时加 `--models mp20_u c2db51_u joint_u`。可以对不同子队列选择不同运行选项，但同一运行目录同时只允许一个队列。

```bash
tail -F artifacts/runs/main15_accelerated.driver.log
tail -F artifacts/runs/main15_seed42/mp20_u.log
watch -n 5 python -m json.tool artifacts/runs/main15_seed42/mp20_u/model.status.json
```

`training_start.runtime_options` 和权重中的运行选项记录实际设置。默认省略选项时，续训沿用权重保存的设置；要强制退回默认，显式传 `--checkpoint --edge-chunk-size 4096`。15 套配置、优化器和来源哈希的恢复检查不会放宽。

比较完整五轮窗口的 `wall_seconds` 总和（包含一次验证），同时检查损失、梯度范数和显存。计时改善不能替代生成质量验证。

## 本地验证

本次验证结果见 `VALIDATION.md` 的 2026-10-23 条目。PyTorch2.2.1 CPU 完整回归为 158 passed、24 skipped；另在隔离的 PyTorch2.7.1 / RTX3060 CUDA 环境完成 64 项定向回归，覆盖 FP64/FP32/BF16 与断点续训。对比覆盖空间群、层群、平面群、特殊位置、零自由度、观测条件、输出、所有模型参数与输入状态梯度。真实完整宽度权重的 BF16 参数梯度相对 L2 误差约 0.96%，目标值相对差约 0.039%，不承诺逐位相同轨迹。4090/PyTorch2.2.1 仍需服务器复验，没有整轮加速承诺。
