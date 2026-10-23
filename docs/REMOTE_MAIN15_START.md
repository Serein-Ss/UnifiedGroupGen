# 4090先测速，再确定单模型训练参数

本地RTX3060继续执行三组优化前后各30轮的配对检查。服务器当前只开展运行参数测速，
先确定候选参数，再确认单个模型的整轮耗时；15套模型暂不排入训练队列。
当前加速内核的输出与梯度检查已通过，完整配对训练及生成质量比较仍在进行。
服务器此阶段使用 `benchmark.py`，不需要运行本地的 `paired_training_study.py`。

## 一键执行完整12组测速

在服务器激活cgdit、更新代码后可直接执行：

```bash
conda activate cgdit
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
git pull --ff-only
mkdir -p artifacts/benchmarks
nohup bash scripts/benchmark_4090_grid.sh \
  > artifacts/benchmarks/4090_grid.driver.log 2>&1 &
echo $!
tail -n 30 -F artifacts/benchmarks/4090_grid.driver.log
```

该脚本包括下文环境检查、修复数据导入与本机审计、12个运行参数测速和结果汇总。
沿用现有PyTorch，只安装项目editable元数据及固定PyXtal；GPU0已有计算进程时退出，
不会自动停止其他任务。每次生成新的时间戳目录，结束后输出其绝对路径，保存
`summary.csv`、`summary.json`、`status.tsv`、12组原始结果/失败日志。
汇总包括平均/中位步耗时、样本吞吐、分配/保留显存与相对默认设置的加速比，
并核对同一数据设置的批次、模型配置、精度和数据哈希一致。
最快候选只供下一阶段压力与整轮复测，不自动启动训练。
下文分步命令供手动执行与核对；选择一键脚本时不用重复执行。

## 1. 更新代码与环境

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git status --short
git pull --ff-only
git log -2 --oneline

export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"

python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -c "import sys, torch; from importlib.metadata import version; print(sys.version); print('torch', torch.__version__); print('pyxtal', version('pyxtal')); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
```

继续使用cgdit中的CUDA版PyTorch，不运行整份requirements安装，不更换PyTorch。
PyXtal必须是1.1.3。若上面的版本输出不同，仅更新这一项：

```bash
python -m pip install --no-deps 'pyxtal==1.1.3'
```

## 2. 导入修复数据并生成本机门禁

已有部署通常已经导入基础数据包；先导入修订包并在服务器本机生成15套配置审计：

```bash
python scripts/unpack_training_data.py --bundle datasets/source_integrity_v1
python scripts/prepare_main15.py
python scripts/prepare_main15.py --verify-only
python -m pytest -q
```

全新部署在上面之前额外执行一次 `python scripts/unpack_training_data.py`。
不要使用旧的三来源联合数据或旧检查点恢复这些新实验。
如果前述步骤报错，应先解决错误再执行启动命令。

修复数据的TRAIN/VAL/TEST应分别为：

| 配置来源 | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| MP20 | 27136 | 9047 | 9046 |
| C2DB-51 | 10143 | 3381 | 3381 |
| 两者联合 | 37279 | 12428 | 12427 |

可直接打印15套实验的关键条件配置：

```bash
python - <<'PY'
from pathlib import Path
from unifiedgroupgen.main_experiments import configurations
for name, c in configurations(Path.cwd()).items():
    targets = [k for k, v in c['properties'].items() if not v.get('context')]
    assert c['data_revision'] == 'source_integrity_v1'
    assert c['composition_probability'] == c['observation_probability'] == 0
    print(name, 'revision=', c['data_revision'], 'targets=', targets,
          'dropout=', c['cond_dropout'], 'subset=', c['property_subset_training'])
PY
```

`u`无目标性质；`ef/eg/eh`分别是形成能、带隙、凸包能单性质；`all`是三性质CFG及子集训练。
两来源联合模型保留来源上下文，仍不属于目标性质条件。

## 3. 当前只执行运行参数测速

先确认服务器没有另一个本项目训练队列占用GPU：

```bash
ps -ww -eo pid,ppid,etime,args | grep -E 'train_main15.py|unifiedgroupgen.cli.*train'
nvidia-smi
```

如已经执行了先前的正式训练命令，需要先依据实际PID停止该队列及训练子进程，再测速。
本地3060的配对测试继续；它不会占用远程4090。

先固定batch64、BF16、相同真实数据批次、初始种子、模型和群约束，仅比较：

| 候选 | 激活重计算 | 边块大小 |
|---|---|---:|
| A | 开启 | 4096 |
| B | 关闭 | 4096 |
| C | 关闭 | 16384 |
| D | 关闭 | 32768 |

对MP20、C2DB与联合的三性质代表配置各测四项，共12个短测速。
每项6个真实批次、6步预热、24步计时，包含前向、反向、梯度裁剪和AdamW更新，
只在内存中运行，不保存模型权重，不启动正式训练。
三性质模型用于参数筛选，不代表其性能已优于其他条件配置。

```bash
export UGG_TUNE_DIR="artifacts/benchmarks/4090_tuning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$UGG_TUNE_DIR"

for tune_model in mp20_all c2db51_all joint_all; do
  for tune_setting in checkpoint:4096 no-checkpoint:4096 no-checkpoint:16384 no-checkpoint:32768; do
    tune_mode=${tune_setting%:*}
    tune_chunk=${tune_setting#*:}
    tune_path="$UGG_TUNE_DIR/${tune_model}_${tune_mode}_${tune_chunk}"
    echo "[START] $tune_model $tune_setting"
    if CUDA_VISIBLE_DEVICES=0 python scripts/benchmark.py \
      --config "configs/main15/${tune_model}.yaml" --device cuda \
      --seed 42 --batch-size 64 --threads 4 \
      --selection random --batches 6 --warmup 6 --steps 24 \
      "--$tune_mode" --edge-chunk-size "$tune_chunk" \
      --output "$tune_path.json" > "$tune_path.log" 2>&1; then
      echo "[DONE] $tune_path.json"
    else
      echo "[FAILED] inspect $tune_path.log"
    fi
  done
done
```

同一数据设置四项的 `record_ids`、`atoms`、`batch_sizes` 和配置/划分哈希应一致。
失败日志必须保留，不能把所有失败都认定为显存不足；先查看实际异常。
若基础候选都失败，先修复错误再扩大搜索。

## 4. 打印汇总，确定下一阶段候选

```bash
python - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['UGG_TUNE_DIR'])
print('model | recompute | chunk | median s/step | samples/s | allocated GiB | reserved GiB')
for model in ('mp20_all', 'c2db51_all', 'joint_all'):
    reports = []
    for mode, chunk in (('checkpoint',4096), ('no-checkpoint',4096),
                        ('no-checkpoint',16384), ('no-checkpoint',32768)):
        path = root/f'{model}_{mode}_{chunk}.json'
        if not path.exists():
            print(model, mode, chunk, 'FAILED / missing result; inspect its log')
            continue
        r = json.loads(path.read_text())
        reports.append(r)
        print(model, r['checkpoint_layers'], r['edge_chunk_size'],
              round(r['median_seconds_per_step'],4), round(r['records_per_second'],2),
              round(r['peak_cuda_bytes']/2**30,2), round(r['peak_cuda_reserved_bytes']/2**30,2), sep=' | ')
    if reports:
        keys = ('record_ids','atoms','batch_sizes','config_sha256','train_split_sha256','val_split_sha256')
        assert all(all(r[k] == reports[0][k] for k in keys) for r in reports), 'Unmatched timing inputs'
print('Results:', root)
PY
```

选择以 `records_per_second` 和实际步耗时为主，显存保留余量；不要以显存占满或某一瞬间
`GPU-Util=100%` 为最优标准。最优是当前候选及负载下的实测最优，不是全局保证。
不同数据设置可使用不同候选，不强求一个设置覆盖全部模型。

下一阶段只对最快的候选及默认设置倒序复测，并测试高边数真实批次 `--selection largest`，
检查是否OOM、损失/梯度非有限或吞吐明显下降。短测留出数GB显存余量仍不能保证全数据绝不OOM。
然后比较CPU线程1/2/4/8，控制其他参数相同；每次启动前同步设置相应OMP/MKL线程与`--threads`。
结果确认后仅对单个模型做5轮整轮试跑，覆盖一次验证、权重保存、最大实际图和缓存建立。
比较稳定训练轮及完整5轮窗口的实际墙钟时间，再确定正式运行设置。

batch64和精度先保持不变。增大batch会改变每轮更新次数及优化统计，若要调整需单独比较学习率
和有效batch等训练定义，不能把它当作此次等价运行优化自动套用到15套配置。
本阶段的命令不会启动任何正式训练；将上述汇总与失败日志用于下一步选择即可。

本地配对结果完成后再比较是否存在明显退化；GPU测速不等于生成质量结论已经成立。
测速及运行选项见 [TRAINING_ACCELERATION.md](TRAINING_ACCELERATION.md)，
完整任务与评估见 [MAIN15_TRAINING_AND_TASKS.md](MAIN15_TRAINING_AND_TASKS.md)。
