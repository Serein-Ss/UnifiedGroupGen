# 本地配对测试期间启动4090正式训练

本地RTX3060继续执行三组优化前后各30轮的配对检查；服务器可独立开始修复数据上的新模型。
当前加速内核的输出与梯度检查已通过，完整配对训练及生成质量比较仍在进行。
服务器使用 `train_main15.py`，不需要运行本地的 `paired_training_study.py`。

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

## 3. 先启动三个基础模型

先确认服务器没有另一个本项目训练队列占用GPU：

```bash
ps -ww -eo pid,ppid,etime,args | grep -E 'train_main15.py|unifiedgroupgen.cli.*train'
nvidia-smi
python scripts/train_main15.py --models mp20_u c2db51_u joint_u --plan

mkdir -p artifacts/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py \
  --models mp20_u c2db51_u joint_u --device cuda --resume \
  > artifacts/runs/main15_base.driver.log 2>&1 &
echo $!
```

默认激活检查点开启、边块4096、batch64、BF16、每5轮验证，保持正式配置。
每套最多1000轮，允许原策略早停。`--resume`仅恢复此新队列中同配置、同划分的检查点；
没有新检查点时从头训练。输出位于 `artifacts/runs/main15_seed42/`。

```bash
tail -n 20 -F artifacts/runs/main15_base.driver.log
tail -n 10 -F artifacts/runs/main15_seed42/mp20_u.log
watch -n 5 python -m json.tool artifacts/runs/main15_seed42/mp20_u/model.status.json
```

第一个任务结束后，详细日志切换为 `c2db51_u.log`、`joint_u.log`。
Ctrl+C退出看日志不停止后台训练。

## 4. 后续十二套性质模型

三个基础模型队列结束后，再启动下面这一条。不要同时运行两个同seed队列。

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_main15.py --device cuda --resume \
  > artifacts/runs/main15_remaining.driver.log 2>&1 &
echo $!
```

它会跳过已完成的三个基础模型，再按清单完成其余十二套。
若希望一次排入全部15套，可直接使用本节命令替代第3节的启动命令，不能两条一起执行。
旧锁只在确认对应队列和训练子进程均退出后处理，不删除权重或数据。

本地配对结果完成后再比较是否存在明显退化；本次服务器启动不等于质量结论已经成立。
测速及运行选项见 [TRAINING_ACCELERATION.md](TRAINING_ACCELERATION.md)，
完整任务与评估见 [MAIN15_TRAINING_AND_TASKS.md](MAIN15_TRAINING_AND_TASKS.md)。
