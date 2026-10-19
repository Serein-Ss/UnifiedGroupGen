# 已停止训练后的联合模型评估

本流程直接读取 `artifacts/runs/joint_space_layer_large_seed42/model.best.pt`，不启动训练。
2026-10-17 下载快照中的最佳检查点来自第195轮；脚本以实际权重内的 epoch 和 SHA256 为准。
测试集沿用该检查点审计通过的完整联合划分，结果包含三来源各自的损失和元素覆盖分母。

## 1. 更新代码和检查环境

在服务器现有项目中执行：

```bash
conda activate cgdit
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
git status --short
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -c "import torch; print('torch:', torch.__version__, 'CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
python scripts/verify_training_data.py
python -m pytest -q tests/test_joint_evaluation.py tests/test_native_identifier.py tests/test_workflow.py
```

已安装的 PyTorch/CUDA 不需要重装。数据已经解包且通过检查时，不重复解包，也不覆盖原有模型。
若 `git pull --ff-only` 报本地改动冲突，保留输出并处理冲突，不执行 `reset --hard`。
新测试中的完整流程使用很小的随机权重和 CPU，只验证程序链路；不代替正式模型的科学评估。

## 2. 查看执行计划

```bash
python scripts/evaluate_joint.py --plan
```

七项任务依次执行：完整测试集，MP20无条件/条件，MPTS52无条件/条件，C2DB无条件/条件。
每组默认256次尝试，六组合计1536次；失败保留在分母中，不通过重采样凑足成功数。
每次连续生成64步 Heun；条件 CFG=2，无条件 CFG=0。
三类任务分别固定来源域，故“无条件”是来源域和材料类别给定下的性质无条件。

演示条件：MP20/C2DB 为形成能−1 eV/atom、带隙1 eV；MPTS52为形成能−1 eV/atom、凸包能0 eV/atom。
MPTS52无带隙训练标签，因此不请求带隙条件。输入条件并不等于输出已经达到该性质。
这些演示值不是扫描过的最优目标，也不是性质命中率实验。

## 3. 后台执行

确认旧训练确实已停止，再执行一次：

```bash
mkdir -p artifacts/evaluations
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_joint.py \
  --device cuda --count 256 --steps 64 \
  --output artifacts/evaluations/joint_best_seed142 \
  > artifacts/evaluations/joint_best_seed142.driver.log 2>&1 &
echo $!
```

这是评估作业，不会增加训练 epoch。输出与旧训练目录分离。
若先检查小规模效果，可用 `--count 8` 并指定不同目录 `--output artifacts/evaluations/joint_best_smoke8`；
此时仍评估完整测试集，生成48次。正式1536次评估应使用正式目录，不复用不同参数的试跑目录。

## 4. 查看进度

```bash
tail -n 20 -F artifacts/evaluations/joint_best_seed142.driver.log
```

总状态：

```bash
watch -n 5 'python -m json.tool artifacts/evaluations/joint_best_seed142/pipeline.json'
```

更简洁地显示当前任务及其内部进度：

```bash
python - <<'PY'
import json
from pathlib import Path
root = Path('artifacts/evaluations/joint_best_seed142')
s = json.loads((root/'pipeline.json').read_text())
print('state:', s['state'], 'phase:', s.get('phase'), 'epoch_index:', s['checkpoint_epoch'])
for name, job in s['jobs'].items():
    folder = root/job['directory']
    print(name, job['state'], folder)
    progress = folder/'sampling.status.json'
    if progress.exists():
        print(json.loads(progress.read_text()))
    elif job['state'] == 'running':
        log = folder/'stdout.log'
        if log.exists():
            print('\n'.join(log.read_text().splitlines()[-3:]))
PY
```

测试每25批输出进度，采样每10次输出进度、每次更新状态。首次读取检查点和校验文件时，状态文件可能尚未出现。
`Ctrl+C` 退出 `tail/watch` 不会停止后台作业。

## 5. 断点与失败处理

正常失败后检查对应任务 `stderr.log`。再次执行相同命令并添加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_joint.py \
  --device cuda --count 256 --steps 64 --resume \
  --output artifacts/evaluations/joint_best_seed142 \
  >> artifacts/evaluations/joint_best_seed142.driver.log 2>&1 &
```

已完成任务只有在输出文件哈希一致时才复用；未完成任务在 `attempt_002` 等新目录重新执行，旧结果保留。
这是任务级恢复，不是从单个样本续算；同一任务的尝试次数不会跨重试合并。
恢复要求权重、测试数据、执行代码、主要依赖版本及参数保持一致，否则使用新输出目录。
强制终止可能留下锁。先检查 `pipeline.json` 中的父/子 PID 和 `ps`，确认相关进程都停止，再清除本评估的锁：

```bash
ps -ww -eo pid,ppid,etime,args | grep -E 'evaluate_joint|evaluate_checkpoint|unifiedgroupgen.cli.*sample'
# 仅在确认上述本评估进程全部退出之后执行：
rm -f artifacts/evaluations/joint_best_seed142/pipeline.lock
```

不要在评估运行时更新代码或覆盖检查点。

## 6. 输出与回传

- `pipeline.json`：状态、每项任务的实际目录、命令、代码/权重/数据指纹、依赖版本。
- `summary.json`：完整测试报告与六组生成汇总。
- `summary.csv`：各组尝试、生成、失败、群识别未知、精确群匹配、几何检查计数。
- `test/attempt_001/result.json`：完整测试集及各来源的损失/覆盖。
- `<来源>_<模式>/attempt_001/`：CIF、逐条 `samples.jsonl`、采样状态、汇总及日志。

恢复后有效目录以 `pipeline.json` 为准，不固定假定 `attempt_001`。
`state=complete` 说明七项任务执行和结果汇总完成，不表示所有生成结构都有效。
精确群识别未知、失败尝试与化学/稳定性/性质未检验分别保留，不能把未检验当作不合格或合格。
几何联合计数要求群闭合、有限度量残差、0.6 Å距离诊断、轨道重数规则、无轨道重合；不是化学或稳定性认证。
当前距离检查沿用旧版原子对距离口径；不足以替代按元素半径和全部周期近邻的后续物理筛查。

完成后在项目根目录打包（包含结果、CIF与日志，不重复打包训练数据和权重）：

```bash
tar -czf artifacts/evaluations/joint_best_seed142_results.tar.gz \
  artifacts/evaluations/joint_best_seed142 \
  artifacts/evaluations/joint_best_seed142.driver.log
```

下一阶段用这些结果决定是否需要修复生成质量，再开展化学筛查、去重、新颖性、独立性质评估和匹配划分的对照。


> 2026-10-19结果同步：联合训练已在245轮早停，49次验证；最优仍为第195轮。新包最优权重、七组评估输出及数据划分已核验一致。当前结果以 `assets/results/current_snapshot.json` 为准，完整更新清单见 `docs/RESULTS_UPDATE_20261019.md`。本次未新增性质或稳定性认证。
