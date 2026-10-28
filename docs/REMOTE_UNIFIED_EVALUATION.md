# 服务器：15套模型统一评估（仅提供命令，不自动启动）

使用既有cgdit环境与修复后数据，不重装PyTorch。旧模型权重和历史评估不覆盖。
调度按已确认15套顺序串行，每个job分别记录采样/代理/指标日志，支持独立断点续跑。

## 更新并检查环境

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git status --short
git pull --ff-only
git log -1 --oneline
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
python -m pip install --no-deps --no-build-isolation -e '.[test,evaluation]'
python -c "import torch,smact,matminer,matplotlib; print(torch.__version__); print(torch.cuda.get_device_name(0))"
python -m pytest tests/test_unified_evaluation.py tests/test_standard_metrics.py -q
nvidia-smi
```

editable安装带 `--no-deps` 不会自动补依赖；如import显示缺包，仅在cgdit环境中安装缺失的评估包，勿替换torch。
运行适配器前会检查checkpoint的TRAIN/VAL审计指纹、当前完整TRAIN/VAL/TEST字节和实验名称。
若检查失败应核对数据与审计，不跳过或重新伪造审计。

## 只生成计划，先确认预算

```bash
python scripts/evaluate_main15_unified.py \
  --checkpoint-root artifacts/runs/main15_seed42 \
  --output artifacts/evaluations/main15_unified_seed142 \
  --count 1024 --csp-cases 256 --csp-k 20 --conditional-count 256 \
  --steps 64 --guidance 2 --seed 142 --threads 4 --plan
```

查看输出 `plan.json` 的任务清单和 `generation_attempts`。本命令不运行模型。
每套模型包含完整测试集的描述Transformer/OrbitFlow模块测试，及每个已训练来源的：

1. 无目标性质DNG，1024次；
2. 仅组成CSP，256案例×20候选；
3. 固定描述诊断，相同256案例×20候选；
4. 每个已训练性质按TRAIN的10/50/90分位数分别执行DNG与CSP；
5. 多性质模型再选取TRAIN真实共现标签元组（按第一性质排序的10/50/90位置）执行联合条件DNG与CSP。

条件CSP仍按256×20计算，`--conditional-count`仅控制条件DNG。条件任务数量多，
完整15套可能需要大量生成预算；计划中的总次数才是最终分母。
按当前修复后数据与本页参数，本地仅计划核对得到235个子任务、655360次生成（未运行）；
服务器必须用实际checkpoint再次生成计划确认。将CSP改为64案例×5候选时，同样全扫描约79360次。
若先做小预算评估，可以另建目录并将案例数改为64、K改为5、条件DNG改为256。
**预算、seed、步数、性质条件或检查点变更时使用新输出目录。**
联合5模型各分MP20/C2DB两个上下文来源，因此一共20个模型-数据域组合。
不会把测试集组成传给DNG，也不会把测试坐标或晶格传给组成CSP。
固定描述诊断明确额外给了真实离散描述。

如需指定群，每个群5次：在计划与实际执行命令中同时添加 `--per-group 5`。
它会覆盖支持kind的全部230 SG或80 LG，而非只抽训练中常见的群，失败同样计入。

## 确认计划后手动启动（本次交付不会执行）

不要同时运行15套训练和GPU评估。脚本发现GPU有计算进程时会拒绝运行，既不停止训练也不抢占显存。
先保留和核验已有训练，再在空闲GPU上执行：

```bash
mkdir -p artifacts/evaluations
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_main15_unified.py \
  --checkpoint-root artifacts/runs/main15_seed42 \
  --output artifacts/evaluations/main15_unified_seed142 \
  --count 1024 --csp-cases 256 --csp-k 20 --conditional-count 256 \
  --steps 64 --guidance 2 --seed 142 --threads 4 \
  --device cuda --run --resume \
  > artifacts/evaluations/main15_unified_seed142.driver.log 2>&1 &
echo "PID=$!"
```

以上先完成结构指标，性质评价保留not_evaluated。要同时使用已有FE/BG代理，在**新目录**的计划与运行命令都加入：

```bash
  --cgdit-root /root/private_data/rszhong/workspace/test_CGDiT \
  --predictor-root /root/private_data/rszhong/workspace/test_CGDiT/output/newton_4g_6p_20261003/predictors \
  --predictor-seed 123
```

默认只在MP20域预测。若需要C2DB探索性分布，另显式加 `--allow-proxy-transfer`；其结果不是已验证的二维性质精度。
代理构图/预测在CPU上执行，不额外占GPU。参考预测跨模型按哈希缓存。
Eh没有独立预测器时保留缺测，FE/BG不会替代它；三性质联合命中因此可能未知。

## 进度与结果

```bash
python scripts/evaluate_main15_unified.py \
  --output artifacts/evaluations/main15_unified_seed142 --status
tail -n 30 -F artifacts/evaluations/main15_unified_seed142.driver.log
watch -n 5 nvidia-smi
```

子目录示例：`mp20_u/mp_20_csp`，内含 `job.json,identity.json,samples.jsonl,sampling.status.json`、CIF与日志；
`metrics/`内为逐样本CSV、逐案例CSV、汇总JSON、PNG和哈希manifest。
`mp20_u/components/`为完整测试集逐记录CSV、模块指标JSON、token/flow PNG。
全部结束后根目录写出 `summary.csv,summary.json,model_comparison.png`，完整任务状态保存在 `pipeline.json`。
组件测试、固定描述CSP、仅组成CSP和DNG均分开标记，不跨任务直接比较同一个“成功率”。

正常中断后，等确认旧主控和子进程已退出，再用**同一条**运行命令的 `--resume`恢复。
意外断电残留 `evaluation.lock` 时，先查锁中PID及命令身份，确认不再有该评估主控后才手动移走锁。
脚本不会自动杀进程、删除锁、截断ledger或丢弃失败样本。

按需打包（先确认state=complete）：

```bash
tar -czf artifacts/evaluations/main15_unified_seed142_audit.tar.gz \
  --exclude=reference_cache --exclude=proxy_reference_cache \
  -C artifacts/evaluations main15_unified_seed142
sha256sum artifacts/evaluations/main15_unified_seed142_audit.tar.gz \
  > artifacts/evaluations/main15_unified_seed142_audit.tar.gz.sha256
```

不需要把训练数据或权重再拷贝进评估压缩包；已保存对应输入/权重哈希。
