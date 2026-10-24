# 旧M1/M2/M3：分模块测试、十组4096次生成与外部性质评价

本任务只评价历史模型，不训练或恢复任何生成模型。新15套训练继续运行；历史结果仍属于修订前数据实验。

## 1. 评价层级与预算

1. 测试集：M1/M2/M3分别使用其检查点对应的完整审计测试文件。描述Transformer报告整体损失、G/W/Z/STOP交叉熵及Top-1、真实前缀下整序列准确率。OrbitFlow输入真实描述，报告晶格、原始坐标、按检查点尺度归一化的坐标误差与流损失。这是单次随机路径/时间的速度预测评价，不是结构重建匹配率；性质条件为已有标签，不指定组成或观测。
2. 生成：M1两组、M2两组、M3三个来源各两组，共十组。每组4096次尝试，合计40960次。失败保留分母；不改变64步Heun、seed142或CFG设置。
3. 外部性质：使用前期CGDiT的冻结FE/BG预测器，默认seed123。每个生成结构及相应测试参考均由同一预测器推理；报告有限预测数量、均值/标准差/分位数、同预测器参考分布Wasserstein、目标MAE与命中率，并只输出PNG累计分布图。
4. 标准质量：原有精确群、几何、SMACT、COV、VUN和二维适配指标。额外报告“联合有效且性质命中”的全尝试产率。

| 模型 | 来源 | 无性质目标 | 多性质目标 |
| --- | --- | --- | --- |
| M1 | MP20 | CFG0，4096次 | 形成能−1 eV/atom、带隙1 eV；CFG2，4096次 |
| M2 | MPTS52 | CFG0，4096次 | 形成能−1 eV/atom、凸包上能量0；CFG2，4096次 |
| M3 | MP20 | 同上 | 形成能−1、带隙1；source_domain=0 |
| M3 | MPTS52 | 同上 | 形成能−1、凸包上能量0；source_domain=1 |
| M3 | C2DB | 同上 | 形成能−1、带隙1；source_domain=2 |

每组同时计算FE和BG分布，即使生成时没有指定该性质。仅有FE/BG预测器，不能评价凸包能命中或稳定率；请求包含凸包能时，完整联合目标命中率保留未知。命中阈值沿用前期：FE 0.06 eV/atom、BG 0.45 eV，元数据显式记录，不混用原评估默认阈值。

## 2. 预测器适用边界

- 加载实际最终测试对应的checkpoint；多个checkpoint无法由run.log唯一解析时退出，不猜测文件。
- 使用预测器训练时的niggli/primitive/graph_method；反归一化由M3GNetSurrogate.forward执行，不重复拟合或缩放。
- CIF原子序数直接使用真实Z，不使用旧扩散输出的零基类别转换。
- 默认要求已有test_preds.npy与test_targets.npy，并记录两者误差、文件哈希。生成与参考预测均绑定权重、图设置及预测代码摘要。
- 三维MP20训练的预测器在MPTS52/C2DB及未弛豫结构上属于外推；对应参考标签误差单独报告。C2DB形成能能量参考、带隙计算方法须另行核对，不能直接把跨域预测命中写成DFT性质命中。这里不提供SUN或稳定性认证。
- 参考分布使用对应生成检查点的测试来源；不直接拿预测器自己的test_preds.npy替代MPTS52/C2DB参考。联合模型测试划分中某些记录可能属于预测器的训练集，因此新增的参考标签误差称为标签一致性，不自动称为预测器独立测试精度。

## 3. 不改正在训练的工作目录：只提取评估工具

已有cgdit环境，沿用现有PyTorch。执行git fetch只更新远程引用，不更新当前训练工作树；将新增脚本提取到独立工具目录。无需git pull、pip install或重启训练。

```bash
conda activate cgdit
# 在UnifiedGroupGen仓库目录执行
export UGG="$PWD"
export CGDIT="$(dirname "$UGG")/test_CGDiT"
export UGG_PRED="$CGDIT/output/newton_4g_6p_20261003/predictors"
export UGG_EVAL_OUT="$UGG/artifacts/evaluations/legacy_4096_cpu_seed142"

git -C "$UGG" fetch origin main
export UGG_EVAL_TOOLS="$UGG/artifacts/evaluation_tools/$(git -C "$UGG" rev-parse origin/main)"
mkdir -p "$UGG_EVAL_TOOLS" "$(dirname "$UGG_EVAL_OUT")"
git -C "$UGG" archive origin/main scripts/legacy_eval scripts/evaluate_generation.py \
  | tar -x -C "$UGG_EVAL_TOOLS"

export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONIOENCODING=utf-8

python "$UGG_EVAL_TOOLS/scripts/legacy_eval/campaign.py" \
  --project-root "$UGG" --cgdit-root "$CGDIT" \
  --predictor-root "$UGG_PRED" --predictor-seed 123 \
  --output "$UGG_EVAL_OUT" > "$UGG_EVAL_TOOLS/plan.json"

# 只核对真实预测器，不生成结构、不训练
CUDA_VISIBLE_DEVICES='' nice -n 19 python "$UGG_EVAL_TOOLS/scripts/legacy_eval/properties.py" \
  --project-root "$UGG" --cgdit-root "$CGDIT" \
  --predictor-root "$UGG_PRED" --predictor-seed 123 \
  --inventory --output "$UGG_EVAL_TOOLS/predictor_inventory"
```

检查应解析FE/BG的seed123、正确target、训练图设置、缩放参数和最终测试数组。缺少依赖、模型或标签数组时先解决对应问题，不在训练期间覆盖安装整套环境。

## 4. 提交完整CPU评估

上述预测器核验通过后，运行：

```bash
nohup nice -n 19 ionice -c 3 python -u "$UGG_EVAL_TOOLS/scripts/legacy_eval/campaign.py" \
  --project-root "$UGG" --cgdit-root "$CGDIT" \
  --predictor-root "$UGG_PRED" --predictor-seed 123 \
  --output "$UGG_EVAL_OUT" --count 4096 --steps 64 --seed 142 \
  --run --resume > "${UGG_EVAL_OUT}.driver.log" 2>&1 &
echo "评估PID：$!"
echo "评估工具：$UGG_EVAL_TOOLS"
echo "评估日志：${UGG_EVAL_OUT}.driver.log"
```

工具强制CPU、1个PyTorch线程、1个参考/图构建进程；所有步骤串行，先完成三套测试，再执行十组采样、性质及标准评价。nice/ionice降低CPU和磁盘调度优先级，不是硬资源隔离。GPU没有新增评估负载，但CPU/RAM/磁盘仍共享，不能承诺训练墙钟时间完全不变。4096次CPU采样可能很慢，尚无本机实测ETA；如优先保证训练吞吐，可先只跑`--stages test`，训练结束后另行安排GPU生成。默认完整CPU命令会执行全部十组。

同一工具目录、预算、权重和预测器参数下，重复`--run --resume`会跳过完成子任务；中断子任务从该子任务开头重做。不要同时启动两个评估队列。残留pipeline.lock只在确认评估父子进程退出后处理，绝不处理main15的runner.lock。

## 5. 查看结果

```bash
tail -n 30 -F "${UGG_EVAL_OUT}.driver.log"
python -m json.tool "$UGG_EVAL_OUT/pipeline.json"
tail -n 20 -F "$UGG_EVAL_OUT/M1_test.log"
watch -n 2 nvidia-smi
```

每个模型：`M*/test/summary.json`及逐测试记录`records.jsonl`。每个采样组：`samples.jsonl`、成功CIF、`properties/assessments.csv`、同预测器参考CSV、性质summary、PNG分布图；`metrics/summary.json`和逐结构CSV为标准质量与性质联合筛查。每完成一组记录耗时。pipeline的`complete`表示33个子任务结束，`stages_complete`表示选定阶段结束，不能混同完整结束。

## 6. 本地验证

PyTorch2.2.1 CPU与原标准指标联合回归：17 passed，7 warnings，46.89秒；随后补充冻结预测器清单验证，新增工具定向测试8 passed。覆盖分token损失与原损失一致、固定描述流损失尺度、失败/未知分母、串行恢复、前期性质阈值传入及原标准指标。另用两条真实MP20测试结构和本地已有真实FE/BG预测器完成CPU推理、物理单位与PNG接口验证；这不替代远程指定权重的核验，也不是4096次正式实验结果。未修改训练核心或15套配置，未在本地或服务器启动正式训练。
