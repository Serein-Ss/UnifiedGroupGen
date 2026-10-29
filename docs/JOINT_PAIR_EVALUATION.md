# 两个联合模型：先生成，再评价

模型固定为 `joint_u/model.best.pt`（基础模型）和 `joint_all/model.best.pt`（三性质CFG训练模型）。
复用修复后的MP20和C2DB数据，不重训、不改模型、不运行其他13套模型。
每个模型分别产生space/MP20和layer/C2DB结果；目前仍需source_domain=0/1上下文，不能称为来源无条件生成。

## 任务表

### 第一批 discovery：先取得完整的生成质量和性质控制结果

|任务|模型|如何产生结构|数量|生成后评价|
|---|---|---|---:|---|
|无目标性质从头生成|joint_u、joint_all|只给kind与已有来源上下文，随机先验；不输入测试组成或描述，CFG=0|每模型每域1024，共4096|生成成功率、结构/化学有效性、精确群/操作包含、唯一/新颖/VUN、COV-P/R、AMSD/AMCD、Wasserstein/JSD、指纹多样性；独立FE/BG分布|
|指定群生成|两个模型|3D全部230 SG、2D全部80 LG，每群固定群token后产生元素/轨道/几何|每群5，共3100|逐群精确群率、群操作残差、几何/化学有效性、宏平均及失败原因；保持均衡群采样标签，不冒充自然群分布|
|形成能条件|joint_all|TRAIN各域10/50/90分位，CFG=2|6组×256=1536|上述结构质量、独立FE目标误差/命中/有效命中、性质响应与参考分布距离|
|带隙条件|joint_all|TRAIN各域10/50/90分位，重复目标去重，CFG=2|当前MP20为2组、C2DB为3组，共1280|结构质量、独立BG误差/命中/有效命中及响应|
|形成能+带隙条件|joint_all|TRAIN真实共现标签元组，按形成能排序选10/50/90位置，CFG=2|每域3组，共1536|同一结构同时满足FE/BG的联合命中与有效联合命中；各性质误差/分布|
|模块测试|两个模型|不生成材料；完整测试集真实前缀与真实描述|每模型12427条，共24854条测试请求|描述NLL、G/W/Z/STOP准确率；OrbitFlow晶格/坐标/无量纲速度误差|

当前数据下共27个任务、11548次结构生成。服务器按实际checkpoint重新生成plan.csv后确认；分位数重复数依赖训练数据。
表中数量是目标输出预算；生成失败保留序号，不补采样凑足成功数量。
`joint_all`仍是原来的三性质训练权重，本轮单/双性质输入不是重新训练的单性质模型。
Eh缺少独立预测器，本轮先不安排Eh控制任务，也不把FE当成Eh。
模块测试对joint_all使用测试记录中可用的真实训练性质；这是teacher-forced条件诊断，不是无条件生成性能，其条件清单写入模块summary。

### 第二批 reconstruction：完整测试集的CSP和OrbitFlow重建

|任务|模型与输入|数量|评价|
|---|---|---:|---|
|仅组成CSP|两个模型；每个测试结构的精确整数计数，不给真实群/轨道/坐标|每模型(9046+3381)×20=248540，两模型497080|组成满足率、Strict和DiffCSP两套Match@1/5/20、最佳RMSE/RMSD、晶格误差、候选有效性/多样性、自由度分层|
|固定描述诊断|两个模型；额外给真实群、轨道及元素；仍从随机连续先验开始|相同预算，两模型497080|相同重建/质量指标，主要诊断OrbitFlow；与组成CSP分开标记|

两类合计994160次生成；先取得第一批结果，再决定何时运行这一批。没有将全测试集缩成256条。
每例20候选保留顺序，前1/5/20分别计分；两套匹配参数共享候选，不重复生成。
本批不额外扫描性质条件CSP，避免一开始将完整测试集乘以大量目标组合。

## 原报错与修复

附件使用f8e31a8，结果为27 passed、1个旧并行参考指纹测试超过120秒；后续只是--plan，未启动正式生成。
日志不足以断言死锁；冷启动、CPU配额/争用以及线程嵌套均可能延迟完成。
现对spawn参考工作进程限制Torch/BLAS为1线程，将chunksize=4改为1，真实分配多个任务，
记录工作进程与串行/并行完成耗时。测试保留有限300秒看门狗，超时在Linux停止该测试自行创建的进程组，避免遗留子进程。
测试仍要求串并行结果相等、顺序与失败记录相同；没有跳过测试或更改指标公式。
新的统一指标入口本身默认串行构建参考缓存，不依赖旧并行池。

## 服务器：更新与检查

以下均从项目根目录执行。不要把本页所有启动命令同时执行。

```bash
set -euo pipefail
conda activate cgdit
git status --short
git pull --ff-only
git log -1 --oneline
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PWD/src"
export PYTHONIOENCODING=utf-8
export CUDA_VISIBLE_DEVICES=0
python -m pip install --no-deps --no-build-isolation -e '.[test,evaluation]'
python -c "import torch,threadpoolctl,smact,matminer,matplotlib; print(torch.__version__); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pytest tests/test_unified_evaluation.py tests/test_standard_metrics.py::test_reference_metrics_parallel_matches_serial_order_and_failures -q
nvidia-smi
```

如仅缺threadpoolctl，用 `python -m pip install 'threadpoolctl>=3.1'` 补齐该小型CPU线程管理包，随后重跑检查；不安装或升级PyTorch。
`set -e`让测试未通过时立即停止，避免像附件中那样pytest失败后仍继续下一条命令。
生成阶段若GPU已有计算进程，调度会拒绝启动，不停止已有任务。

## 第一批：保存参数、只检查计划

```bash
mkdir -p artifacts/evaluations
cat > artifacts/evaluations/joint_pair_discovery.args.sh <<'SH'
UGG_PAIR_ARGS=(
  --suite discovery
  --output artifacts/evaluations/joint_pair_discovery_seed142
  --checkpoint-root artifacts/runs/main15_seed42
  --count 1024 --conditional-count 256
  --steps 64 --seed 142 --guidance 2 --threads 4 --device cuda
  --cgdit-root ../test_CGDiT
  --predictor-root ../test_CGDiT/output/newton_4g_6p_20261003/predictors
  --predictor-seed 123
  --allow-proxy-transfer
)
SH
source artifacts/evaluations/joint_pair_discovery.args.sh

# 只读取预测器权重/元数据并核对目标、图设置和已保存测试数组
python scripts/legacy_eval/properties.py --project-root "$PWD" \
  --cgdit-root ../test_CGDiT \
  --predictor-root ../test_CGDiT/output/newton_4g_6p_20261003/predictors \
  --predictor-seed 123 --output artifacts/evaluations/joint_pair_predictor_inventory --inventory

python scripts/evaluate_joint_pair.py "${UGG_PAIR_ARGS[@]}" --plan
```

查看 `plan.csv`：必须仅有joint_u/joint_all，没有缺失checkpoint，数量约27任务/11548请求。
示例启用C2DB探索性代理评价：FE/BG预测器来自MP20，C2DB分布会明确标为跨域探索，不能作为已校准的二维性质精度。
若不做该跨域探索，在生成前从参数文件移除allow-proxy-transfer；MP20仍可评价，C2DB性质标为未开展。
两个阶段必须复用同一参数文件。若预测器seed并非123，先按真实目录修改，再生成计划；不要生成后改参数。

## 第一批：阶段一只生成

```bash
source artifacts/evaluations/joint_pair_discovery.args.sh
nohup python -u scripts/evaluate_joint_pair.py "${UGG_PAIR_ARGS[@]}" \
  --stage generate --run --resume \
  > artifacts/evaluations/joint_pair_discovery.generate.log 2>&1 &
echo "生成PID=$!"
```

仅产生samples.jsonl、CIF和生成状态，跳过模块测试、性质预测、指纹和质量指标。
每条失败保留原因；正常退出时pipeline.state=generation_complete。

## 第一批：生成结束后核对，再计算指标

```bash
source artifacts/evaluations/joint_pair_discovery.args.sh
python scripts/evaluate_joint_pair.py "${UGG_PAIR_ARGS[@]}" --verify-generated

nohup python -u scripts/evaluate_joint_pair.py "${UGG_PAIR_ARGS[@]}" \
  --stage metrics --run --resume \
  > artifacts/evaluations/joint_pair_discovery.metrics.log 2>&1 &
echo "评价PID=$!"
```

核验会检查生成记录和CIF哈希。评价阶段先确认全部生成任务完整，再进行模块测试、CPU代理预测和指标计算；不重新生成材料。
中断恢复仍用相同阶段命令。完整结果出现根目录summary.csv、summary.json、model_comparison.png，pipeline.state=complete。
逐样本CSV、汇总JSON、PNG和每任务日志均保留；已完成阶段通过哈希后复用。
根目录summary.csv同时汇总生成数/失败数、有效性、唯一/新颖/VUN、两种COV及AMSD/AMCD、
FE/BG预测均值、目标误差、命中率/有效命中率和同代理参考分布距离，便于比较两个模型。

## 进度

```bash
python scripts/evaluate_joint_pair.py --suite discovery --status
tail -n 20 -F artifacts/evaluations/joint_pair_discovery.generate.log
# 进入指标阶段后改看：
# tail -n 20 -F artifacts/evaluations/joint_pair_discovery.metrics.log
nvidia-smi
```

状态显示当前模型/任务、阶段、完成的生成任务数和评价任务数；当前任务的sampling_seconds为已记录样本计算/导出耗时。
进度状态不会启动或停止任何任务。

全部评价完成后可打包下载：

```bash
python -c "import json; p='artifacts/evaluations/joint_pair_discovery_seed142/pipeline.json'; assert json.load(open(p))['state']=='complete'"
tar -czf artifacts/evaluations/joint_pair_discovery_seed142_audit.tar.gz \
  --exclude=reference_cache --exclude=proxy_reference_cache \
  -C artifacts/evaluations joint_pair_discovery_seed142
sha256sum artifacts/evaluations/joint_pair_discovery_seed142_audit.tar.gz \
  > artifacts/evaluations/joint_pair_discovery_seed142_audit.tar.gz.sha256
```

## 第二批：先保留完整CSP计划

```bash
python scripts/evaluate_joint_pair.py --suite reconstruction --plan
```

将得到8个生成任务、994160次请求。待第一批分析后，再执行：

```bash
# 仅在决定开始完整CSP时执行
nohup python -u scripts/evaluate_joint_pair.py --suite reconstruction --stage generate --run --resume \
  > artifacts/evaluations/joint_pair_reconstruction.generate.log 2>&1 &
```

生成完成后，用同一suite的 `--verify-generated` 核验，再执行 `--stage metrics --run --resume`。
本批默认不额外运行性质代理，先取得完整CSP/固定描述重建指标；首批已进行独立性质评价。
其余13个模型继续使用主入口的--models选择，待两个联合模型的输出、指标和口径核对后再扩展。
