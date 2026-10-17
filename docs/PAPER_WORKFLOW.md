# 服务器：从现有联合检查点补齐论文证据

所有命令在 /root/private_data/rszhong/workspace/UnifiedGroupGen 中执行。先确认旧 evaluate_joint 进程已经结束再更新源码；运行中的流程会校验源码指纹，跨版本不能复用断点身份。更新不会重新训练现有检查点。

## 1. 更新并验证

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git status --short
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -m pip install -r requirements-evaluation.txt
python -m pytest -q
```

无需更换现有PyTorch。只有使用本地ASE计算器桥接时才安装 requirements-assessment.txt 和你选择的计算器；不会自动下载势函数权重。

## 2. 先完成两项现有检查点工作

完整源重编码可能比代数统计慢；日志提供阶段进度。该任务使用CPU，可以与已完成采样后的离线统计分开执行。

```bash
mkdir -p artifacts/evaluations
nohup python -u scripts/verify_representation.py --workers 4 \
  --output artifacts/evaluations/representation_full \
  --figures assets/representation_full \
  > artifacts/evaluations/representation_full.log 2>&1 &
```

上一轮256次/模式的联合生成若已完成，可直接评估；不必先重采样：

```bash
python scripts/evaluate_generation.py \
  --run artifacts/evaluations/joint_best_seed142 \
  --reference-workers 8 \
  --output artifacts/evaluations/joint_best_seed142_metrics \
  --figures assets/generation_metrics/joint_best_seed142
```

若 pipeline.json 并非 complete，先检查其 phase、对应job的stdout.log/stderr.log和进程；不要把部分结果当完整统计。如果旧版本任务中断且源码已更新，用新输出路径启动新任务，保留旧结果。

## 3. 正式无条件生成：每来源4096次

先查看计划，总尝试数为3×4096=12288（单个采样种子）。不自动启动新训练。

```bash
python scripts/prepare_paper_experiments.py --suite unconditional \
  --count 4096 --seeds 142 \
  --output artifacts/paper_protocols/unconditional4096
python scripts/evaluate_joint.py --campaign artifacts/paper_protocols/unconditional4096/campaign.json --plan
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/evaluate_joint.py \
  --campaign artifacts/paper_protocols/unconditional4096/campaign.json \
  --skip-test --device cuda --output artifacts/evaluations/paper_unconditional4096 \
  > artifacts/evaluations/paper_unconditional4096.log 2>&1 &
```

完成后运行标准评价：

```bash
python scripts/evaluate_generation.py --run artifacts/evaluations/paper_unconditional4096 \
  --reference-workers 8 \
  --output artifacts/evaluations/paper_unconditional4096_metrics \
  --figures assets/generation_metrics/paper_unconditional4096
```

若需要独立采样重复，用新的协议目录和 --seeds 143 144。它们只是同一个检查点的采样重复；独立训练重复需要不同训练种子和新检查点。估计时间应使用已完成任务的 sampling.status.json 中 seconds/attempted × 新预算，分来源计算，再另加指纹、匹配及独立计算成本，不按显卡峰值算力估计。

参考指纹支持 --reference-workers N 的CPU进程并行，默认1。多进程结果保持输入顺序、失败计数和评价定义；使用8进程前确认CPU与内存预算，并把OMP_NUM_THREADS、MKL_NUM_THREADS设为1避免每个进程再启动多线程。缓存以完整文件原子替换，便于重复评价复用。

## 4. 条件和连续对照：先准备，再顺序执行

```bash
python scripts/prepare_paper_experiments.py --suite conditions \
  --count 128 --seeds 142 --output artifacts/paper_protocols/conditions128
python scripts/prepare_paper_experiments.py --suite controls \
  --cases 8 --count 128 --seeds 142 --output artifacts/paper_protocols/controls128
```

每份 campaign.json 都明确列出目标、标签数量、种子和总尝试数。条件扫描只选择该来源TRAIN中实际存在的性质。controls选取有非零坐标自由度的TRAIN实例，固定其描述与元素；这不是未见规则实验。

在GPU空闲时分别执行，每次替换 SUITE 为 conditions128 或 controls128；不要并行挤占同一4090：

```bash
SUITE=conditions128
python -u scripts/evaluate_joint.py --skip-test --device cuda \
  --campaign artifacts/paper_protocols/$SUITE/campaign.json \
  --output artifacts/evaluations/paper_$SUITE
python scripts/evaluate_generation.py --run artifacts/evaluations/paper_$SUITE \
  --output artifacts/evaluations/paper_${SUITE}_metrics \
  --figures assets/generation_metrics/paper_$SUITE
```

controls只绕过连续模型生成先验，仍加载检查点来保留词表、来源和配置；固定描述时不采样Transformer。相同种子对应相同初始连续状态。固定晶格观测使用TRAIN晶格，必须与完全自由晶格任务分开报告。

## 5. 独立性质与弛豫

导出生成候选；可选择同时导出测试集，以计算同一评价器下的性质分布距离：

```bash
python scripts/assess_structures.py --run artifacts/evaluations/paper_conditions128 \
  --reference-config configs/joint_space_layer_large.yaml \
  --output artifacts/assessment_tasks/conditions128
```

tasks.json 中不包含请求性质值，避免把目标误当预测。可把它交给已有CGDiT性质/DFT流程，返回 docs/STANDARD_EVALUATION.md 定义的CSV；也可调用已经安装的ASE计算器工厂：

```bash
# my_calculator:create 必须返回你已经验证适用的 ASE calculator。
# 这里是接口占位，需替换为自己的实际模块及本地权重路径。
python scripts/assess_structures.py --tasks artifacts/assessment_tasks/conditions128/tasks.json \
  --calculator-factory my_calculator:create \
  --calculator-kwargs '{"model_path":"/absolute/local/model"}' \
  --evaluator 'actual-model-version-and-training-scope' \
  --output artifacts/assessments/conditions128
```

- 默认单点计算，无弛豫；--relax 执行FIRE，--relax-cell额外允许3D晶胞变化。2D保持面内胞及非周期法向约定，只弛豫原子，不声称二维应力已优化。
- 工厂可以在 calculator.results 提供 band_gap、formation_energy；没有这些键时保持缺失。普通势函数的总能量不是带隙或形成能。
- --energy-reference JSON 可提供 evaluator、scope、elemental_eV_per_atom 和 phase_entries（composition字典、energy_eV总能量）。所有能量必须来自同一计算器和兼容参考设置；仅生成候选不能构造可靠凸包。不同来源参考不兼容时分开运行、分开报告。
- ML势的“独立”是相对生成器，不代表其训练数据没有重叠；必须报告预训练数据和适用域。2D要求计算器支持 partial PBC；不要悄悄加入法向周期相互作用。
- 参数、输入SHA、错误、失败弛豫均记录；没有合法能量参考时凸包能/SUN保持未评价。该工具不提交DFT、声子或下载模型。
- 长任务中断后，同一命令增加 --resume，按已写入的评估日志前缀继续；数据、计算器、设置和代码必须一致。已记录的失败不会自动重算，重试应另立输出并保留失败分母。

完成独立计算后重跑标准评价到新目录：

```bash
python scripts/evaluate_generation.py --run artifacts/evaluations/paper_conditions128 \
  --assessments artifacts/assessments/conditions128/generated.csv \
  --assessment-metadata artifacts/assessments/conditions128/metadata.json \
  --reference-assessments artifacts/assessments/conditions128/references.csv \
  --output artifacts/evaluations/paper_conditions128_assessed \
  --figures assets/generation_metrics/paper_conditions128_assessed
```

未导出/计算测试参考时省略 --reference-assessments，此时不评价性质Wasserstein距离。要做弛豫后稳定性另用新assessment目录执行 --relax，并保留未弛豫性质任务，避免混淆计算阶段。

## 6. 公平的分来源训练

需要新增训练，不使用旧原始划分的MP20/MPTS52检查点充当公平对照。

```bash
python scripts/prepare_paper_experiments.py --suite matched-training \
  --output artifacts/paper_protocols/matched
```

逐个来源执行下列命令，替换 DATASET；三来源为 mp_20、mpts_52、c2db_51。保留相同组/元素词表、主干宽度/层数和来源类别数量，但训练文件仅含该来源；该来源完全缺失的性质编码器省略（如MPTS52带隙），所以需报告实际参数量，不能宣称各模型参数严格相等。审计仍验证完整父数据和分组谱系，允许的子集在配置中显式声明。

```bash
DATASET=mp_20
python -m unifiedgroupgen.cli preflight \
  --config artifacts/paper_protocols/matched/$DATASET.yaml \
  --output artifacts/paper_protocols/matched/$DATASET.audit.json
python -u -m unifiedgroupgen.cli --seed 42 train --device cuda \
  --config artifacts/paper_protocols/matched/$DATASET.yaml \
  --validation-interval 5 --output artifacts/runs/matched_${DATASET}_seed42/model.pt
python scripts/prepare_paper_experiments.py --suite unconditional --count 4096 --seeds 142 \
  --config artifacts/paper_protocols/matched/$DATASET.yaml \
  --output artifacts/paper_protocols/matched_${DATASET}_generation
python scripts/evaluate_joint.py --device cuda \
  --checkpoint artifacts/runs/matched_${DATASET}_seed42/model.best.pt \
  --campaign artifacts/paper_protocols/matched_${DATASET}_generation/campaign.json \
  --output artifacts/evaluations/matched_${DATASET}
python scripts/evaluate_generation.py --run artifacts/evaluations/matched_${DATASET} \
  --checkpoint artifacts/runs/matched_${DATASET}_seed42/model.best.pt \
  --reference-config artifacts/paper_protocols/matched/$DATASET.yaml \
  --output artifacts/evaluations/matched_${DATASET}_metrics \
  --figures assets/generation_metrics/matched_$DATASET
```

需要同时报告每来源数据曝光/更新数、最佳轮数和分开模型的总参数。共享训练与单来源训练的训练记录数不同；同样1000轮不等于同样计算预算。新颖性默认对实际检查点训练集计算，共享模型看到的3D训练参考更多；若比较新颖率，还需使用共同参考全集作补充分析。

## 7. 外部基线导入与统一PNG图表

外部方法提供完整尝试清单CSV（index,status,cif,可选target_group）；失败也占一行，index覆盖0到count-1，不能只收成功文件。

```bash
python scripts/import_baseline_samples.py --manifest artifacts/baseline_attempts.csv \
  --dataset mp_20 --count 4096 --output artifacts/baseline_import/mp20
python scripts/evaluate_generation.py --samples artifacts/baseline_import/mp20/samples.jsonl \
  --dataset mp_20 --reidentify --output artifacts/evaluations/baseline_mp20_metrics \
  --figures assets/generation_metrics/baseline_mp20
```

未指定目标群的基线，其精确目标群回收为不适用；恢复群仍可统计。导入不证明基线训练划分已对齐；需要自己的原仓库复现实验。

创建 artifacts/paper_results_manifest.json，例如：

```json
{"experiments": [
  {"label": "Shared", "model_seed": 42, "metrics": "artifacts/evaluations/paper_unconditional4096_metrics/summary.json"},
  {"label": "Shared", "model_seed": 42, "metrics": "artifacts/evaluations/paper_controls128_metrics/summary.json"},
  {"label": "Shared", "model_seed": 42, "metrics": "artifacts/evaluations/paper_conditions128_assessed/summary.json"}
]}
```

只列真实已经完成的目录；某项缺失时删去其条目。相同样本的“未计算性质/已计算性质”两个版本不能同时计为两次重复。可用 jobs 字符串列表选择部分任务。

```bash
python scripts/build_paper_results.py --manifest artifacts/paper_results_manifest.json --output assets/paper
```

输出主表、按群/自由度分层表、精确设置下的重复统计、可追踪输入指纹和PNG。缺少独立性质/稳定性值时对应图标记pending。Wilson区间仅描述该批尝试；同描述、同检查点的样本相关性及单模型种子不被它消除。

## 8. 现有文章、图1–6与本轮结果

正文统一维护在 [UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md](UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md)。图1为TiO₂/MoS₂与统一模型流程，图2为表示检查，图3为本轮共享生成；图4–6的缺失实验明确占位。剩余证据见正文附录E。

将已经完成的标准结果导入论文图表时，使用实际服务器结果目录：

```bash
python assets/paper/source/build_manuscript_figures.py \
  --joint-run artifacts/evaluations/joint_best_seed142 \
  --standard-metrics artifacts/evaluations/joint_best_seed142_metrics/summary.json
```

程序核对检查点、生成pipeline、测试划分、六组尝试数和逐样本表指纹；不接受运行中的结果。论文目录保存数值汇总及逐样本CSV，Git保留其原始字节，避免换行转换破坏指纹。生成CIF、完整权重及参考指纹缓存仍位于artifacts。

已提交的本轮标准图可直接从论文数据表重画，无需重新采样或计算参考指纹：

```bash
python scripts/build_paper_results.py \
  --manifest assets/paper/source/standard_results_manifest.json \
  --output assets/paper/standard
```

图像仅保存PNG。更新到服务器后可直接阅读已提交文章与图表；新增正式实验继续使用前述新任务目录，完成后再替换相应占位和正文数值。
