# 2026-10-24 吞吐测速分析与三个5轮试跑

## 输入核验

- 本地输入：`tmp/batch_throughput_20261024_125502.zip`（项目父目录下）。
- SHA256：`4ae9953914c9d2041da2bafd9ab8deb4ee0c990070188a76b92117d4da987471`，此值是本地计算记录，不是独立发布端校验。
- ZIP CRC通过，200个文件，33组batch/edge组合均完成随机池与高边数测试，没有OOM。
- 三种设置分别有12、9、12组候选；每种设置的配置、TRAIN/VAL、固定4096条计时样本文件指纹、种子和精度一致。逐条核对各候选record IDs（含重复计数）相同，计时恰好覆盖4096条。

## 选定的5轮候选

| 设置 | batch size上限 | edge budget | 记录吞吐量（样本/秒） | 随机池平均实际batch | 高边数吞吐量 | 峰值分配/保留GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MP-20 | 256 | 400000 | 114.52 | 240.94 | 523.47 | 16.22 / 16.93 |
| C2DB | 768 | 200000 | 705.92 | 682.67 | 664.78 | 9.74 / 10.78 |
| 联合 | 256 | 400000 | 117.74 | 240.94 | 477.91 | 16.61 / 17.22 |

这些是当前已测候选中的5轮优先验证设置，不是全局最优或正式长期训练配置。

- MP-20、联合的256/400000是本次随机池记录吞吐量最高者，也通过了高边数压力测试。边预算仍可能将实际batch限制在256以下。
- C2DB的768/400000记录711.85样本/秒，仅比768/200000快0.84%；保留显存却从10.78升至13.96 GiB，高边数测试反而略慢（654.54 vs 664.78样本/秒）。因此先用768/200000。
- C2DB 1024/400000的吞吐量696.19低于768，峰值保留21.39 GiB，没有因为占用更多显存而更快。
- 所有候选都只测了一次；MP-20、联合的最大测试batch是256，不能推断更大的batch一定更慢或更好。

## 必须说明的测速局限与修正

旧脚本固定预热6步，MP-20 B64有64个池批次，只预热其中6批；C2DB B768只有6个池批次，预热已覆盖全部记录。原始step_seconds中可见小批量最后6批明显快于前面的首次访问批次。

因此，表中随机池平均吞吐量混入了不同程度的首次访问开销，不能把C2DB约5.2倍等差异当作稳态加速比；不能与过去已热缓存的400–600样本/秒基准直接比较。新的`--full-pass`已修正为至少预热一整遍池数据。无需为了本次试跑重跑全部候选，先用完整5轮第2–4轮的训练中位数和实际样本/秒确认上述候选。

## 试跑设置

- 仅三个代表模型：`mp20_all`、`c2db51_all`、`joint_all`，顺序串行。
- 每个从seed42新初始化训练5轮，第5轮完整验证；不恢复旧模型。
- loss = descriptor + lattice + raw_coordinates / 9；显式传`--coordinate-length-scale 3`。
- 固定4线程、原学习率0.0003和bf16、关闭activation checkpointing、edge chunk32768。
- 原15套配置及数据审计不修改，运行参数保存在`runtime_options`。独立测试也读取运行时的尺度/批量/边预算。
- 额外逐分量梯度诊断保持现有配置默认值；每批次梯度范数/裁剪率仍进入history。不能将少了额外诊断的计时与旧诊断实验当作纯batch加速比。

## 服务器执行

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git pull --ff-only
mkdir -p artifacts/runtime_pilots
export UGG_PILOT_ROOT="artifacts/runtime_pilots/selected_3A_$(date +%Y%m%d_%H%M%S)"
nohup bash scripts/run_selected_pilots.sh "$UGG_PILOT_ROOT" \
  > "${UGG_PILOT_ROOT}.driver.log" 2>&1 &
echo "PID: $!"
echo "日志: ${UGG_PILOT_ROOT}.driver.log"
echo "最终结果包: ${UGG_PILOT_ROOT}.zip"
```

脚本先检查前一轮损失评估完成、GPU0空闲和输出目录全新，不会停止进程。失败时停止后续试跑，不自动改batch重试，也不启动main15。

每个模型保留`model.history.jsonl`、`pilot.json`、`timing.json/csv`及运行日志；总目录生成`summary.json/csv`和排除`.pt`权重的ZIP及SHA256。每轮统计总/描述/晶格/原始坐标/除9坐标损失，训练/验证/保存/其他耗时，实际平均batch、样本/秒、梯度范数/裁剪率和峰值分配/保留显存。第1轮冷启动单独保留，以第2–4轮判断稳态；这5轮不作收敛或最终生成质量结论。

## 本地验证

Torch 2.2.1+cpu / PyXtal 1.1.3隔离环境，17项相关测试通过；包含运行时batch/边预算/3 Å保存并恢复后权重逐项一致，独立评估按3 Å计算flow=lattice+coordinates/9，分项损失与计时汇总、误用损失尺度拒绝检查、完整采样池预热及原评估恢复保护。`bash -n scripts/run_selected_pilots.sh`通过。没有在本地执行三个完整代表模型的5轮任务，没有启动15套正式训练。
