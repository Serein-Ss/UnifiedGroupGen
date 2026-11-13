# 三个新基础模型的本地独立评估

## 范围与当前状态

本次只评估修复数据 `source_integrity_v1` 上的 `mp20_u`、`c2db51_u`、`joint_u` 最优总验证损失检查点。不是旧 M1/M2/M3，也不启动或修改 15 套训练。用户已改为在本地 RTX 3060 执行。

运行目录：`artifacts/evaluations/baselines_256_local_seed142_20261026`。`pipeline.json` 是状态依据；本文件不把计划视为已完成结果。检查点放在独立的 `artifacts/evaluation_inputs/main15_seed42`，与下载原件的 SHA256 相同。

环境：现有 `artifacts/acceleration_cuda/Scripts/python.exe`，继承 cgdit 的 CUDA PyTorch，使用已验证的 PyXtal 1.1.3；没有安装或替换依赖。

## 预算

| 模型 | 测试集请求记录数 | 从头生成 | 指定组成 CSP | 固定真实描述 | 指定群 |
|---|---:|---:|---:|---:|---:|
| mp20_u | 9,046 | 256 | 256 | 256 | 230 × 5 = 1,150 |
| c2db51_u | 3,381 | 256 | 256 | 256 | 80 × 5 = 400 |
| joint_u | 12,427 | 128 SG + 128 LG | 128 SG + 128 LG | 128 SG + 128 LG | 230 × 5 + 80 × 5 = 1,550 |
| 合计 | 24,854 | 768 | 768 | 768 | 3,100 |

总生成预算 5,404 次，包括失败尝试，不反复抽样补齐成功结构。

## 模块测试

1. 描述 Transformer：全测试集真实前缀下的 G/W/Z/STOP NLL、Top-1、真实前缀整段准确率。不能把真实前缀准确率解释为自由生成整段序列的准确率。
2. OrbitFlow：真实描述条件下的随机路径速度误差；固定测试随机种子 918，每条记录一次随机路径与时刻。晶格、原始坐标、坐标/9、总流损失分别保存。
3. 未见元素等不支持记录单独列出，保留请求分母，禁止仅选择容易或支持的案例后冒充完整测试。

## 三类结构生成

- 从头生成：不提供组成、群、轨道、目标性质或真实结构。基础模型 guidance=0；联合模型仍输入 source_domain=0/1 与 space/layer 类型，因此不是来源完全无条件的模型。
- 指定组成 CSP：从完整测试集按固定种子排序，取各域前 256/128 条；给出标准化晶胞的精确元素计数，不给真实群或轨道。通过解码硬计数约束实现，不启用未训练的组成编码。该任务同时考察两个网络。
- 固定真实描述：只给群、完整轨道列表与轨道元素，不给真实 k/q、晶格或坐标；连续状态从随机先验生成，隔离考察 OrbitFlow。

每个案例一次生成，统计 Match@1、匹配后物理 RMSD（Å）、组成满足率、有效率。RMSD只在成功匹配且可对齐的结果上平均，必须同时报告其数量。分开统计零与非零坐标自由度。

同组成多晶型可以不匹配测试集中的单一参考结构；不能把不匹配自动视为材料无效。标准化常规胞计数也不等同于原始 MP20 原胞预算；跨论文比较前必须统一晶胞约定与匹配协议。

## 指定群与通用生成质量

- 全 230 SG / 80 LG，每群5次；区分训练见过与未见群。
- 保存操作保持、晶格相容残差、重数退化、轨道重合、精确群识别。
- 导出 CIF 独立重识别，主容差 0.001 Å；逐群另在 0.0001/0.001/0.01 Å 下复查。识别失败保留为 unknown，不作为精确群成功。
- 复用现有标准指标：几何、周期镜像距离、原子半径距离、SMACT 化学筛查、指纹成功、联合有效、唯一性、新颖性、VUN、COV-P/R 与分布差异。
- 新颖性参考对应检查点实际同类型 TRAIN；覆盖率和分布参考相应来源 TEST。256与128样本的COV-R不能无条件横比，表中保留样本量。
- 2D 使用部分周期边界、实际面外距离、面积/原子和层厚；去除任意真空对匹配与指纹的影响。不是把二维适配分数当作原始 MP20 基准分数。
- 不评估或声称性质达标、弛豫后稳定、凸包能或 DFT 稳定。

## 运行、恢复与结果

父调度顺序为三个模型各自完整模块测试和四类生成，然后四个模型/来源的 CPU 指标任务。采样模型常驻进程；不为每个案例重新加载。每次尝试使用独立可重建随机种子。

`--resume` 必须具有相同代码哈希、检查点哈希、种子和运行设置；保留已记录尝试及失败。恢复时核对已有 CIF 哈希。截断的最终 JSON 行单独保留备查，只重做尚未完整记录的该次尝试。

查看：

```powershell
Set-Location 'E:\WORKSPACE\CodePlace\test_CGDiT\UnifiedGroupGen'
& '.\artifacts\acceleration_cuda\Scripts\python.exe' scripts/evaluate_baselines.py --status --output artifacts/evaluations/baselines_256_local_seed142_20261026
Get-Content artifacts/evaluations/baselines_256_local_seed142_20261026.driver.log -Tail 15
nvidia-smi
```

中断后使用原始启动参数恢复（不要在主控仍存活时重复启动）：

```powershell
& '.\artifacts\acceleration_cuda\Scripts\python.exe' -u scripts/evaluate_baselines.py --run --resume --device cuda --threads 4 --count 256 --per-group 5 --steps 64 --seed 142 --test-batch-size 8 --checkpoint-root artifacts/evaluation_inputs/main15_seed42 --output artifacts/evaluations/baselines_256_local_seed142_20261026 --update-manuscript
```

完整结束后自动写入 `assets/results/baselines_256_local_seed142_20261026`：

1. `01_component_tests.png`：各模块测试误差；分来源统计。
2. `02_reconstruction.png`：组成 CSP 与固定描述 Match@1、已匹配 RMSD。
3. `03_de_novo_quality.png`：从头生成质量、VUN、覆盖。
4. `04_all_groups.png`：逐群操作保持、精确群与联合有效；标记未见群。
5. `05_generated_distributions.png`：真实与生成的规模、体积/面积、层厚分布。
6. `06_generated_structures.png`：每域前4次成功结构总图，不人工挑选，不逐个堆放图片。

同时保存 CSV、中文结果稿、图表与输入哈希清单，并替换正文现有的 `BASELINE256_RESULTS` 占位段；先备份正文。只输出 PNG。

网页应在上述 manifest 完整后更新，沿用既有私有 Sites 站点。不得把局部结果或集成验证的22次样本写成5,404次正式结果。

完整打包：

```powershell
& '.\artifacts\acceleration_cuda\Scripts\python.exe' scripts/evaluate_baselines.py --pack --output artifacts/evaluations/baselines_256_local_seed142_20261026
```

ZIP及SHA256在评估目录同级；包含逐样本记录、CIF、指标、PNG与表，不包含模型权重或完整训练数据。

## 已完成的启动前验证

- 9项针对性单测通过：预算/域分配、断点/CIF保留、种子、GPU保护、数据路径与哈希、描述损失一致、3D/2D匹配、独立群容差识别。
- 三个真实检查点的小规模集成：16个模型/域/任务组合，22次生成成功。该规模只证明流程可运行，不代表正式生成质量。
- 全部三个模型的训练/验证分割SHA与审计一致，测试输入请求数为9,046/3,381/12,427。
- 未修改模型主干、权重、训练数据或15套训练配置。

## 运行期恢复记录：参考缓存父目录缺失

2026-10-26，三个模块测试与5,404次生成全部结束后，首个指标任务在写入参考缓存时因 `reference_cache/mp20_u` 目录不存在而退出。没有丢失采样记录或CIF。

恢复时核验全部5,404次尝试及成功CIF哈希、所有锁定的代码哈希和三个检查点哈希；补齐三个模型的缓存父目录。MP-20的9,046条、C2DB的3,381条参考缓存与既有本地缓存的输入路径及文件哈希、依赖版本、指标代码哈希完全一致，按字节复制复用。联合模型不满足同一缓存身份，继续按原代码计算，不手改缓存身份。

以原参数 `--resume` 仅恢复尚未完成的指标计算，没有重采样、重训或修改评估身份。原始失败日志与 `recovery_missing_cache_directory_20261026.json` 均保留。恢复日志使用 `.recovery_20261026.driver.log` 与 `.recovery_20261026.driver.err.log`，与原日志分开。

新建相同流程目录时，应在启动前显式创建 `reference_cache/mp20_u`、`reference_cache/c2db51_u`、`reference_cache/joint_u` 三个子目录；本轮运行期间不改动已锁定的评估脚本。
