> 后续分层重构：最新目录、Hydra训练入口、续训和预训练初始化见[PACKAGE_REFACTOR.md](PACKAGE_REFACTOR.md)。下文保留先前迁移记录。

# PyTorch Lightning训练管理

## 范围与已有训练

新CLI训练默认使用PyTorch Lightning。描述Transformer、OrbitFlow、群—轨道编译器、
性质编码、CFG和损失公式保持原实现。框架重构不会修改或重新训练已经归档的18套模型。
旧训练器保留在 `unifiedgroupgen/tests/reference_training.py`，使用 `--trainer legacy`；
旧main15、旧参考训练、历史配对和原5轮测速入口明确固定到legacy，保持历史协议。
冻结的历史代码和日志不改写。

新增训练管理入口：

| 模块 | 职责 |
|---|---|
| `run_config.py` | 读取.env、分配真实开始时间目录、记录run.json |
| `CrystalDataModule` | 读取已准备记录、缓存群约束图表、按原子数和边预算分批 |
| `UnifiedGroupGenModule` | 包含原描述网络和OrbitFlow，定义训练/验证/测试与优化器 |
| `TrainingArtifacts` | 保存原生恢复检查点、兼容推理权重、历史、耗时与状态 |
| `Trainer` | 自动反向传播、混合精度、单卡/CPU设备、训练循环与进度条 |
| `WandbLogger`/`CSVLogger` | 在线或离线记录，保存本地CSV；续训保留各会话CSV |

本轮支持单GPU或CPU；没有声明已支持DDP、多节点或跨设备逐位一致。
数据通过独立随机源按seed与epoch分批，避免DataLoader读批耗用模型的随机数。
它与原手写循环的全局随机调用顺序不同，因此不能承诺重新训练的整条轨迹
与旧框架逐位相同。固定输入、权重和随机状态下，底层损失与梯度可直接对照。

## 损失与训练语义

两个网络使用同一个AdamW联合优化：

\[
\mathcal L=\lambda_d\mathcal L_{descriptor}
 +\lambda_f\left(\mathcal L_{lattice}+\mathcal L_{coordinates}/\ell_0^2\right).
\]

新入口默认参考长度为3 Å，坐标损失除以9；显式运行参数会记录在权重与run.json中。
验证每5轮及最终轮执行，固定验证随机数918，并恢复训练随机状态。
降低学习率和提前停止的patience仍按训练轮数计算，不因5轮验证一次而放大5倍。
运行参数仍支持激活检查点、边分块、batch和边预算，原15套配置文件不改字节。
原先的source_domain上下文、性质可见方式与组成/观测训练概率在本轮保留；
这次工程重构没有实现此前讨论的移除来源标签或重新训练组成条件。

`Trainer.test`必须提供明确的held-out `test_records`，不会默认把验证集当测试集。
CSP、从头生成、逐群生成和标准结构指标继续使用已有评估工作流；
训练/测试损失不能替代这些生成质量指标。

## 环境与日志

现有cgdit环境已能导入PyTorch Lightning 2.5.5、wandb 0.22.3、python-dotenv 1.2.1，
本次未安装或升级PyTorch。安装包的训练依赖声明为：

```bash
python -m pip install -e './unifiedgroupgen[training,test,evaluation]'
```

服务器若需要与旧test_CGDiT保持环境隔离，可先clone原cgdit到独立环境，
然后安装 `unifiedgroupgen/environment/requirements-training.txt` 及项目。
先验证依赖与小预算运行，再提交正式训练。

`.env`仍只需要四个主要变量：

```dotenv
PROJECT_ROOT="/absolute/path/to/UnifiedGroupGen"
HYDRA_JOBS="${PROJECT_ROOT}/outputs"
WANDB_DIR="${PROJECT_ROOT}/wandb"
USE_WANDB_LOGGING=1
```

`UGG_PROJECT_ROOT`或导出的`PROJECT_ROOT`可指定工作目录；从项目根目录启动时自动发现。
HYDRA_JOBS在本项目中作为输出根目录使用，不代表引入Hydra的配置覆盖机制。

- `USE_WANDB_LOGGING=1`：W&B在线；由用户自行执行 `wandb login` 配置账号。
- `USE_WANDB_LOGGING=0`：W&B离线；仍保留本地W&B记录。
- `--wandb-mode offline`：对本次运行明确指定离线，不修改.env。
- `--wandb-mode disabled`：只保留项目JSONL/CSV/日志，用于工程诊断。

两种W&B模式的本地记录均在WANDB_DIR下，SDK会创建其标准wandb子目录。
不自动上传模型文件，`log_model=False`；在线指标上传仅在用户启动在线运行时发生。
离线续训会保留各个offline-run会话；不能声称W&B离线会话具有原生在线resume语义。

## 输出目录

```text
$HYDRA_JOBS/<experiment>/<YY-MM-DD-HH-mm-ss>/
  run.json                    真实运行开始时间、W&B ID、配置哈希与实际参数
  last.ckpt                   Lightning完整恢复检查点
  best.ckpt                   最佳验证损失的Lightning检查点
  model.pt                    兼容现有采样/评估的轻量推理权重
  model.best.pt               最佳推理权重
  model.history.jsonl         原有主要字段及训练/验证/保存耗时
  model.status.json           当前批次、状态或完成原因
  model.gradients.jsonl       配置启用梯度诊断时生成
  train.log                   每轮简洁摘要
  logs/metrics.csv            首次会话的本地指标
  logs/resume_<id>/metrics.csv 续训会话指标，保留先前CSV
```

最佳权重只在执行验证并改善时产生。历史导入目录仍标为import_time，
新运行的run.json标为training_run_start，二者不混同。
只有显式传入 `--output` 时才采用指定位置；普通新训练自动使用HYDRA_JOBS日期层级。

## 单个模型与恢复

```bash
# 示例命令；本文档不会执行训练
python -m unifiedgroupgen.common.cli --seed 42 --threads 4 train \
  --trainer lightning --experiment joint_u \
  --config configs/main15/joint_u.yaml --device cuda \
  --batch-size 256 --edge-budget 400000 \
  --coordinate-length-scale 3 --no-checkpoint --edge-chunk-size 32768 \
  --validation-interval 5 --wandb-mode offline

python -m unifiedgroupgen.common.cli --seed 42 --threads 4 train \
  --trainer lightning --experiment joint_u \
  --config configs/main15/joint_u.yaml --device cuda \
  --resume outputs/joint_u/<原运行时间>/last.ckpt --wandb-mode offline
```

原生.ckpt恢复网络、AdamW、AMP、学习率、循环位置、随机状态与自定义patience计数。
配置、训练/验证数据哈希和运行参数不匹配时拒绝恢复。恢复保持原目录与run.json。
也可传入Lightning生成的model.pt，它会定位同目录对应的原生.ckpt。

旧18套模型的.pt仍由旧采样和评估代码直接读取。
继续旧手写训练使用 `--trainer legacy --resume <旧model.pt>`；
若要以旧权重开启新的Lightning实验，使用 `--initialize-from <旧model.best.pt>`，
重新建立优化器。这叫权重初始化，不是完整断点恢复；不会伪造旧Lightning循环状态。

底层Python API默认在fit结束后关闭W&B会话。若需在同一会话紧接着调用
`Trainer.test`，调用 `run_lightning(..., finish_logging=False)`，
完成测试后在调用方的finally中执行 `wandb.finish()`，避免对已关闭的会话继续写日志。

## 新版串行计划

```bash
# 只检查，不启动；顺序沿用已确认的15套顺序
python -m unifiedgroupgen.tests.reference_workflows.train_lightning_experiments --plan

# 本地诊断或远程明确准备完成后，用户再执行--run
python -m unifiedgroupgen.tests.reference_workflows.train_lightning_experiments \
  --models mp20_u c2db51_u joint_u --wandb-mode offline --run
```

加 `--resume` 时只查找匹配配置、seed、运行参数且有run.json的Lightning目录，
不把历史导入的18套模型当成新的Lightning运行。
每个子任务启动时分配自己的真实时间，而非把整批计划时间当成所有实验开始时间。
本次开发只执行诊断与计划，没有提交这些正式训练命令。

## W&B查看内容

分别记录train/val/test的描述、有效flow、晶格、原始坐标及无量纲坐标损失；
记录学习率、裁剪前梯度范数、裁剪比例、显存峰值和训练/验证/保存耗时。
配置启用 `gradient_diagnostics_interval` 后，还记录描述/晶格/坐标各自的加权梯度范数。
性质命中率、稳定性或物理有效性不会由训练损失曲线自动得出。

## 本地验证记录（2026-10-30）

- 固定源码后完整回归 **324 passed**，其间217份源码与配置哈希不变。
- 随后补充中断向串行调度传播的保护，Lightning与评估复用专项 **17 passed**。
  这次全量之后的变动仅涉及中断分支与其测试，单独记录在verification.json。
- 原DescriptorTransformer、OrbitFlow类及joint_loss函数与重构前AST一致，33份配置
  保持原字节，旧18套模型归档清单保持不变。
- 固定批次、权重和随机状态的3D/2D混合损失及全部梯度逐位一致；
  新Lightning的CPU连续训练与原生中断续训权重、调度器及Torch RNG状态一致。
- RTX3060 CUDA BF16训练、原生保存/恢复和明确held-out测试通过。
- 真实数据工程诊断：4条MP20/C2DB训练记录、独立2条验证、2条测试，
  使用18,030,038参数的原联合配置完成1轮；2个固定描述、2步采样结构成功导出，
  群操作闭合检查均通过。这不是收敛模型、生成质量或稳定性结论。
- W&B离线train/val/test记录、CSV和续训日志保留均实测；在线账号与网络未实测。

验证总报告：
`outputs/tests/lightning_migration_validation/26-10-30-13-28-21/verification.json`。
真实数据报告：
`outputs/tests/lightning_real_data/26-10-30-13-41-34/engineering_report.json`。
首次全量测试期间修改源码触发的身份拒绝、首次真实诊断中的W&B生命周期错误
均保留原始记录，修复与最终验证没有删除这些失败证据。
