# Linux / RTX 4090完整训练

本仓库包含独立代码以及完整数据包。服务器不需要原`test_CGDiT`仓库，不需要从C2DB网站重新下载数据。

## 1. 下载和安装

```bash
git clone https://github.com/Serein-Ss/UnifiedGroupGen.git
cd UnifiedGroupGen
conda create -n unifiedgroupgen python=3.10 -y
conda activate unifiedgroupgen
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu118
python -m pip install -r requirements-remote.txt
python -m pip install -e ".[test]"
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(), torch.cuda.get_device_properties(0).total_memory)"
python scripts/unpack_training_data.py
python scripts/verify_training_data.py
python -m pytest -q
```

PyTorch的CUDA 11.8安装命令来自[官方版本安装说明](https://pytorch.org/get-started/previous-versions/#v271)。不需要torchvision或torchaudio。
已有`cgdit`环境也可以使用，最低PyTorch版本为2.2；训练器会自动选择对应版本的GradScaler接口。上面的版本锁定文件用于新建环境，复用已有环境请使用下面的流程。
软件版本对应本机实际训练：Python 3.10、PyTorch 2.7.1、NumPy 2.2.6、SciPy 1.15.2。完整Linux安装及4090吞吐需要在服务器执行上述检查，尚未声称远程数值复现。

### 复用已部署的cgdit环境

用户提供的服务器包清单包含Python 3.10.18、PyTorch 2.2.1/CUDA 12.1、NumPy 1.26.4、PyXtal 1.1.3、pymatgen 2025.4.17及spglib 2.6.0。项目目录为`/root/private_data/rszhong/workspace/UnifiedGroupGen`。
该环境中的`torch-scatter`等扩展绑定PyTorch 2.2，所以使用已有依赖并安装本项目：

```bash
source /public/home/achv5zyqua/miniconda3/bin/activate
conda activate cgdit
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps --no-build-isolation -e ".[test]"
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"
nvidia-smi
python scripts/unpack_training_data.py
python scripts/verify_training_data.py
python -m pytest -q
```

这条流程不执行`requirements-remote.txt`，该文件锁定另一套PyTorch 2.7.1环境。上述具体服务器路径来自用户提供的本次终端输出，适用于该实例；其他实例需要使用实际项目路径。
全部检查成功后执行下一节的`--stage all`完整训练命令；如有其他任务正在使用GPU，先确认资源足够再启动。

AMP兼容验证：本机PyTorch 2.7.1/CUDA完整测试91项通过，含强制旧接口下CPU/CUDA/BF16断点续训一致性；独立PyTorch 2.2.1 CPU、NumPy 1.26.4、PyXtal 1.1.3、pymatgen 2025.4.17环境80项通过、11项CUDA测试因CPU构建跳过。该独立环境实际SciPy为1.15.2，用户提供的服务器清单为1.15.3；这是旧接口和主要依赖兼容性验证，Linux/CUDA 12.1/4090测试仍须在服务器执行。

## 2. 启动

完整流程包含两个独立参考训练及一个共享主模型训练，依次执行。每个训练都使用完整训练/验证划分，然后完整测试集损失评估，以及各来源64次无条件、64次性质条件生成尝试。

```bash
mkdir -p artifacts/runs
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage all > artifacts/runs/remote_driver.log 2>&1 &
echo $!
tail -f artifacts/runs/remote_driver.log
```

如果优先训练本工作的二维/三维共享主模型，使用：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage main > artifacts/runs/remote_driver.log 2>&1 &
```

后续用`--stage references`训练两个独立参考模型。主模型从头训练，不依赖参考模型权重。
同一个GPU上只启动一个入口；`--stage all`本身按顺序执行。
本机编排历史使用的`complete_pipeline.py`依赖Windows已有进程句柄，Linux应使用`train_remote.py`。

查看完整计划而不启动训练：

```bash
python scripts/train_remote.py --stage all --plan
```

## 3. 进度与断点

```bash
cat artifacts/runs/remote_pipeline/pipeline.json
cat artifacts/runs/joint_space_layer_large_seed42/model.status.json
tail -n 2 artifacts/runs/joint_space_layer_large_seed42/model.history.jsonl
tail -n 30 artifacts/runs/remote_pipeline/joint_train.stdout.log
nvidia-smi
ps -ww -eo pid,etime,args | grep -E 'train_remote|train_local|unifiedgroupgen.cli|evaluate_checkpoint'
```

正常退出后继续，保持同一stage：

```bash
CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/train_remote.py --stage all --resume >> artifacts/runs/remote_driver.log 2>&1 &
```

检查点保存模型、AdamW、学习率调度、AMP及Python/NumPy/CPU/CUDA随机状态；数据哈希和训练配置必须一致。
如果服务器被强制关机，先用`ps`确认相关进程均已不存在，再删除旧锁文件：

```bash
rm -f artifacts/runs/remote_pipeline/pipeline.lock artifacts/runs/local_large_seed42/runner.lock
```

然后执行上述`--resume`命令。重新执行会重新评估测试集和生成尝试。
更换GPU或CUDA软件栈后，随机状态可恢复，但不保证跨硬件逐位相同。

## 4. 完整数据与划分

| 来源 | 原始完整记录数 | 独立参考train/val/test |
|---|---:|---|
| MP20 | 45,229 | 27,136 / 9,047 / 9,046 |
| MPTS52 | 40,476 | 27,380 / 5,000 / 8,096 |
| C2DB | 16,905 | 10,143 / 3,381 / 3,381 |

共享主模型按材料跨库分组后的train/val/test为82,185 / 10,190 / 10,235，总数102,610，全部记录保留。
该划分隔离共享材料ID、相同CIF及已验证的坐标图哈希候选，不代表穷尽的原型/近重复隔离。
参考模型沿用原始划分，不能直接把参考分数与共享模型新划分分数当成公平对照。

数据包`datasets/full_training.tar.gz.part000`等共104,214,509字节，68个文件。
包括三套转换记录、转换清单、审计、共享划分、原始CSV、恢复后的C2DB CSV，以及用于精确恢复缺失记录的原始ASE归档。
`datasets/manifest.json`保存各部分、整个归档和每个文件的SHA256。
解包逐文件核对原始字节，只重新绑定JSON元数据中的本机路径；结构记录和性质标签不改写。
Git下载会直接获得三个数据包文件，无需Git LFS或额外云盘。

原始CSV解包到`raw_data/`；现有配置的历史`../data`输入位置不用于服务器训练。重新转换时需要显式指定`raw_data`输入并使用新的输出目录。
来源记录中Windows路径作为历史来源证据保留，不是服务器运行所需的文件地址。
C2DB原始归档来源为[固定版本的公开归档](https://huggingface.co/spaces/atomind/mlip-arena/blob/1508879571257075528db9522e845605256f6b74/benchmarks/c2db/c2db.db)，归档SHA256为`caf58205692de480e06149ac43a437385f18e14582e7d9a8dab8b3cb5d4bd678`。
数据保留各来源的使用条件；本仓库没有为上游数据授予额外许可。

## 5. Epoch与本机耗时

2026-10-05 的批量几何、条件编码和缓存优化及服务器更新/续训步骤见 [TRAINING_PERFORMANCE.md](TRAINING_PERFORMANCE.md)。
下面的本机耗时是优化前的历史测量，不代表更新后的速度。

三次训练各自上限1000 epoch，验证损失连续50轮没有改善时早停；学习率连续10轮没有改善时减半。
描述Transformer与几何流模型共同训练，不需要为每一种性质条件重新训练模型。
1000是上限，不是已确认的收敛轮数。实际停止轮数由验证损失决定，不能提前保证100或300轮足够。

本机RTX3060 12GB，batch64、BF16，MP20最近10轮中位耗时750.5秒，即约12.5分钟/轮（含验证）。
MPTS52和共享模型还没有本机完整轮实测；按数据记录/节点/边规模粗略外推，分别约15–35和30–60分钟/轮。这些是预算范围，不是实测结果。

| 每个模型实际训练轮数 | 仅共享主模型，本机粗估 | 两个参考+共享主模型，本机粗估 |
|---|---:|---:|
| 100 | 2.1–4.2天 | 4.0–7.5天 |
| 300 | 6.3–12.5天 | 12.0–22.4天 |
| 1000 | 20.8–41.7天 | 39.9–74.7天 |

上述从头训练预算未包含安装、数据校验、最终生成/测试成本，也未扣除本机已完成轮数。
4090耗时需要最初1–3个完整epoch实测；不能按显卡峰值算力直接换算，因为数据处理、轨道展开和小算子也占时间。
默认batch64沿用已经验证的配置，模型参数和完整数据不缩减。可在测量吞吐和显存后再调整batch，调整后作为新训练配置记录。

## 6. 结果边界

数学/代码测试和数据审计通过不等于材料性能已经验证。完成的`pipeline.json`意味着训练、完整测试集损失和生成尝试均运行完成。
仍需检查生成有效率、对称性/轨道符合性、性质命中、弛豫及物理稳定性；目前不声称这些实验已完成。
