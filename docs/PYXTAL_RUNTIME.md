# PyXtal依赖与层群运行前验证

## 修复范围

项目固定使用PyXtal 1.1.3。PyXtal 1.1.1中的LG52 4c将最后一个位置列为重复的`(0,0,0)`，缺少`(0,1/2,0)`；不能满足指定群操作闭合。依赖文件和运行时检查现在使用相同版本。未修改网络结构、参数维数、训练配置、性质标度器或数据文件。

训练数据门禁、模型构建（包括采样和检查点评价）、CLI数据准备/审计，以及远程训练入口会在实际计算前调用层群目录检查。每个Python进程检查成功后缓存一次，不进入每个batch或ODE步骤；检查不使用全局随机数，也不改变采样随机状态。`--plan`仍只打印计划。

检查覆盖全部80个层群、477种单Wyckoff轨道：

- 操作闭合及原子置换的一一匹配；重复或缺失的轨道成员会阻止运行。
- 轨道参数基底秩、稳定子约束核维数及切向群作用一致性。
- 群相容晶格、晶格基底正交性；同时记录目录数值指纹和残差。

检查在每种轨道的确定性一般参数点确定原子置换，再验证该置换下的仿射常数项和全部自由参数系数，因此闭合检查不限于该单个点。不表示穷举全部可重复轨道组合，也不保证不同轨道不碰撞、精确回收目标群或材料稳定性。不能跳过检查或自动改群来继续运行。

## 在服务器更新

在已有cgdit环境与项目目录执行：

```bash
conda activate cgdit
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
python -m pip install --no-deps "pyxtal==1.1.3"
python -m pip install --no-deps --no-build-isolation -e '.[test]'
python scripts/verify_symmetry_runtime.py
python scripts/verify_training_data.py
python -m pytest -q
```

单独更新PyXtal不会由pip升级PyTorch、CUDA、NumPy或绑定扩展。更换依赖后重新启动Python任务；已有进程不会自动刷新缓存。无需重新解压数据包或从头训练。

已有联合检查点的兼容验证：

```bash
python scripts/verify_symmetry_runtime.py \
  --checkpoint artifacts/runs/joint_space_layer_large_seed42/model.best.pt \
  --output artifacts/audits/joint_pyxtal113_compatibility.json
```

此命令在CPU上严格加载两个网络，逐条检查检查点声明的train/val/test文件的状态维数与有限性、训练/验证哈希，并对每个来源的一条记录执行实际前向损失计算。报告保留文件哈希和LG52 4c出现数。`--records path1 path2 ...`可显式指定待检查文件；若不指定检查点或记录，只检查目录。

通过后可继续原有评估流程，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_joint.py \
  --device cuda --count 256 --steps 64 \
  --output artifacts/evaluations/joint_best_pyxtal113_seed142
```

使用新的评估输出目录保留旧结果；目录指纹和运行日志应随新评估一起归档。

## 本地兼容性证据

2026-10-19在PyTorch 2.2.1 CPU、PyXtal 1.1.3环境验证：

- 80个层群、477种单轨道通过运行前检查；错误版本和伪装成正确版本的错误操作表均有阻止运行的回归测试。
- 对比1.1.1与1.1.3的全部2,280种SG/LG/PG单轨道仿射映射、自由轴及晶格基底，按绝对容差1e-12比较，仅LG52 4c的位置偏移表不同。
- 已有联合数据102,610条状态维数与有限性检查通过，数据文件哈希未改变；没有LG52 4c记录。该修复没有改变这些记录所使用的单轨道映射。
- 已归档第195轮联合最佳检查点严格加载成功：描述模型3,718,346参数、连续模型14,312,460参数；三个来源的实际前向损失均有限。

本地详细报告在`artifacts/audits/pyxtal113_runtime_compatibility.json`和`artifacts/audits/pyxtal_chart_compatibility.json`。该验证不是全测试集损失复算、原始CIF全量重编码或物理认证。旧检查点没有群表指纹，不能据此保证跨软件环境的历史采样逐位复现；其他历史检查点应分别运行上述兼容检查。

目录审计的NumPy展开另对零坐标自由度轨道直接使用常量偏移，避免零长度收缩的偶发非有限结果；不改变训练主干的Torch展开或参数维数。
