# 统一CUDA环境与数值精度

唯一环境清单为`unifiedgroupgen/environment.yml`，环境名`unifiedgroupgen_cuda`；包范围与入口在`unifiedgroupgen/pyproject.toml`。面向Linux CUDA，Windows可使用相同环境进行小型验证，不注入Windows专用环境变量。旧环境保留作历史复现。

|组件|固定版本|
|---|---|
|Python|3.10.18|
|PyTorch|2.7.1，官方CUDA11.8构建|
|NumPy / SciPy|1.26.4 / 1.15.2|
|Lightning / PyG|2.5.5 / 2.7.0|
|PyG二进制扩展|匹配Torch2.7、cu118的scatter/sparse/cluster/spline wheels|
|晶体学|PyXtal1.1.3、pymatgen2025.10.7、spglib2.6.0|
|骨架依赖|einops0.8.1、e3nn0.5.1、opt-einsum-fx0.1.4|
|公开性质代理|matgl0.9.2、CPU DGL1.1.2，无GraphBolt依赖|

沿用已经验证的Torch CUDA构建，NumPy选择1.26.4。MatterGen和FAIRChem完整发行包固定不同的Torch/Lightning版本，故只分发本项目调用的官方计算核心子集及MIT许可证，不安装上游训练CLI和应用依赖。核心计算保留，绝对导入迁移到私有命名空间；MatterGen设备工具去除了修改PROJECT_ROOT等训练副作用。来源与原始/本地哈希见`backbones/official/SOURCES.json`及`vendor_manifest.json`。

## 新建与检查

从项目根目录执行：

```bash
conda env create -f unifiedgroupgen/environment.yml
conda activate unifiedgroupgen_cuda
python -m pip install --no-build-isolation -e ./unifiedgroupgen
python -m pip check
python -m unifiedgroupgen.tests.check_environment --require-cuda --output intermediate/checks/environment.json
```

`--no-build-isolation`使用已固定的构建工具；没有用`--no-deps`绕过冲突。先安装清单中的Torch/PyG二进制扩展，避免错误的源码编译。已有完整上游发行包时建议新建环境，不在历史训练环境上强制卸载升级。

Windows PowerShell若系统默认GBK，可仅对当前会话设置`$env:PYTHONUTF8='1'`与`$env:PYTHONIOENCODING='utf-8'`。它们不是环境的强制Windows设置。

应验证pip依赖无冲突、CUDA前后向成功及群目录闭合。Linux全新安装仍需在目标服务器运行同样检查，本机Windows证据不能替代Linux实测。

## 精度策略

统一采用BF16混合精度，不把整个模型、参数或ODE状态转换为BF16。

|计算|精度|
|---|---|
|参数和优化器状态|FP32|
|MLP、注意力等神经矩阵计算|CUDA BF16 autocast，CPU FP32|
|独立状态k/q和ODE累计|FP32|
|周期图物理位移、距离、邻居选择|FP32|
|矩阵指数、Cholesky、群/观测投影、坐标转换求解|FP32；审计按需FP64|
|EqV2球谐归一化；GemNet内部距离/应力几何|局部FP32，其余神经层可BF16|
|物理损失、数值规约|FP32|
|CIF、匹配、几何化学指标及统计|各指标原有FP32/FP64协议|

训练和原生验证由Lightning的`bf16-mixed`管理，不使用梯度缩放器；保留FP32对照配置。生成默认`generation.precision=auto`，支持BF16的CUDA设备选择BF16，CPU为FP32；可显式选择fp32/bf16，不支持的BF16请求会报错。精度写入生成身份及记录，不能跨精度静默恢复。

组件诊断默认`component_check.precision=fp32`，作为不同模型的共同参考，也可显式选择auto/bf16。复现旧结果时显式使用原FP32采样协议。换精度可能改变离散采样、连续轨迹及指标，不承诺位级一致或最终质量不变。

BF16保留接近FP32的指数范围，但尾数精度较低，所以保留上述敏感FP32区域。小batch或图构建占主导时可能更慢，不保证统一倍数；应在目标GPU实测吞吐和显存。新配置用`numerical_policy=amp_geometry_fp32_v1`记录本策略。[PyTorch AMP](https://docs.pytorch.org/docs/2.7/amp.html)、[Lightning精度](https://lightning.ai/docs/pytorch/stable/common/precision_intermediate.html)。

## 复核

```bash
python -m pytest -q unifiedgroupgen/tests/unit/test_cuda_environment.py unifiedgroupgen/tests/unit/test_periodic_backbones.py unifiedgroupgen/tests/unit/test_backbone_generation.py
python unifiedgroupgen/tests/workflows/benchmark_precision.py --output intermediate/checks/precision_benchmark.json
ugg experiment=new_joint_all task=plan
```

测速固定真实MP20/C2DB输入、相同权重和目标，只做前后向、不更新优化器。缩小骨架的数值/成本检查不能替代正式尺寸测速、收敛或生成质量。证据见历史工程记录（已清理）及文章第10.5.7节。

控制台ugg由pyproject.toml的project.scripts生成，等价于`python -m unifiedgroupgen.run`；生成与指标仍使用独立的generate/evaluate入口。
