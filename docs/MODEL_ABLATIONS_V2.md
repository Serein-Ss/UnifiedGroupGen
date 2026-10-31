# 新模型输入、主干、先验与补全实验

本文件记录2026-10-31实现的变更。原15套已训练模型仍采用其保存的配置、权重和来源上下文。下面新增的无来源模型、骨架与补全方案尚无正式训练或生成质量结果；本次只做代码验证，没有启动正式任务。

## 1. 与SGEquiDiff的关系

用户所指的SEquiDiff确认为[SGEquiDiff](https://arxiv.org/html/2505.10994v2)。两者都利用群和Wyckoff轨道约束，不能把“Transformer选轨道，再生成几何”本身当作本工作的独有创新。

|环节|SGEquiDiff|本项目|
|---|---|---|
|分解顺序|空间群、晶格、Wyckoff/元素、坐标|描述a包含群与元素轨道，随后联合生成z=(k,q)|
|晶格|离散telescoping采样|群相容对数度量空间中的连续流|
|坐标|ASU内Wyckoff形状与空间群扩散核|仿射轨道图F=b+Bq及流匹配|
|速度约束|空间群等变场与Wyckoff切空间关系|完整原子候选速度经度量最小二乘约化为q速度|
|目前实例|三维空间群|共用编译/网络接口的SG和LG；PG有编译支持，尚无本次数据训练证据|

可讨论的方法贡献是：明确群作用输入契约、晶格与位置空间的统一构造，以及这些图上的生成模型接口。还必须用SG/LG共享训练、规则准确性、组件质量和消融支持。当前仍有群编号及Wyckoff字母嵌入，不等于已经学会任意未知规则，也未实现磁群/自旋群的完整生成。

## 2. H₀是什么，为什么保留

设R为群操作在所选分数坐标系中的线性部分，实际度量H=LᵀL必须满足RᵀHR=H。平移部分不改变长度和夹角，因此不进入这个方程。

编译器对有限线性操作集合构造：

\[
\widetilde H_0=|P|^{-1}\sum_{R\in P}R^TR,\qquad
H_0=9\,\widetilde H_0/(\det\widetilde H_0)^{1/d}.
\]

群的右乘只会重排求和项，故TᵀH₀T=H₀；每项正定，平均仍正定。9 Å²是参考尺度约定，对应3 Å，并非要求每个真实晶格矢量长3 Å。

令A=H₀^(1/2)，U=ARA⁻¹，则UᵀU=I。在对称矩阵中求UᵀSU=S的解空间，取Frobenius正交基Sα。网络状态中的晶格部分是系数k：

\[
S(k)=\sum_\alpha k_\alpha S_\alpha,
\quad H(k)=A\exp(S(k))A,
\quad L=\operatorname{chol}(H)^T.
\]

矩阵指数保证正定，不变空间保证群相容。H₀是从操作表预先算出的常量，不是训练标签或材料来源特征。

例如六角分数坐标的R=[[0,-1],[1,-1]]并不满足RᵀR=I。相容的二维度量与[[1,-1/2],[-1/2,1]]成比例，对应120°夹角。直接删除H₀而沿用普通对称矩阵基，会连k=0时的晶格都不相容。可以换用其他相容SPD参考，或者直接在原坐标求解相容度量；必须同步改变编码、先验与输出变换。已有权重不能直接换参考后继续解释为同一模型。

## 3. 实际网络输入、候选输出与合法化

描述Transformer接收已生成的描述前缀、可见性质；启用组成训练时另接收组成计数。训练目标是下一描述token。采样时先选择合法kind/群，再选轨道、元素及终止符；化学约束负责过滤不合法token。

OrbitFlow接收已固定的a、当前噪声状态z=(k,q)、时间t、可见性质、可选组成及结构观测。原子图由编译器展开，不输入真实终态的未知坐标。

|特征|实际形式|作用|
|---|---|---|
|全局时间|8个频率的sin/cos，共16维，经MLP|指示流的阶段|
|全局晶格|当前H=LᵀL的9个数，经MLP|当前长度和夹角|
|群|kind偏移后的群编号嵌入|告知离散规则实例；尚非规则内容编码器|
|性质|各性质独立MLP/类别嵌入，相加；缺失用可学习null|性质条件及CFG|
|组成|118维计数的log1p，经MLP；仅在训练过该任务时使用|硬组成条件的可学习上下文|
|节点|元素、Wyckoff字母、全局上下文之和|区分化学及轨道|
|观测|6个补齐晶格系数+6个掩码；每节点3个已知位置贡献+3个掩码贡献|结构补全，不由性质CFG丢弃|
|原主干边|周期相对坐标傅里叶、非周期相对坐标、H|相对几何；原主干为自定义消息传递|
|新周期主干边|周期像、相对位移及其官方径向/角向特征|按各官方核心建图；二维法向不产生周期像|

候选坐标输出是N×3，晶格候选是每图3×3。所有主干共用后续约化：令B为坐标展开的Jacobian，W=I_N⊗H，

\[
v_q=(B^TWB)^{-1}B^TWu,\quad v_F=Bv_q,
\quad (v_k)_\alpha=\langle S_\alpha,\operatorname{sym}(M)\rangle_F.
\]

实现按轨道分块求解，不组装大逆矩阵。零坐标自由度的轨道无q输出。二维/三维的候选头形状一致，合法k/q维数由编译器决定。这个约束确保指定群操作成立；不会保证结构的完整对称群恰好等于目标群、化学合理或稳定。

## 4. 删除新模型的数据来源条件

- 新默认是`experiment=new_joint_u`，实际实验名`joint_u_v2`。新15套目录为`configs/experiments_v2.yaml`；选择器均为`new_<旧名>`。
- 新data配置不添加source_domain，没有dataset_domains，没有逐来源性质归一化；性质统计只从合并训练集计算。
- `evaluation_domains`只用于按SG/LG分层汇总、选择参考数据，不传入网络，不用于CFG。
- `input_contract=source_free_v2`拒绝注入来源编码或逐来源scaler。
- 原`configs/experiments.yaml`及旧15个选择器保留，供原模型复现和推理；它们不是新的默认训练方案。

原数据中的dataset字符串仍保留为数据溯源，不是网络特征。新联合模型仍需说明要生成SG晶体还是LG材料：空间/周期维数是物理规则输入，与MP20/C2DB数据库来源不同。当前不实现自动在不同维数之间抽样的任务策略。

来源条件并非原则上错误的机器学习输入，但与本项目期望的无来源统一模型不同，因此从新方案删除。删除标签也不会自动统一不同数据库的性质计算基准；仍需检查形成能参考、计算方法和缺失标签，不能把全局归一化当成消除物理标签差异。

移除来源、改变性质可见策略会改变训练分布，应重新训练新15套；不会把旧模型改名后声称已完成这项实验。旧联合模型评估继续携带保存的来源上下文。

## 5. 三种可选官方骨架

原模型是自定义MessageLayer，不是CSPNet。新配置在保留OrbitFlow共同条件编码、编译与投影的前提下切换候选主干：

|配置|真正使用的核心|适配范围|
|---|---|---|
|model=default|原MessageLayer|旧15套兼容路径|
|model=cspnet|[DiffCSP官方CSPNet](https://github.com/jiaor17/DiffCSP/blob/main/diffcsp/pl_modules/cspnet.py)|原子条件注入、二维非周期方向、公共读出接口|
|model=mattergen|[官方MatterGen GemNetT](https://github.com/microsoft/mattergen/blob/main/mattergen/common/gemnet/gemnet.py)|条件输入、周期图、Cartesian候选力到分数速度、官方应力候选到k|
|model=equiformer_v2|FAIRChem1.10.0的官方EquiformerV2|周期图、l=0条件注入及新增晶格头，官方向量头转换为分数速度|

后两项使用固定版本的官方计算核心子集，无自制替代网络；核心源代码随包分发，绝对导入路径迁移至私有命名空间，去除上游训练CLI的安装依赖。EqV2采用官方维护实现，并非声称与最初AtomicArchitects仓库逐字相同。新增条件/晶格适配后，整体生成器也不等于原论文整套模型。GemNet不代表复制了MatterGen扩散过程。官方来源、commit及许可证在`pl_modules/backbones/official`；运行时依赖版本、核心源码及资源哈希写入新骨架检查点配置，加载不一致会拒绝。

此处默认核心层数/通道参考官方结构，局部测试使用缩小宽度和层数以控制显存。需重新测速合理batch；原15套的batch256/768不能直接当作这些更大骨架的最优值。新骨架的半径图实际边数不同于原完全图的N(N-1)，原边预算仍是批处理代理，不是新骨架显存保证。

EqV2官方归一化在BF16下需要显式提升输入；适配器只将球谐归一化保持FP32，其余神经层可使用BF16。GemNet坐标转换、周期距离和应力几何也明确保持FP32。统一策略为BF16混合精度神经计算与FP32合法状态/几何/投影，不能理解为全流程使用16位。实际效率仍需按主干和batch实测。

### 统一CUDA环境

使用`unifiedgroupgen/environment.yml`构建`unifiedgroupgen_cuda`：Python3.10.18、Torch2.7.1 CUDA11.8、NumPy1.26.4、Lightning2.5.5及匹配的PyG扩展。所有候选主干的实际依赖已纳入清单；不再安装完整`mattergen`或`fairchem-core`发行包，也不以`--no-deps`绕过其冲突。原来的可选骨架安装脚本已撤下。

```bash
conda env create -f unifiedgroupgen/environment.yml
conda activate unifiedgroupgen_cuda
python -m pip install --no-build-isolation -e ./unifiedgroupgen
python -m pip check
python -m unifiedgroupgen.tests.check_environment --require-cuda --output intermediate/checks/environment.json
```

核心源码、许可证和提取清单在`backbones/official`，保留原数学计算，迁移导入路径；MatterGen设备工具提取时去除了训练工程的环境变量副作用。`vendor_manifest.json`同时记录原始及本地哈希。新建Linux环境仍需在实际服务器运行同一检查，Windows的CUDA实测不等于Linux已验证。完整上游训练CLI不属于本项目环境的支持范围。

## 6. 先验与训练路径消融

必须区分“t=0从什么分布取样”与“训练途中怎样加噪/插值”。[DiffCSP](https://github.com/jiaor17/DiffCSP)的周期坐标均匀终态与wrapped-normal扰动核不是同一个概念。[SGEquiDiff](https://arxiv.org/html/2505.10994v2)还涉及受限Wyckoff形状上的先验及空间群噪声核；普通q上的wrapped normal不等于复现它。

[FlowMM](https://arxiv.org/abs/2406.04713)使用材料空间的流形与物理启发基础分布；[CrystalFlow](https://www.nature.com/articles/s41467-025-64364-4)涉及晶格表示和周期坐标上的流匹配。它们支持做先验消融，但不能直接把晶格长度分布原样当作本项目k系数分布。

|参考工作|取样/加噪设计|与本次实现的关系|
|---|---|---|
|DiffCSP|晶格高斯终态；周期坐标均匀终态，训练使用wrapped-normal噪声|保留图内正态/均匀对照；不复制其完整score训练|
|MatterGen|晶格均值、方差按原子数与参考密度缩放；坐标wrapped VE噪声也按原子数缩放|见[官方corruption代码](https://github.com/microsoft/mattergen/blob/main/mattergen/common/diffusion/corruption.py)；尚未复制这套SDE，GemNet骨架接入不代表噪声过程接入|
|SGEquiDiff|受限Wyckoff形状及空间群噪声核|wrapped_normal只作独立参数先验消融，不是该方法复现|
|FlowMM|环面坐标基础分布；晶格长度对数正态和有界角度|physical_metric把物理晶格取样推入本项目群相容图，带明确投影|
|CrystalFlow|周期坐标基础分布与wrapped Gaussian条件路径|本次保留周期最短位移直线路径，另有可调高斯桥；未完整复制其路径协议|

本项目的`prior`配置均在合法图中采样，训练和生成读取同一检查点配置：

|选择|实现|含义|
|---|---|---|
|legacy|原compiled.prior：k正态，周期q均匀，非周期q正态|旧路径及新模型对照默认；保持原随机调用|
|scaled_normal|调整k及非周期高度尺度|简单尺度消融|
|wrapped_normal|周期q取取模高斯|图内先验消融；不是SGEquiDiff完整方法|
|physical_metric|长度对数正态、角度有界取样，正定检查后白化取log并投影到不变基|FlowMM启发的推前先验；不是论文完整复现|
|train_gaussian|仅训练集、逐kind/群拟合k均值与方差|经验基分布；未见群显式退回标准正态|

默认训练t均匀取自[0,1]。无观测时周期方向使用最短周期位移Δ，z_t=z₀+tΔ，目标速度u_t=Δ。有观测Az=c时使用同一仿射纤维上的连续lift，不逐分量独立wrap，以免破坏线性观测。

可选`flow_path.sigma`加入高斯桥：

\[
z_t=z_0+t\Delta+\sigma\sin(\pi t)\epsilon,
\quad u_t=\Delta+\sigma\pi\cos(\pi t)\epsilon.
\]

有观测时ε先投影至ker A。这是本项目可控CFM消融路径，不声称已经完整实现某篇扩散论文的SDE、score loss及反向采样器。更换prior/path须重新训练；不支持对旧权重随意换先验并冒充公平比较。

保持当前3 Å坐标尺度和坐标损失除以9。不同先验、骨架和掩码消融应分别变一个因素，先检查损失/梯度再讨论最终质量。

## 7. 用户生成条件与积分/批量

用户条件包括：SG/LG/PG类型、指定群、固定完整描述、性质数值、组成计数或化学式、允许元素集合/化学体系、必含元素、已观测结构/CIF及原子映射、独立参数的线性观测。模型只在其支持的类型/元素/任务训练范围内具有学习证据。

其中性质使用CFG；组成和结构观测是硬约束，在两条CFG分支都保留。步数、种子、温度、solver和batch不是材料性质。旧联合模型的source_domain只属历史兼容条件。

Heun每步先算v₁，再用预测点算v₂，以平均斜率更新：

\[
\tilde z=z+h v(z,t),\quad
z_{next}=z+\tfrac h2[v(z,t)+v(\tilde z,t+h)].
\]

这是二阶数值积分，不是模型必须“生成两次”。新可选Euler每步只算一次，通常要更多步才能达到同等误差。CFG强度非0/1时每次速度还需有条件/无条件各一次；64步Heun通常对应128或256次网络调用。

`generation.batch_size`现在将多个不同图的连续积分打包到一次`forward_batch`，节点/边不会跨结构连接。描述的自回归选择及CIF校验仍逐条处理，不能把它称作所有步骤完全并行。每个候选用seed+index建立自己的随机数发生器；打包与单条在浮点容差内验证，不承诺GPU归约位级一致。

批量输出一个带内嵌CIF、哈希和失败记录的samples.jsonl。batch_wall_seconds是真实整批墙钟；摊到每条的wall_seconds不是单条延迟。无法数值积分的记录保留为失败；不悄悄重采样补足成功数。GPU OOM明确报错。已有prior_samples精确复用分支保留串行兼容路径。

示例（需要实际存在的检查点；此处不会自动执行）：

```bash
python -m unifiedgroupgen.generate generation.checkpoint=/path/model.best.pt \
  generation.kind=space generation.count=256 generation.batch_size=8 \
  generation.solver=heun generation.steps=64 generation.guidance=0 \
  generation.output=outputs/generation/example
```

## 8. 组成与部分结构补全

[MatterGen官方仓库](https://github.com/microsoft/mattergen)给出了固定原子类型的CSP训练/目标组成采样接口；DiffCSP的CSP同样把组成作为已知输入。因此“指定组成”不要求使用CFG。但生成器若从未学习组成→规则，仅靠硬掩码保证的是可行性，不是正确群/轨道的高概率。

新`learning=composition`以概率0.5把真实组成作为两阶段上下文，同时保留无组成任务；描述阶段仍使用剩余元素计数和轨道重数的硬可行性过滤。组成不经过性质dropout。`learning=multitask`同时训练组成和观测任务。只是推理时过滤旧Transformer不能补出这部分学习。

新补全采用[Palette](https://arxiv.org/abs/2111.05826)启发的**学习式掩码条件流匹配**。训练随机选择独立参数掩码、轨道块或空间局部patch，得到Az=c；从同一仿射纤维的先验起点走向真实终点，网络显式读入已知值和mask，损失监督未确定的合法方向。每步数值回缩保持观测。空间patch模式同时固定晶格，且留有未观测自由度。

[SCIGEN](https://github.com/RyotaroOKabe/SCIGEN)强调无需额外训练的采样约束。本方案的差别是增加显式掩码任务训练，不把“投影操作”本身当作差别；二者仍共享约束生成思想，也不能据此直接宣称方法创新已充分建立。

当前学习式补全的边界：必须给定群设置；完整已知轨道可作描述前缀，未完成轨道须提供完整描述及atom_map。不支持随意不相容的结构片段，也未实现同时推断未知群与原子对应关系。

`completion_method=masked_flow`会拒绝未训练过掩码任务的检查点；`projection_only`保留不依赖补全训练的硬投影基线；`auto`按检查点训练配置选择。二维展示CIF可能重新居中并增加真空，观测保留应按物理位置/晶格度量检查，而不是要求导出文件的真空c和分数z逐字相等。

## 9. 两阶段性质可见掩码

新模型每个样本先抽一次可见性质集合M及整体丢弃决定，然后分别送入两个独立的性质编码器：pθ(a|y_M)pφ(z|a,y_M)。参数不共享，但条件任务一致。各性质仍保留各自的MLP，不改成所有性质共用一个标量MLP。

旧独立掩码训练各阶段的条件边缘目标也可成立，并非数学错误；但同一样本的一阶段可能看形成能、另一阶段只看带隙，不能解释为一次共同条件任务。独立模式继续用于消融。CFG论文及[MatterGen实现](https://github.com/microsoft/mattergen/blob/main/mattergen/diffusion/sampling/classifier_free_guidance.py)支持有/无条件分支；并不构成“两阶段必须同掩码”的直接证明，本项目也不声称这是普遍强制规则。

特别是当前两阶段采用相加损失、且各自条件编码器参数独立：若每个阶段的掩码边缘分布相同，共享与独立抽样并不改变各项损失的期望。它改变的是同一样本任务的一致性和梯度噪声相关性，不能预先断言一定改善生成质量。是否改善须靠配对消融；移除来源与共享掩码也应分别比较，避免混淆原因。

## 10. 只检查方案的命令

```bash
python -m unifiedgroupgen.run experiment=new_joint_all task=plan
python -m unifiedgroupgen.run experiment=new_joint_all model=cspnet task=plan
python -m unifiedgroupgen.run experiment=new_joint_all model=mattergen task=plan
python -m unifiedgroupgen.run experiment=new_joint_all model=equiformer_v2 task=plan
python -m unifiedgroupgen.run experiment=new_joint_all prior=physical_metric task=plan
python -m unifiedgroupgen.run experiment=new_joint_all prior=train_gaussian task=plan
python -m unifiedgroupgen.run experiment=new_joint_all learning=composition task=plan
python -m unifiedgroupgen.run experiment=new_joint_all learning=masked_completion task=plan
python -m unifiedgroupgen.run experiment=new_joint_all learning=multitask task=plan
python -m unifiedgroupgen.run experiment=new_joint_all learning.condition_mask_policy=independent task=plan
python -m unifiedgroupgen.run experiment=new_joint_all +learning.flow_path.sigma=0.1 task=plan
```

新配置需要新审计收据；原数据字节不变，不能伪造旧配置审计可用于新配置。正式训练前执行`python -m unifiedgroupgen.pl_data.prepare --mode audit --experiments configs/experiments_v2.yaml`，然后检查结果；本次未运行完整新15套数据审计或正式训练。改变输入、先验或骨架应建新输出目录，不覆盖旧模型，不能跨架构严格载入所有旧权重。

本机验证记录见历史工程记录（已清理）；中间检查日志在`intermediate/checks/model_extensions`。通过单元/集成检查不代表新配置已收敛或生成质量优于旧模型。

本次检查合计124项通过，3项CPU混合精度组合按预期跳过。覆盖三种官方核心的小型CPU/CUDA前后向、CUDA AMP（EqV2核心显式FP32）、参考尺寸CPU构建、打包推理、SG/LG公开批量入口、线性观测保留、JSONL/CIF哈希、原生Lightning保存/恢复、评估调度及复用。原15套全部严格载入，20组真实输入与原服务器代码配对通过；150个受保护模型/配置/历史文件哈希无变化。另有12组新配置组合检查通过。本节前一轮124项检查属于旧环境快照；统一环境的新检查及精度测量另见历史工程记录（已清理）。
