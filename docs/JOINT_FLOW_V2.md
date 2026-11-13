# 周期晶体联合流 v2：实现、复现与边界
版本：2026-11-13。主模型 EquiformerV2；同路径对照 JointAtomGraph。均从随机晶格＋空原子图生成。

## 实际实现
- Generator Matching 统一几何流、元素CTMC和轨道跳转。
- Branching Flows 提供分支、出生、删除与锚点思想；Trajectory Matching 提供有限有序轨道轨迹监督。
- 主干接收当前所有原子的周期半径图，支持周期自像和非周期法向，保留边界等距邻居壳层。
- 官方 EquiformerV2 的表示与注意力保持不变；新增上下文只注入l=0。使用原生门控激活选项，向量头不转成标量。
- 标量晶格头生成群不变基中的变化；笛卡尔向量经当前晶格的轨道切空间转换。相同轨道仅抽样一次元素。
- 输入没有真实终点、下一动作或数据集标签。性质编码器保持原实现。
- 公共坐标数组已在允许仿射空间内，不再学习法向残差；晶格公共系数也只沿允许子空间变化。
- 联合训练是真正的packed周期图单次核心前向；对照采用带mask的批注意力。采样对异步时间的活动样本合批。
- 轨道事件改变图后重新计算特征。freeze只冻结轨道，元素与几何继续；每个新轨道生命期最多一次MASK解掩码。
- 替换核明确采用死亡/重生，不用相同重数假设原子一一对应。临时分支与替换监督为可选增强。
- 辅助训练路径用六个独立派生随机流划分程序、训练时刻、结构时钟、初始几何、出生核和元素生命周期；预算回退只影响程序侧。时钟修复及种子兼容边界见[随机流修复说明](PATH_RNG_FIX_20261113.md)。

## 三项独立任务
无性质生成与限定元素集合使用协同MASK离散流；允许集合可以只出现一个或任意子集元素。
新的 flow_*_dng 默认 chemical_set_probability=0，为纯无条件独立训练；限定元素集合的独立训练覆盖为1，并对相同覆盖重新审计。旧 task_* 的兼容默认保持不变。
目标性质任务保持既有标量编码、可见性和dropout机制，不重新设计条件模块。
精确化学式CSP使用库存守恒的带元素轨道动作，避免独立MASK采样违反计量。其元素DFM项为零；仍用相同几何主干与动作读出。

## 配置和训练
主选择器：
flow_mp20_dng、flow_mp20_property、flow_mp20_csp；
flow_c2db51_dng、flow_c2db51_property、flow_c2db51_csp。

从项目根目录使用项目环境：

    python -m unifiedgroupgen.run experiment=flow_mp20_dng device=cuda:0
    python -m unifiedgroupgen.run experiment=flow_c2db51_dng device=cuda:0
    python -m unifiedgroupgen.run experiment=flow_mp20_dng model=joint_flow_atom_graph device=cuda:0

以上是正常训练入口，不是本次自动执行的完整训练。短测命令：

    python scripts/short_test_joint_flow_v2.py --device cuda:0 --steps 4

新检查点格式为ugg_joint_flow_v2，保存后端、几何核、配置、原生优化器和随机状态。旧ugg_joint_symmetry_v1继续加载原模型；不把旧权重直接用于新架构。旧task_*选择器仍保留。

## 批量生成
检查点已包含TRAIN词表和按类型群先验。默认无群输入，仍可指定类型：

    python -m unifiedgroupgen.joint sample --checkpoint PATH/model.pt --kind space --count 16 --batch-size 4 --steps 128 --device cuda:0 --output outputs/new_samples.jsonl

层材料用--kind layer。指定群可增加--group，CSP额外指定--formula和--formula-units或--composition。
每条样本的随机数源独立；打包仅加速前向，不让图间消息交叉。账本保留每次尝试、未完成、数值失败与验证拒绝。

可选完整结构轨迹：增加 `--trajectory-dir outputs/run/trajectories` 保存每次尝试；
增加 `--trajectory-indices 0 7` 只记录全局编号0和7。记录空图、每个已接受连续步、离散事件后状态及最终状态，
逐帧写入独立JSONL，最终账本保留轨迹路径。默认关闭，不改变原输出。详见[轨迹格式与使用](GENERATION_TRAJECTORIES.md)。

## 数学与等变边界
本版保证原生SO(3)旋转表示契约；EquiformerV2原论文本身不考虑反演。完整晶体群由硬约束保证，不宣称主干已具全局O(3)等变。
当前坐标路径是在未折返轨道仿射图中的线性桥，不宣称最短周期测地线。
当前几何损失是约束内公共参数MSE，不宣称额外能量或通用物理度量。
显式框架朝向使旋转测试可同时旋转晶格和原子；基底/原点规范另行处理。
有限步长、epsilon和Heun风险定位均为数值近似，必须记录未完成分母。
轨道数、原子数及事件预算有限；标准自由生成禁止删除最后一个轨道，最后可用事件冻结配置。CSP沿用库存完成见证与失败机制。
CPU教师路径仍逐记录构造；已批量化的是神经计算和活动采样图，尚未声称整个流水线完全GPU化。

## 测试与计时
测试真实MP-20/C2DB-51端点、三个独立任务、两主干、空图、完整跳转核、批图/单图一致、旋转、周期邻居壳层、层群/平面群及原生恢复。
短跑每项只做少量优化步，选取有限且带标签的小结构；不证明全数据训练、收敛、生成质量或性质控制。
训练区分端到端时间和热身后的CUDA同步纯计算中位数。生成报告尝试数、完成数、事件数和批大小。不同主干参数量不相同，不把吞吐比较当作公平材料质量比较。
原始报告位于intermediate/checks/joint_flow_v2_20261113，网页只展示方法与验证范围，不发布短跑模型结果。


## 2026-11-13 概率闭环补充

现行理论主稿第37—43节及[完整补充](PROBABILITY_CLOSURE_20261113.md)区分理想过程和实际网络。有限预算保证最多2B次事件；在共享目的核、合法有限程序与满秩Gaussian先验等前提下，理想边缘过程有有限截断存在唯一性与正确终端，其系数弱距离截断为O(epsilon)。这不是有限训练网络的终端保证。硬邻居切换、固定根容差、监督时间尾部及未完全总化的异常仍需处理；生产采样器保持不变。性质/库存的全条件数据目标还要求TRAIN群权重与群内目标共同校准；当前固定/可行性群先验仅定义受控混合。

`scripts/check_probability_closure.py`是r5历史独立诊断，依赖当时的受保护源码基线，并写入本地站点；不能直接把该旧基线检查当作修复后代码的验证。本补丁使用`python -m pytest unifiedgroupgen/tests/unit/test_joint_flow_path_rng.py -q`核对当前生产路径，旧r5报告与基线保留。
