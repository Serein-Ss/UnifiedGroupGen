# joint_flow_v2 生成轨迹保存

该功能只观察采样状态，不训练、不改变网络、积分器、事件核或采样概率。
默认不启用，原来的最终结果 JSONL 不变。适用于 EquiformerV2 和 JointAtomGraph 的
joint_flow_v2 检查点；旧生成器遇到轨迹参数会明确报错，不改变其采样方式。
新v2入口同时写摘要和状态文件，命名与默认目录见[统一输出规范](OUTPUT_LAYOUT.md)。
配套文件不改变账本行、轨迹帧或采样随机数，但输出目录会增加这些文件。

## 命令

从项目根目录运行。以下 `PATH/model.pt` 替换为实际 v2 检查点；输出文件与已有轨迹不能重名。

单个结构，保存从空图到结束的所有已接受状态：

```powershell
python -m unifiedgroupgen.joint sample --checkpoint PATH/model.pt --kind space --count 1 --batch-size 1 --steps 128 --device cuda:0 --output outputs/trace_single/samples.jsonl --trajectory-dir outputs/trace_single/trajectories
```

批量生成，保存全部32次尝试的轨迹：

```powershell
python -m unifiedgroupgen.joint sample --checkpoint PATH/model.pt --kind layer --count 32 --batch-size 4 --steps 128 --device cuda:0 --output outputs/trace_batch/samples.jsonl --trajectory-dir outputs/trace_batch/trajectories
```

仍生成32次，但仅保存全局编号0、7、15的轨迹：

```powershell
python -m unifiedgroupgen.joint sample --checkpoint PATH/model.pt --kind layer --count 32 --batch-size 4 --steps 128 --device cuda:0 --output outputs/trace_selected/samples.jsonl --trajectory-dir outputs/trace_selected/trajectories --trajectory-indices 0 7 15
```

编号从0开始，范围为 `[0, count)`，不是批次内编号，也不是随机种子。
只给 `--trajectory-dir` 就保存全部；不给该参数则不保存轨迹。
这些参数可与原有 `--group`、`--chemical-system`、`--formula`、`--formula-units` 等组合，
仍须使用匹配任务的检查点。例如 CSP 的 Ti₂O₄ 使用 `--kind space --formula TiO2 --formula-units 2`；
MoS₂ 使用 `--kind layer --formula MoS2 --formula-units 1`。

## 输出与“每一步”的含义

```text
outputs/trace_selected/
  samples.jsonl
  summary.json
  sampling.status.json
  trajectories/
    sample_000000.jsonl
    sample_000007.jsonl
    sample_000015.jsonl
```

每条轨迹是独立、逐行写盘的 JSONL；不会把整批的所有帧留在内存中。
每个批次结束后关闭相应文件。最终账本的对应行新增 `trajectory_path` 和 `trajectory_frames`；
未选中样本不新增这两个字段。采样摘要另给出轨迹目录和已写完的轨迹数量。

每个文件按以下顺序保存：

1. `record_type=metadata`：格式版本、全局样本编号、种子、原请求群、完整编译规范
   （操作及轨道模板）、化学控制、性质请求、检查点路径、配置与采样参数。
2. `record_type=frame, stage=initial`：时间0的随机相容晶格和空原子图。
3. `stage=continuous`：每次真正接受的 Heun 连续更新，包括落到事件时间的更新。
4. `stage=event`：该时刻离散动作执行后的完整状态，附动作、动作前后原子数。
   前一个 continuous 帧保留事件前的状态，因此事件两侧都可查看。
5. `stage=final`：结束状态及结束原因，成功和采样失败都保存。
6. `record_type=result`：最终账本行，含结构检查、接受/拒绝状态及能够导出的 CIF。

这里的“每一步”是采样器接受的实际结构状态，不包括 Heun 临时预测器、神经网络内部层、
事件二分搜索的试探点。一般帧数为 `2 + integration_steps + len(events)`。
final 帧可能与最后一个更新有相同几何，但会额外标记终止状态和原因。
一个事件前后时间可以相同，节点数和元素可以不同；不要按时间去重这些帧。

## 每帧内容与坐标约定

- `frame_index`、`stage`、`time`、`integration_step`：顺序、阶段、连续时间与积分计数。
- `num_atoms`、`letters`、`orbit_elements`、`atomic_numbers`、`orbit_index`：当前图的节点和轨道。
  元素0表示 MASK，不能直接当作真实化学元素导出 CIF。
- `metric_state`、`coordinate_state`：原始公共几何状态。
- `lattice`：当前物理晶格，晶格矢量按列；尚未人为加入真空。
- `coordinates`：周期分量折回的坐标；层材料的非周期 z 是 Å 高度。
- `coordinates_unwrapped`：未折回的合法轨道坐标，便于查看连续桥。
- `cartesian_coordinates`：当前笛卡尔位置，包含已有整体取向和原点设置。
- `frame`、`origin`、`periodic`：取向、原点及周期方向；未设置时前两项为 null。
- `frozen`、`orbit_step`、`budget`、`state_status`、`reason`：编辑与终止状态。
- `chemical_control`、`remaining_inventory`：元素支持或精确组分库存。
- `geometry_finite`、必要时的 `decode_error`：异常几何标记。

原子和轨道下标属于当前帧；删除、替换后不能把相同下标无条件视为同一个持久原子。
离散事件与每帧轨道信息共同描述变化。不保存神经网络隐藏特征、注意力、速度和概率数组，
也不逐帧重复保存周期边；构图参数在头部配置中，边可从当前结构按原构图规则重建。
中间状态允许 MASK、空图和近距离原子，因此不把它们强制转换成普通 CIF。
最终 CIF 仍由原有最终结构导出器生成；二维 CIF 添加真空后与内部坐标约定有区别。

## 失败、异常与复现边界

无可行群而未启动采样的尝试，仅有 metadata 和 result，没有伪造的空图或几何帧。
采样失败保存已走过的轨迹；未完成、被拒绝的样本不会被悄悄删除。
无法解码的异常状态仍保存原始参数；非有限数写为 null 并标记无效，文件不含非法 NaN JSON。
若未捕获异常中断整个调用，已打开轨迹会尽可能追加 `interrupted` 记录并关闭；
这不等于为原采样器补齐了全部异常恢复机制。进程被强制终止时可能只有已写入的部分帧。

记录器不消耗随机数，不修改状态。两主干及 SG/LG/PG 的开关对比使用相同的样本生成器
和全局随机状态；原生 EqV2 的辅助边参考系也使用全局 RNG，因此仅给样本种子
不意味着跨环境、跨批大小都能逐位复现。保留实际检查点和启动配置；轨迹头不包含模型权重。
逐帧 CPU 传输与写盘会增加时间和磁盘开销，批量任务可只选择少数样本编号。

## Python 采样接口

`sample_batch(..., trajectory_callback=callback)` 和 `sample_one(..., trajectory_callback=callback)`
支持观察接口，不自动向 JointSample 塞入全部帧：

```python
callback(local_index, stage, state, time, integration_step, event, reason)
```

local_index 是当前调用的批次内编号。调用者应只读取 state，不修改张量、不抽取随机数。
命令行适配器将它映射到全局样本编号并交给 `TrajectoryWriter` 流式写盘。
直接使用 Writer 时，先 start，每条采样结束后 finish，外层使用 with 保证文件关闭。

测试入口：`python -m pytest unifiedgroupgen/tests/unit/test_generation_trajectories.py -q`。
上述测试检查轨迹与数值采样契约，不构成材料质量或完整训练结果。
