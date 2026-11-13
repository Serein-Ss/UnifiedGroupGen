> 后续分层重构：最新目录、Hydra训练入口、续训和预训练初始化见[PACKAGE_REFACTOR.md](PACKAGE_REFACTOR.md)。下文保留先前迁移记录。

# 项目目录与复现入口（2026-10-30）

## 主目录约定

```text
UnifiedGroupGen/
  unifiedgroupgen/          可独立安装的唯一核心实现
    run.py                 Hydra训练、原生续训、预训练初始化
    common/                路径、检查点、日志设置、结构绘图等
    group_constraints/     群—轨道编译、反向编码、合法空间
    conditional/           性质编码、化学硬约束、结构观测
    pl_data/               编码、划分、边预算批次、Lightning数据模块
    pl_modules/            模型、骨架、生成目标、Lightning管理
    generation/            描述和连续采样、统一结构输出
    evaluation/            CSP、从头生成、性质指标及工作流
    tests/                 运行前检查、数值审计、unit测试和历史参考
    environment/           环境依赖
  configs/                 原实验YAML + run.yaml + experiment选择配置
  scripts/                 绘图与其他工具，原命令仅薄兼容入口
  data/
    scr/                   原始数据与来源恢复、数据分发包
    dataset/               群—轨道模型输入与测试夹具
  assets/                  研究图表与视频
  examples/                结构与群目录示例
  docs/                    文章、理论、方案与操作说明
  web/                     现有网页源码与发布材料
  tmp/                     临时处理与重构备份、验证证据
  outputs/                 完整模型；tests/保存短程与工程实验
  artifacts/               历史产物保持原身份，按既有规则解析
```

`scr`按用户命名保留。源CSV从父项目复制，父项目其他工作的数据未移动。
运行产生的 `__pycache__`、`.pytest_cache`、`*.egg-info` 是自动缓存/安装元数据。

## 安装与执行

```bash
conda activate cgdit
python -m pip install --no-deps --no-build-isolation -e './unifiedgroupgen[test,evaluation]'
python -m unifiedgroupgen.common.cli --help
python -m unifiedgroupgen.common.workflows --help
python -m unifiedgroupgen.tests.reference_workflows.train_main15 --plan
python -m pytest -q
```

新环境可按 `unifiedgroupgen/environment/environment.yml` 创建，再安装与GPU匹配的PyTorch及包依赖。
根目录 `pip install -e .` 仍兼容。根依赖txt是指向包内environment的兼容包含文件。

`ugg` 对应模型CLI；`ugg-run <工作流名称>` 对应包内复现入口。
旧 `python scripts/<name>.py` 仅执行包内同一源码，没有第二份模型实现。
绘图等支持代码保留在scripts和assets/source中，不在scripts保存训练数据或测试夹具。

安装到其他环境后从项目根目录执行，或者设置 `UGG_PROJECT_ROOT` 为含configs/data的工作目录。
工作流的Python子进程改用已安装模块，不依赖site-packages旁边存在scripts目录。

## 历史路径与证据

旧配置与检查点内嵌的路径不改字节，避免使原有配置SHA、训练身份及来源审计失效。
磁盘访问时统一映射：

| 原逻辑位置 | 当前实际位置 |
|---|---|
| datasets/ | data/scr/bundles/ |
| raw_data/、父项目data/ | data/scr/ |
| artifacts/source_recovery/ | data/scr/source_recovery/ |
| artifacts/full_verified_v1/等输入目录 | data/dataset/下同名目录 |
| tests/fixtures/ | data/dataset/test_fixtures/ |
| tests/ | unifiedgroupgen/tests/unit/ |
| src/unifiedgroupgen/ | unifiedgroupgen/ |

审计比较先解析新旧路径，再核对原始内容SHA；没有忽略或伪造哈希。
历史数据版本的撤销规则、组成约束、CFG、损失与采样预算均不改变。
历史归档内的源程序、环境或测试快照是实验出处，不作为维护中的第二套实现迁移。

现有评估pipeline的代码身份保护仍生效。文件布局变化不能被伪装成同一次旧评估；
旧样本可直接交给统一指标入口读取，不应为目录调整而重新生成或删除失败记录。

## 实验产物暂存位置

详见 `EXISTING_OUTPUT_LOCATIONS.md` 和 `TRAINING_OUTPUT_ARCHIVE.md`。
用户确认后，18套完整历史训练模型从新服务器ZIP复制到outputs，源文件原位保留。
本次没有新增正式训练/评估，也没有连接训练服务器；独立生成/评估批次仍未迁移。
