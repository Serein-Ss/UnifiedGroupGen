# UnifiedGroupGen

配置仓库根目录的 `.env` 后，可从项目根目录检查训练计划。

```bash
python -m unifiedgroupgen.run task=plan experiment=full_mp20_eqv2_dng
```

`ugg` 是训练入口的控制台命令。训练配置在项目的 `configs/`，数据预处理使用 `data/process.py`。生成和评估分别使用 `python -m unifiedgroupgen.generate` 和 `python -m unifiedgroupgen.evaluate`。

详见[核心代码布局](CORE_LAYOUT.md)、[项目入口](../README.md)和[安装环境](../docs/ENVIRONMENT.md)。
