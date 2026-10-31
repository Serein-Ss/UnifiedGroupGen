# 项目目录与复现入口

核心库为unifiedgroupgen；环境与安装配置均在该目录。configs保存实验参数，scripts保存绘图/汇总工具，检查代码在unifiedgroupgen/tests。

data/scr保存原始数据，data/dataset保存模型输入与永久数据审计。outputs保存训练模型、日志与正式生成评估；assets保存展示图表，docs保存说明与参考源码，examples保存真实材料示例，web保存本地网页。

intermediate统一承接可重新生成的检查日志、缓存和临时工作区，实验结束后可清理。artifacts已拆分，不再保留物理目录。完整规则和当前评价位置见[FILE_STORAGE.md](FILE_STORAGE.md)及[EXISTING_OUTPUT_LOCATIONS.md](EXISTING_OUTPUT_LOCATIONS.md)。旧隔离区暂不删除。
