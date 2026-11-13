# 可清理的中间目录

checks保存运行前检查和测试输出，cache保存可重建缓存，staging保存临时解包及转换工作区。相关进程结束后可以删除其内容；需要时由程序重新创建。

模型训练、生成与指标在outputs；原始数据在data/scr，编码数据与必要审计凭据在data/dataset；成品图表在assets。最新小型核验摘要在docs/validation，不依赖这里的旧日志。旧模型归档及过期检查输出已清理。
