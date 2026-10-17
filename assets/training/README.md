# 训练曲线

用户已确认服务器停止，图中Joint为停止快照。横轴为完成轮数，从1开始。MP20重复epoch保留最后记录；未验证轮次不填0，验证曲线只连接真实验证点。竖虚线为已完成轮数；Joint第199轮的部分批次不作为完整epoch。

损失为描述与流损失之和；训练/验证条件机制不同，绝对值不用于不同数据集的性能排名。星号为各自最佳验证点。

![训练曲线](training_history.png)

[PDF](training_history.pdf) · [SVG](training_history.svg)
