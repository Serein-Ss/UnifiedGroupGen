# 旧模型1024次评估：保留现有采样的CPU并行调度

本调度器只管理 `artifacts/evaluations/legacy_1024_cpu_seed142` 中原有的33项任务。
不训练模型，不修改15套训练的代码、配置、进程、CPU亲和性或模型权重。
每组仍是1024次尝试，包含失败；原有种子、积分步数、CFG、性质预测器和科学评估代码均保持不变。

## 调度与资源保护

- 接管时先校验冻结代码、三套旧权重、预测器、训练/验证/测试数据哈希，以及已完成结果的文件哈希。
- 短暂停住经命令行核实的旧串行**主控**，再次检查子任务，随后仅终止主控。
  当前采样解释器和其子进程继续运行，PID和已生成内容保留；接管记录写入 `parallel_handoff_*.json`。
- 初始总并发4路，包含接管的采样进程。每路内部只用1个CPU线程；所有新子进程屏蔽CUDA。
- 读取CPU亲和性、cgroup v1/v2配额和内存上限，预留至少4个有效CPU；并发最高8路。
  内存预算取启动时可用内存的四分之一与16 GiB中的较小值。
  若容器资源不足4路，启动前拒绝切换，旧任务继续。
- 新评估进程使用 `nice 19`、空闲I/O优先级和独立CPU亲和性；不调整训练进程。
  接管的现有评估进程也会限制CPU亲和性。训练仍可能使用这些CPU，故不能保证完全没有争用。
- 用同一正式训练模型切换前最近3至5个已完成轮次的训练吞吐中位数作基线。
  新完成至少3轮且最近3轮中位数达到基线95%，才允许升到8路（受资源上限限制）。
  新模型开始时重新建立基线，回到4路。该基线包含原先串行评估的开销，不是无评估基线。
- 最近两轮吞吐低于基线90%时，将评估CPU集合缩到2核并暂停新派发，保留现有任务；
  最近3轮低于95%时限制为4路。训练PID异常、进度长时间未更新、可用内存不足4 GiB或评估RSS超过预算时也暂停派发。
  这是软保护，不是cgroup硬内存隔离；已有工作不会被强杀，RSS不会被立即回收。
- 每个任务按依赖启动：模型测试→采样→性质→标准指标。
  有结果可处理时优先后处理。共享同一性质参考缓存的两组任务串行，避免缓存写入竞争。
  已完成、相同数据/参考配置的指纹缓存可复制给后续指标任务，指标程序还会独立核验缓存身份。

## 服务器部署：不更新正在训练的工作树

下面的 `REV` 替换为本次发布提交号。只fetch并将新增脚本解压到独立工具目录；
不要在15套训练期间执行 `git pull`、重新安装依赖或修改 `src`。

```bash
cd /root/private_data/rszhong/workspace/UnifiedGroupGen
conda activate cgdit
git fetch origin main
REV=<本次发布提交号>
TOOLS="$PWD/artifacts/evaluation_tools/parallel_${REV}"
mkdir -p "$TOOLS"
git archive "$REV" scripts/parallel_legacy_evaluation.py docs/PARALLEL_LEGACY_EVALUATION.md | tar -x -C "$TOOLS"
export UGG_PAR_TOOL="$TOOLS/scripts/parallel_legacy_evaluation.py"

# 以下两条只读，不停任务。plan核验较多文件，可能耗时。
python "$UGG_PAR_TOOL" --project-root "$PWD" --resources
nice -n 19 ionice -c 3 python "$UGG_PAR_TOOL" --project-root "$PWD" --plan > "$TOOLS/plan.json"
```

资源检查中 `capacity` 必须至少4；达到8才具备升到8路的资源条件。
`effective_cpus` 才是考虑配额后的CPU数量，不能只看服务器标称64核。
如计划核验报错，先查看错误，不切换、不删锁、不清空输出。

```bash
# 真正切换：只结束经核验的串行评估主控，保留当前采样和全部已完成结果。
nohup nice -n 19 ionice -c 3 python -u "$UGG_PAR_TOOL" \
  --project-root "$PWD" --run --takeover-serial \
  > artifacts/evaluations/legacy_1024_cpu_seed142.parallel.driver.log 2>&1 &
echo "并行评估主控PID：$!"
```

不得同时启动另一个旧串行调度器。原 `pipeline.lock` 保留为屏障；新增 `parallel.lock` 防止重复并行主控。
脚本会在接管过程中再核验一次，防止只读检查与切换之间产生时序变化。

## 查看进度与训练影响

```bash
python "$UGG_PAR_TOOL" --project-root "$PWD" --status
watch -n 30 "python '$UGG_PAR_TOOL' --project-root '$PWD' --status"
# 退出watch按Ctrl+C，不会停止后台任务。
tail -n 30 -F artifacts/evaluations/legacy_1024_cpu_seed142.parallel.driver.log
nvidia-smi
ps -ww -eo pid,ppid,ni,psr,pcpu,pmem,rss,etime,args --sort=-pcpu | head -n 25
```

状态显示任务完成数、复用数、活动PID、尝试进度、评估RSS、并发上限、正式训练状态、吞吐基线及当前保护原因。
单个采样的ETA使用它自己的累计实测速度；并行任务的剩余时间不能直接相加。
性质/指标首次参考缓存较慢，缺少实测时不提供虚假的全流程ETA。
完成判据为 `state: complete` 且 `33/33`。训练PID与 `nvidia-smi` 中GPU进程应仍为正式训练进程。

## 异常后的恢复

若并行主控已退出，子任务可仍在运行。先看状态和日志，再用同一个固定版本恢复：

```bash
nohup nice -n 19 ionice -c 3 python -u "$UGG_PAR_TOOL" \
  --project-root "$PWD" --run --resume \
  >> artifacts/evaluations/legacy_1024_cpu_seed142.parallel.driver.log 2>&1 &
echo "恢复主控PID：$!"
```

活动主控存在时会拒绝重复启动。恢复时重新核验已完成输出，接管仍活着的原子任务。
已完成的采样不会重跑；主控故障前已写完但未登记的结果可检查完整性后登记。
部分采样若已没有活的原进程，因旧采样没有保存随机数中间状态，脚本会拒绝覆盖；需另行审计后制定恢复办法。
失败的子任务明确记录为失败，不静默改写样本或改变统计分母。

## 本地验证范围

`tests/test_parallel_legacy_evaluation.py` 覆盖依赖与缓存互斥、4→8及降速保护、配额约束、
仅向主控发送信号、保留采样和训练进程、PID复用防护、CPU环境、输出哈希与失败尝试保留、接管后完成登记和重复主控保护。
本地为Windows，因此Linux `/proc`、信号及亲和性使用受控模拟；服务器实际资源、接管PID与吞吐仍须通过上述检查确认。
