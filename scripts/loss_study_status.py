"""Read-only progress, phase timings, losses, GPU memory and workload ETA."""

import argparse
from datetime import datetime
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ('mp20', 'c2db51', 'joint')
SOURCES = {'mp20': ('mp_20',), 'c2db51': ('c2db_51',), 'joint': ('mp_20', 'c2db_51')}
TRAIN_PRIOR = {'mp20': 56.24, 'c2db51': 19.25, 'joint': 80.17}
VAL_PRIOR = {'mp20': 113.02, 'c2db51': 38.68, 'joint': 161.08}
SAMPLE_PRIOR = {('mp_20', 'unconditional'): 262.61/256, ('mp_20', 'conditional'): 491.79/256,
                ('c2db_51', 'unconditional'): 397.30/256, ('c2db_51', 'conditional'): 511.07/256}


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def read_rows(path):
    if not path.exists():
        return []
    lines = path.read_text(encoding='utf-8').splitlines()
    rows = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may still be appending the final line.
            if index != len(lines)-1:
                raise
    return rows


def median(values, fallback):
    return statistics.median(values) if values else fallback


def snapshot(study, now=None):
    now = time.time() if now is None else now
    protocol = read_json(study/'protocol.json')
    state = read_json(study/'status.json')
    epochs, interval = protocol.get('epochs', 20), protocol.get('validation_interval', 5)
    runs, calibration = [], {}
    for dataset in DATASETS:
        for version in ('before', 'after'):
            folder = study/'runs'/dataset/version
            history = read_rows(folder/'model.history.jsonl')
            gradients = read_rows(folder/'model.gradients.jsonl')
            progress = read_json(folder/'model.status.json')
            done = max((r['epoch']+1 for r in history), default=0)
            last, valid = (history[-1] if history else {}), [r for r in history if r.get('val')]
            runs.append({'dataset': dataset, 'version': version, 'epochs': done, 'history': history,
                'progress': progress, 'train': last.get('train', {}), 'val': valid[-1] if valid else {},
                'gradient': gradients[-1] if gradients else {}, 'clip_fraction': last.get('gradient_clipped_fraction'),
                'training_seconds': sum(r.get('training_seconds', 0) for r in history),
                'validation_seconds': sum(r.get('validation_seconds', 0) for r in history),
                'save_seconds': sum(r.get('checkpoint_seconds', 0) for r in history),
                'allocated_GiB': max((r['peak_cuda_bytes']/2**30 for r in history if 'peak_cuda_bytes' in r), default=None),
                'reserved_GiB': max((r['peak_cuda_reserved_bytes']/2**30 for r in history if 'peak_cuda_reserved_bytes' in r), default=None),
                'status_age_seconds': now-(folder/'model.status.json').stat().st_mtime if progress else None})
    for dataset in DATASETS:
        histories = [r['history'] for r in runs if r['dataset'] == dataset]
        warm = [h['training_seconds'] for history in histories for h in history[-5:]
                if h['epoch'] > 0 and h.get('training_seconds', 0) > 0]
        scale = calibration.get('mp20', {}).get('train', TRAIN_PRIOR['mp20'])/TRAIN_PRIOR['mp20']
        train = median(warm, TRAIN_PRIOR[dataset]*scale)
        validation = [h['validation_seconds'] for history in histories for h in history
                      if h.get('validated') and h.get('validation_seconds', 0) > 0]
        saves = [h['checkpoint_seconds'] for history in histories for h in history if h.get('checkpoint_seconds', 0) > 0]
        cold = [max(history[0]['training_seconds']-train, 0) for history in histories
                if history and 'training_seconds' in history[0]]
        calibration[dataset] = {'train': train, 'val': median(validation, VAL_PRIOR[dataset]*train/TRAIN_PRIOR[dataset]),
                                'save': median(saves, 1.1), 'cold_extra': median(cold, 335.0),
                                'train_measured': bool(warm), 'val_measured': bool(validation)}
    for run in runs:
        budget = calibration[run['dataset']]
        remaining = 0.0
        progress = run['progress']
        for epoch in range(run['epochs'], epochs):
            train = budget['train']+(budget['cold_extra'] if epoch == 0 else 0)
            val = budget['val'] if (epoch+1) % interval == 0 or epoch == epochs-1 else 0
            fraction = min(1.0, progress.get('step', 0)/max(progress.get('steps', 1), 1))
            if progress.get('epoch') == epoch:
                if progress.get('phase') == 'train':
                    if epoch > 0 and fraction > 0 and progress.get('seconds', 0) > 0:
                        train = progress['seconds']/fraction
                    train *= 1-fraction
                elif progress.get('phase') == 'validation':
                    train = 0.0
                    val *= 1-fraction
            remaining += train+val+budget['save']
        active = state.get('current_job') == 'train/'+run['dataset']+'/'+run['version']
        if run['epochs'] == 0 and not progress and not active:
            remaining += 65.0  # Earlier full-process startup overhead, once per future worker.
        run['training_remaining_seconds'] = remaining
    samples, test_done, metric_done, evaluation_done = [], 0, 0, 0
    for run in runs:
        folder = study/'evaluations'/run['dataset']/run['version']
        test_done += int((folder/'test.json').exists())
        evaluation_done += int((folder/'complete.json').exists())
        for source in SOURCES[run['dataset']]:
            for mode, count in (('unconditional', protocol.get('unconditional_count', 256)),
                                ('conditional', protocol.get('conditional_count', 128))):
                job = folder/(source+'_'+mode)
                status = read_json(job/'sampling.status.json')
                samples.append({'key': (source, mode), 'requested': count,
                                'attempted': min(count, status.get('attempted', 0)),
                                'seconds': status.get('seconds', 0)})
                metric_done += int((job/'standard_metrics.json').exists())
    sampling_remaining = 0.0
    for sample in samples:
        peers = [s for s in samples if s['key'] == sample['key'] and s['attempted'] >= 10 and s['seconds'] > 0]
        rate = (sum(s['seconds'] for s in peers)/sum(s['attempted'] for s in peers)
                if peers else SAMPLE_PRIOR[sample['key']])
        sampling_remaining += max(sample['requested']-sample['attempted'], 0)*rate
    test_remaining = sum(2*calibration[d]['val']*(1-sum(
        int((study/'evaluations'/d/v/'test.json').exists()) for v in ('before', 'after'))/2) for d in DATASETS)
    # CPU fingerprint/matching timings are not instrumented; this is a planning allowance.
    cpu_fraction = max(0, (len(samples)-metric_done)/len(samples))
    training_remaining = sum(r['training_remaining_seconds'] for r in runs)
    estimated = training_remaining+sampling_remaining+test_remaining
    return {'state': state, 'epochs': epochs, 'runs': runs, 'calibration': calibration,
            'test_done': test_done, 'evaluation_done': evaluation_done, 'metrics_done': metric_done,
            'metric_jobs': len(samples), 'attempted': sum(s['attempted'] for s in samples),
            'requested': sum(s['requested'] for s in samples), 'training_remaining_seconds': training_remaining,
            'sampling_remaining_seconds': sampling_remaining, 'test_remaining_seconds': test_remaining,
            'cpu_reserve_seconds': [3600*cpu_fraction, 10800*cpu_fraction],
            'total_remaining_seconds': ([0, 0] if state.get('state') == 'complete' else
                                         [estimated+3600*cpu_fraction, estimated+10800*cpu_fraction])}


def number(value, digits=3):
    return '--' if value is None else f'{value:.{digits}f}'


def duration(seconds):
    minutes = max(0, round(seconds/60))
    return f'{minutes//60}h{minutes%60:02d}m'


def display(study, result):
    state, epochs = result['state'], result['epochs']
    print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), '|', study)
    print('流程:', state.get('state', '尚未写入状态'), '| PID:', state.get('pid'), '| 当前:', state.get('current_job', '--'))
    pid = state.get('pid')
    if pid and sys.platform.startswith('linux'):
        cmdline = Path(f'/proc/{pid}/cmdline')
        try:
            command = cmdline.read_bytes()
            alive = any(name in command for name in (b'loss_balance_study.py', b'resume_loss_evaluation.py'))
        except OSError:
            alive = False
        print('主控进程:', '存在' if alive else '不存在或PID已被复用')
        if not alive and state.get('state') != 'complete':
            print('注意：主控未运行；下列ETA仅表示恢复执行后的剩余工作量。')
    if state.get('state') == 'queued':
        print('队列等待时间未计入ETA；下列估算是获得GPU后的剩余工作量。')
    if state.get('error'):
        print('错误:', state['error'])
    print('\n模型 | 轮次/阶段/批次 | 累计训练min | 验证min | 保存s | 峰值分配/保留GiB | 剩余训练')
    for run in result['runs']:
        p = run['progress']
        phase = '完成' if run['epochs'] >= epochs else p.get('phase', '等待')
        location = f"{run['epochs']}/{epochs} {phase} {p.get('step', '-')}/{p.get('steps', '-')}"
        if run['status_age_seconds'] is not None and run['epochs'] < epochs:
            location += f" ({run['status_age_seconds']:.0f}s前更新)"
        print(f"{run['dataset']}/{run['version']} | {location} | {run['training_seconds']/60:.1f} | "
              f"{run['validation_seconds']/60:.1f} | {run['save_seconds']:.1f} | "
              f"{number(run['allocated_GiB'], 2)}/{number(run['reserved_GiB'], 2)} | {duration(run['training_remaining_seconds'])}")
    print('累计耗时和峰值来自已保存轮次；当前未结束轮次尚未计入。')
    print('\n模型 | 训练总* | 描述 | 晶格 | 坐标原始 | 统一比较值** | 最近验证总*(轮次) | 最近g_k/g_q加权 | 裁剪比例')
    for run in result['runs']:
        t, v, g = run['train'], run['val'], run['gradient']
        total = t.get('descriptor', 0)+t.get('flow', 0) if t else None
        common = t['descriptor']+t['lattice']+t['coordinates']/9 if t else None
        print(f"{run['dataset']}/{run['version']} | {number(total)} | {number(t.get('descriptor'))} | "
              f"{number(t.get('lattice'))} | {number(t.get('coordinates'))} | {number(common)} | "
              f"{number(v.get('val_loss'))}({v['epoch']+1 if v else '-'}) | "
              f"{number(g.get('shared_lattice_weighted_norm'))}/{number(g.get('shared_coordinates_weighted_norm'))} | "
              f"{number(run['clip_fraction'])}")
    print('* 两组总损失定义不同，不直接比较。** 两组均按描述+晶格+原始坐标/9重算。')
    print(f"\n评估：测试集 {result['test_done']}/6；采样 {result['attempted']}/{result['requested']}；"
          f"指标子任务 {result['metrics_done']}/{result['metric_jobs']}；模型评估 {result['evaluation_done']}/6")
    print('剩余训练:', duration(result['training_remaining_seconds']), '| 剩余测试损失:', duration(result['test_remaining_seconds']),
          '| 剩余采样:', duration(result['sampling_remaining_seconds']))
    print('CPU指标预算预留（非实测ETA）:', '～'.join(duration(s) for s in result['cpu_reserve_seconds']))
    print('全部剩余时间粗估:', '～'.join(duration(s) for s in result['total_remaining_seconds']))
    print('ETA优先用本次后续轮次中位数；未开跑数据集按MP20实测速率折算，缺测时使用旧4090基准。')
    print('CPU预算按未完成指标子任务比例粗分配；首次缓存、结构大小和CPU负载会改变耗时。')
    for dataset, c in result['calibration'].items():
        print(f"  {dataset}: 训练 {c['train']:.1f}s/轮 {'本次实测' if c['train_measured'] else '估算'}；"
              f"验证 {c['val']:.1f}s/次 {'本次实测' if c['val_measured'] else '估算'}")
    job = state.get('current_job')
    log = study/(job.replace('/', '_')+'.log') if job else None
    if log is not None and log.exists():
        with log.open('rb') as stream:
            stream.seek(max(0, log.stat().st_size-2048))
            tail = stream.read().decode('utf-8', errors='replace').splitlines()[-3:]
        print(f'\n当前子任务日志（{time.time()-log.stat().st_mtime:.0f}s前更新）: {log.name}')
        for line in tail:
            print(line[:240])
    try:
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu',
                              '--format=csv,noheader'], capture_output=True, text=True, timeout=5, check=True)
        print('\n实时GPU（整卡，含其他进程）:', gpu.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        print('\n实时GPU状态暂不可读；上表仍保留历史峰值。')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='artifacts/loss_balance_4090_seed42')
    args = parser.parse_args()
    study = (ROOT/args.study).resolve()
    if not study.is_dir():
        parser.error('Study directory does not exist: '+str(study))
    display(study, snapshot(study))


if __name__ == '__main__':
    main()
