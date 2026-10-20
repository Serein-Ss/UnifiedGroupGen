"""Build PNG comparisons from the three archived training runs, without retraining."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [('M1', 'MP20单来源', 'local_large_seed42/mp20', '#437da6'),
        ('M2', 'MPTS52单来源', 'local_large_seed42/mpts52', '#c57b40'),
        ('M3', '三来源联合', 'joint_space_layer_large_seed42', '#508c76')]


def save_json(path, value):
    path.write_bytes((json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=ROOT.parent/'tmp/UnifiedGroupGen.zip')
    parser.add_argument('--output', type=Path, default=ROOT/'assets/results/model_comparison_20261021')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    inputs, histories, models, diagnostics, tests = {}, {}, [], [], []
    with ZipFile(args.archive) as archive:
        names = archive.namelist()

        def read(suffix, lines=False):
            matches = [n for n in names if n.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError('Expected exactly one archived file: '+suffix)
            raw = archive.read(matches[0])
            inputs[suffix] = hashlib.sha256(raw).hexdigest()
            return [json.loads(l) for l in raw.decode('utf-8-sig').splitlines() if l.strip()] if lines else json.loads(raw)

        def generation(model, dataset, suffix, mode):
            records = read(suffix+'/samples.jsonl', lines=True)
            summary = read(suffix+'/summary.json')
            good = [r for r in records if r['status'] == 'generated']
            reports = [r['validation'] for r in good]
            n = len(records)
            assert n == summary['attempted'] and len(good) == summary['generated']
            counts = {k: sum(bool(r.get(k)) for r in reports)
                      for k in ('group_operation_closure', 'distance_valid', 'no_coincident_orbits')}
            for k, count in counts.items():
                assert count == summary['counts'][k]
            exact = sum(bool(r.get('exact_target_group')) for r in reports)
            assert exact == summary['exact_group_success']
            conditions = [r['properties_requested'] for r in records]
            assert all(c == conditions[0] for c in conditions)
            diagnostics.append({'model': model, 'dataset': dataset, 'mode': mode, 'attempted': n,
                'generated': len(good), 'exact_target_group': exact,
                'unknown_group': sum(r.get('exact_target_group') is None for r in reports),
                **counts, 'requested_groups': len({r['descriptor']['number'] for r in good}),
                'properties_requested': conditions[0], 'guidance': 0 if mode == 'unconditional' else 2,
                'steps': 64, 'sampling_seed': 142, 'metric_protocol': 'raw_generation_diagnostics'})

        for ident, label, run, color in RUNS:
            suffix = 'artifacts/runs/'+run
            raw = read(suffix+'/model.history.jsonl', lines=True)
            rows = sorted({r['epoch']: r for r in raw}.values(), key=lambda r: r['epoch'])
            status = read(suffix+'/model.status.json')
            assert [r['epoch'] for r in rows] == list(range(status['completed_epochs']))
            validation = [r for r in rows if r.get('val_loss') is not None]
            best = min(validation, key=lambda r: r['val_loss'])
            assert best['val_loss'] == status['best_val_loss']
            histories[ident] = rows
            models.append({'id': ident, 'label': label, 'completed_epochs': len(rows),
                'best_epoch': best['epoch']+1, 'best_val_loss': best['val_loss'],
                'validation_points': len(validation), 'duplicate_epoch_rows_removed': len(raw)-len(rows),
                'train_records': status['train_records'], 'val_records': status['val_records']})
            if ident != 'M3':
                report = read(suffix+'/test.json')
                assert report['checkpoint_epoch'] == best['epoch']
                tests.append({'model': ident, 'dataset': 'MP20' if ident == 'M1' else 'MPTS52',
                    'requested': report['requested_records'], 'evaluated': report['evaluated_records'],
                    **report['loss_on_supported_records']})
                for mode in ('unconditional', 'conditional'):
                    generation(ident, 'MP20' if ident == 'M1' else 'MPTS52', suffix+'/'+mode, mode)
        report = read('artifacts/evaluations/joint_best_seed142/test/attempt_001/result.json')
        for dataset, r in report['source_breakdown'].items():
            tests.append({'model': 'M3', 'dataset': {'mp_20':'MP20', 'mpts_52':'MPTS52', 'c2db_51':'C2DB'}[dataset],
                'requested': r['requested_records'], 'evaluated': r['evaluated_records'],
                **r['loss_on_supported_records']})
            for mode in ('unconditional', 'conditional'):
                generation('M3', tests[-1]['dataset'],
                    'artifacts/evaluations/joint_best_seed142/'+dataset+'_'+mode+'/attempt_001', mode)

    with (out/'all_model_training_history.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model', 'epoch', 'train_total', 'train_descriptor', 'train_flow',
                         'val_total', 'val_descriptor', 'val_flow', 'learning_rate'])
        for ident, rows in histories.items():
            for r in rows:
                val = r.get('val') or {}
                writer.writerow([ident, r['epoch']+1, r['train']['descriptor']+r['train']['flow'],
                    r['train']['descriptor'], r['train']['flow'], r.get('val_loss'),
                    val.get('descriptor'), val.get('flow'), r['learning_rate']])
    with (out/'all_models_raw_generation.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostics[0]))
        writer.writeheader()
        writer.writerows([{**r, 'properties_requested': json.dumps(r['properties_requested'], ensure_ascii=False)}
                         for r in diagnostics])

    plt.rcParams.update({'font.family':'sans-serif', 'font.sans-serif':['Microsoft YaHei', 'DejaVu Sans'],
        'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False, 'legend.frameon':False})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.7))
    for axis, key, title in zip(axes, ['total', 'descriptor', 'flow'],
            ['a  总损失：描述＋流匹配', 'b  描述 Transformer 损失', 'c  流匹配 OrbitFlow 损失']):
        for (ident, label, _, color), model in zip(RUNS, models):
            rows = histories[ident]
            val = [r for r in rows if r.get('val_loss') is not None]
            train = [r['train']['descriptor']+r['train']['flow'] if key == 'total' else r['train'][key] for r in rows]
            valid = [r['val_loss'] if key == 'total' else r['val'][key] for r in val]
            axis.plot([r['epoch']+1 for r in rows], train, color=color, lw=1.15)
            axis.plot([r['epoch']+1 for r in val], valid, color=color, lw=1.5, ls='--')
            best = next(r for r in val if r['epoch']+1 == model['best_epoch'])
            axis.scatter([model['best_epoch']], [best['val_loss'] if key == 'total' else best['val'][key]],
                         color=color, marker='*', s=65, zorder=5)
        axis.set(title=title, xlabel='完成训练轮次（从1开始）', ylabel='损失（保留原训练定义）', xlim=(0, 325))
        axis.grid(axis='y', color='#e4e9ed', lw=.6)
    handles = [Line2D([], [], color=c, lw=2, label=f'{i} {label}') for i,label,_,c in RUNS]
    handles += [Line2D([], [], color='#64727b', lw=1.4, label='实线：训练'),
                Line2D([], [], color='#64727b', ls='--', lw=1.4, label='虚线：验证'),
                Line2D([], [], color='#64727b', marker='*', ls='none', ms=9, label='星号：该模型联合最优轮次')]
    fig.legend(handles=[handles[i] for i in (0,3,1,4,2,5)], loc='lower center', ncol=3, fontsize=9)
    fig.suptitle('三套多性质CFG模型的完整训练历史', fontsize=14, y=.98)
    fig.subplots_adjust(left=.055, right=.99, top=.86, bottom=.27, wspace=.25)
    fig.savefig(out/'all_models_training_comparison.png', dpi=240, facecolor='white')
    plt.close(fig)

    groups = [('M1','MP20'), ('M2','MPTS52'), ('M3','MP20'), ('M3','MPTS52'), ('M3','C2DB')]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    for axis, key, title in zip(axes, ['exact_target_group','distance_valid','no_coincident_orbits'],
            ['a  原始精确目标群回收', 'b  原始距离初筛 ≥0.6 Å', 'c  无轨道重合']):
        x = np.arange(len(groups))
        for shift, mode, label, color in [(-.18,'unconditional','无性质条件采样','#437da6'),
                                        (.18,'conditional','多性质条件采样','#c57b40')]:
            data = [next(r for r in diagnostics if (r['model'],r['dataset'],r['mode']) == (*g,mode)) for g in groups]
            values = [100*r[key]/r['attempted'] for r in data]
            axis.bar(x+shift, values, .34, color=color, label=label, zorder=3)
            for xx, value, row in zip(x+shift, values, data):
                axis.text(xx, value+1.4, f"{row[key]}/{row['attempted']}", rotation=90, ha='center', va='bottom', fontsize=7.5)
        axis.set_xticks(x, [a+'\n'+b for a,b in groups], fontsize=9)
        axis.set(title=title, ylabel='通过比例（%）', ylim=(0,118))
        axis.set_yticks(np.arange(0,101,20))
        axis.grid(axis='y', color='#e4e9ed', lw=.6, zorder=0)
    axes[0].legend(loc='lower left', fontsize=8)
    fig.suptitle('全部已训练模型的已有生成诊断：M1/M2每模式64次，M3每来源每模式256次', fontsize=12)
    fig.subplots_adjust(left=.055, right=.99, top=.84, bottom=.16, wspace=.25)
    fig.savefig(out/'all_models_generation_diagnostics.png', dpi=240, facecolor='white')
    plt.close(fig)
    save_json(out/'summary.json', {'models':models, 'test':tests, 'raw_generation':diagnostics,
        'input_sha256':inputs, 'source_archive':args.archive.name,
        'scope':'Archived observations only. Different splits, domains, sample budgets and loss scales prevent a fair model ranking. Standard quality is available for M3; M1/M2 retain raw diagnostics only.'})
    print(json.dumps({'models':models, 'training_epochs':sum(len(r) for r in histories.values()),
                      'generation_groups':len(diagnostics)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
