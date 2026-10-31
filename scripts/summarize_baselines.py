"""Build PNG figures and evidence tables from a completed baseline evaluation only."""

from unifiedgroupgen.common.configuration import project_root, resolve_path
import argparse
import json
from pathlib import Path
import sys

ROOT = project_root()
sys.path[:0] = [str(ROOT), str(resolve_path(ROOT/'scripts'))]
from unifiedgroupgen.common.structures import sha256 as digest, read_records, write_json as save
MODELS = {'mp20_u': ['mp_20'], 'c2db51_u': ['c2db_51'], 'joint_u': ['mp_20', 'c2db_51']}
MODES = ('de_novo', 'composition', 'fixed_description', 'fixed_group')

def load(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_attempts(folder):
    return read_records(folder/'samples.jsonl')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--update-manuscript', action='store_true')
    args = parser.parse_args()
    study = (resolve_path(ROOT/args.study)).resolve()
    relocation = load(study) if study.is_file() else None
    if relocation:
        if 'study_identity' in relocation:
            study_identity = relocation['study_identity']
        else:
            from zipfile import ZipFile
            with ZipFile(study.parent/'previous_control_records.zip') as archive:
                member = next(n for n in archive.namelist() if n.startswith('baselines_256_local_seed142/') and n.endswith('/pipeline.json'))
                study_identity = json.loads(archive.read(member))['identity']
        path_map = relocation['path_map']
        def relocated(suffix):
            return ROOT/next(value for key,value in path_map.items() if key.startswith('outputs/baselines_256_local_seed142/') and key.endswith(suffix))
        study = ROOT
    else:
        study_identity = load(study/'pipeline.json')['identity']
    output = (resolve_path(ROOT/args.output)).resolve() if args.output else ROOT/'assets/results/baseline256_report'
    if not output.is_relative_to(ROOT):
        parser.error('Keep outputs inside the project')
    # Import the existing PyTorch runtime before NumPy/OpenMP-dependent plotting libraries.
    import torch
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    from unifiedgroupgen.common.rendering import render_structure
    output.mkdir(parents=True, exist_ok=True)
    tests, tasks, groups, references, strata = [], [], [], [], []
    figures, sources = [], {}
    frames = {}
    for model, datasets in MODELS.items():
        test_path = (relocated('/'+model+'/test') if relocation else study/model/'test')/'summary.json'
        test = load(test_path)
        if test['state'] != 'complete':
            raise ValueError('Module tests are incomplete')
        if test['records_sha256'] != digest(test_path.parent/'records.jsonl'):
            raise ValueError('Module test rows changed')
        sources[str(test_path.relative_to(study))] = digest(test_path)
        for dataset, values in test['sources'].items():
            tests.append({'model': model, 'dataset': dataset, **{k: v for k, v in values.items() if k != 'tokens'},
                          **{f'{token}_{key}': stat for token, part in values['tokens'].items() for key, stat in part.items()}})
        for dataset in datasets:
            for mode in MODES:
                folder = relocated('/'+model+'/'+dataset+'/'+mode) if relocation else study/model/dataset/mode
                metric_folder = folder.parent.parent/'evaluate'/folder.name if relocation else folder/'metrics'
                sample_summary = load(folder/'summary.json')
                report_path = metric_folder/'summary.json'
                report = load(report_path)
                if sample_summary['state'] != 'complete' or report['state'] != 'complete':
                    raise ValueError('Generation or standard metrics are incomplete')
                receipt = load(metric_folder/'storage_migration.json') if relocation else None
                if (report['samples_sha256'] != (receipt['source_sha256'] if receipt else digest(folder/'samples.jsonl'))
                        or receipt and receipt['samples_sha256'] != digest(folder/'samples.jsonl')):
                    raise ValueError('Samples differ from the evaluated ledger')
                read_attempts(folder)
                if report['records_csv_sha256'] != digest(metric_folder/'records.csv'):
                    raise ValueError('Metric rows changed')
                sources[str(report_path.relative_to(study))] = digest(report_path)
                row = {'model': model, 'dataset': dataset, 'task': mode, 'attempted': report['attempted'],
                       'generated': report['generated'],
                       **{key: report[key]['rate_over_all_attempts'] for key in
                          ('legacy_structural_valid', 'periodic_distance_valid', 'composition_valid',
                           'joint_valid', 'exact_target_group', 'unique', 'novel_against_reference')},
                       'vun_rate': report['valid_unique_novel_rate_over_all_attempts'],
                       **{key: report['coverage'].get(key) for key in ('cov_precision', 'cov_recall')},
                       **{key: sample_summary.get(key) for key in
                          ('match_at_1', 'composition_satisfaction', 'matched_rmsd_mean_angstrom',
                           'matched_rmsd_evaluated', 'zero_coordinate_dof_cases')}}
                tasks.append(row)
                for name, values in sample_summary.get('coordinate_dof_strata', {}).items():
                    strata.append({'model': model, 'dataset': dataset, 'task': mode, 'coordinate_dof': name, **values})
                frames[model, dataset, mode] = pd.read_csv(metric_folder/'records.csv')
                if mode == 'fixed_group':
                    if report['groups_csv_sha256'] != digest(metric_folder/'groups.csv'):
                        raise ValueError('Per-group results changed')
                    group_frame = pd.read_csv(metric_folder/'groups.csv')
                    group_frame.insert(0, 'dataset', dataset); group_frame.insert(0, 'model', model)
                    groups.extend(group_frame.to_dict('records'))
                    group_frame.to_csv(output/f'{model}_{dataset}_groups.csv', index=False)
            if relocation and relocation.get('reference_statistics'):
                ref = pd.read_csv(ROOT/relocation['reference_statistics'])
                references += ref[(ref.model == model) & (ref.dataset == dataset)].to_dict('records')
            else:
                ref = load((relocated('/reference_cache') if relocation else study/'reference_cache')/model/f'{dataset}_reference.json')
                references += [{'model': model, 'dataset': dataset, **row} for row in ref['rows']]
    test_frame, task_frame, group_frame = pd.DataFrame(tests), pd.DataFrame(tasks), pd.DataFrame(groups)
    test_frame.to_csv(output/'component_tests.csv', index=False)
    task_frame.to_csv(output/'generation_quality.csv', index=False)
    group_frame.to_csv(output/'group_coverage.csv', index=False)
    pd.DataFrame(strata).to_csv(output/'reconstruction_by_coordinate_dof.csv', index=False)
    ref_frame = pd.DataFrame(references)
    ref_frame.to_csv(output/'test_reference_statistics.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'legend.frameon': False})
    colors = ['#437da6', '#c57b40', '#508c76', '#865f9a']
    def save_figure(fig, name):
        fig.savefig(output/name, dpi=240, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        figures.append(name)
    labels = [f'{r.model}\n{r.dataset}' for r in test_frame.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout='constrained')
    for i, row in enumerate(test_frame.itertuples()):
        axes[0].plot(range(4), [getattr(row, k+'_nll') for k in ('G', 'W', 'Z', 'STOP')],
                     'o-', color=colors[i], label=labels[i].replace('\n', ' / '))
    axes[0].set_xticks(range(4), ['Group', 'Orbit', 'Element', 'STOP'])
    axes[0].set_ylabel('True-prefix token NLL'); axes[0].legend(fontsize=7)
    x = np.arange(len(test_frame))
    axes[1].bar(x-.18, test_frame.lattice, .36, label='Lattice')
    axes[1].bar(x+.18, test_frame.coordinates_dimensionless, .36, label='Coordinates / 9')
    axes[1].set_xticks(x, labels, fontsize=8); axes[1].set_ylabel('Held-out flow loss'); axes[1].legend()
    save_figure(fig, '01_component_tests.png')
    recon = task_frame[task_frame.task.isin(['composition', 'fixed_description'])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4), layout='constrained')
    for i, mode in enumerate(('composition', 'fixed_description')):
        part = recon[recon.task == mode]
        axes[0].bar(np.arange(4)+(i-.5)*.36, part.match_at_1, .36, label=mode)
        axes[1].bar(np.arange(4)+(i-.5)*.36, part.matched_rmsd_mean_angstrom, .36, label=mode)
    for ax in axes:
        ax.set_xticks(range(4), labels, fontsize=8); ax.legend(fontsize=8)
    axes[0].set_ylim(0, 1); axes[0].set_ylabel('Match@1 / all test cases')
    axes[1].set_ylabel('Mean RMSD of aligned matches (angstrom)')
    axes[1].set_title('Missing bars = no measured matched RMSD')
    save_figure(fig, '02_reconstruction.png')
    de_novo = task_frame[task_frame.task == 'de_novo']
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), layout='constrained')
    rate_keys = ['legacy_structural_valid', 'composition_valid', 'joint_valid', 'exact_target_group']
    for i, row in enumerate(de_novo.to_dict('records')):
        axes[0].plot(range(4), [row[k] for k in rate_keys], 'o-', color=colors[i], label=f"{row['model']}/{row['dataset']} (n={row['attempted']})")
        axes[1].plot(range(3), [row[k] for k in ('vun_rate', 'cov_precision', 'cov_recall')], 'o-', color=colors[i])
    axes[0].set_xticks(range(4), ['Geometry', 'Chemistry', 'Joint valid', 'Exact group'])
    axes[1].set_xticks(range(3), ['VUN / all', 'COV-P', 'COV-R'])
    for ax in axes:
        ax.set_ylim(-.03, 1.03); ax.set_ylabel('Rate')
    axes[0].legend(fontsize=7)
    save_figure(fig, '03_de_novo_quality.png')
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), layout='constrained')
    for ax, ((model, dataset), part) in zip(axes, group_frame.groupby(['model', 'dataset'], sort=False)):
        ax.plot(part.group, part.target_group_containment, label='Target operations satisfied', color='#437da6', alpha=.75)
        ax.plot(part.group, part.exact_target_group, label='Exact group', color='#865f9a', alpha=.85)
        ax.plot(part.group, part.joint_valid, label='Joint valid', color='#c57b40', alpha=.85)
        unseen = part[part.seen.eq(False)]
        ax.scatter(unseen.group, np.full(len(unseen), -.08), marker='|', color='black', label='Unseen in training')
        ax.set_ylim(-.14, 1.08); ax.set_ylabel('Passes / attempts'); ax.set_title(model+' / '+dataset, loc='left')
        ax.set_xlabel('Layer group' if dataset == 'c2db_51' else 'Space group')
    axes[0].legend(ncol=4, fontsize=8)
    save_figure(fig, '04_all_groups.png')
    keys = [('atoms', 'Atoms per cell'), ('num_elements', 'Element count'),
            ('volume_per_atom_angstrom3', 'Volume / atom (angstrom^3)'),
            ('area_per_atom_angstrom2', 'Area / atom (angstrom^2)'), ('layer_thickness_angstrom', 'Layer thickness (angstrom)')]
    fig, axes = plt.subplots(4, 5, figsize=(17, 11), layout='constrained')
    for row_index, (model, dataset) in enumerate((m, d) for m, ds in MODELS.items() for d in ds):
        generated = frames[model, dataset, 'de_novo']
        generated = generated[generated.joint_valid.eq(True)]
        reference = ref_frame[(ref_frame.model == model) & (ref_frame.dataset == dataset)]
        for ax, (key, label) in zip(axes[row_index], keys):
            for frame, color, title in ((reference, '#999999', 'Test'), (generated, colors[row_index], 'Generated valid')):
                if key in frame:
                    values = np.sort(frame[key].dropna().values)
                    if len(values):
                        ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=title)
            ax.set_title(label, fontsize=8); ax.set_ylabel(f'{model}\n{dataset}\nECDF', fontsize=7)
    axes[0, 0].legend(fontsize=7)
    save_figure(fig, '05_generated_distributions.png')
    fig, axes = plt.subplots(4, 4, figsize=(14, 13), layout='constrained')
    gallery = []
    for row_index, (model, dataset) in enumerate((m, d) for m, ds in MODELS.items() for d in ds):
        folder = relocated('/'+model+'/'+dataset+'/de_novo') if relocation else study/model/dataset/'de_novo'
        samples = [json.loads(s) for s in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
        selected = [s for s in samples if s['status'] == 'generated'][:4]
        for ax in axes[row_index]:
            ax.set_axis_off()
        for ax, sample in zip(axes[row_index], selected):
            compiled = compile_descriptor(Descriptor.from_dict(sample['descriptor']))
            value = compiled.expand(torch.tensor(sample['state'], dtype=torch.float64))
            lattice = value['lattice'].numpy().T.copy(); coords = value['coordinates'].numpy().copy()
            if dataset == 'c2db_51':
                coords[:, 2] += 1-coords[:, 2].min()
            image = render_structure(lattice, coords, compiled.elements, periodic=tuple(compiled.periodic),
                                     repeat=(2, 2, 1) if dataset == 'c2db_51' else (1, 1, 1), size=(640, 560))
            from pymatgen.core import Composition, Element
            from collections import Counter
            formula = Composition({Element.from_Z(int(z)).symbol: n for z, n in Counter(compiled.elements).items()}).reduced_formula
            ax.imshow(image); ax.set_title(f'{formula}\n{model} / {dataset} / attempt {sample["index"]}', fontsize=8)
            gallery.append({'model': model, 'dataset': dataset, 'index': sample['index'], 'formula': formula,
                            'cif_sha256': sample['cif_sha256']})
    fig.suptitle('First four successful attempts per domain; no manual selection or relaxation')
    save_figure(fig, '06_generated_structures.png')
    pd.DataFrame(gallery).to_csv(output/'gallery_selection.csv', index=False)
    text = ['### 三个修复数据基础模型的独立生成评估', '',
            '使用各自总验证损失最优检查点；不拼接独立最优模块。无目标性质，guidance=0，联合模型保留来源与类型上下文。', '',
            '| 模型 | 来源 | 任务 | 尝试 | 生成 | 联合有效率 | 精确群率 | Match@1 |',
            '|---|---|---|---:|---:|---:|---:|---:|']
    def number(value):
        return '未计算' if value is None or pd.isna(value) else f'{value:.4f}'
    for row in tasks:
        text.append(f"| {row['model']} | {row['dataset']} | {row['task']} | {row['attempted']} | {row['generated']} | {number(row['joint_valid'])} | {number(row['exact_target_group'])} | {number(row['match_at_1'])} |")
    text += ['', '固定描述只验证连续生成；指定组成CSP验证两个模块配合。RMSD仅对可对齐的匹配结果统计，不能用其替代全部案例成功率。',
             '所有失败尝试保留分母；每群少量样本用于覆盖诊断，不能据此精确排名各群。COV需要相同样本量与参考集才能公平比较。',
             '联合有效率是几何、化学筛查和指纹可计算性的组合；这些结果不证明弛豫后稳定性或性质控制。', '']
    captions = ['测试集上群、轨道、元素、终止符预测与连续速度误差，定位两个模块的泛化表现。',
                '指定组成与固定真实描述的重建比较，用于区分规则选择和连续生成误差。',
                '从头生成的有效性、精确群、VUN与分布覆盖；所有失败保留在分母中。',
                '全空间群与层群覆盖，区分操作保持、精确群和材料有效性；底部标出未见群。',
                '生成有效结构与真实测试集的规模、体积/面积和层厚分布；空面板表示该维度不适用。',
                '每个模型/数据类型前四次成功生成的结构总图，未做人工挑选或弛豫。']
    for figure, caption in zip(figures, captions):
        text += [f'![{caption}]({figure})', '', caption, '']
    (output/'RESULTS_ZH.md').write_text('\n'.join(text), encoding='utf-8')
    save(output/'manifest.json', {'state': 'complete', 'study_identity': study_identity,
        'source_reports_sha256': sources, 'figures': {name: digest(output/name) for name in figures},
        'tables': {p.name: digest(p) for p in output.glob('*.csv')}, 'export': 'PNG only',
        'scope': 'Module tests, reconstruction, unconditioned-on-properties generation and group coverage; no physical stability claim.'})
    if args.update_manuscript:
        article = resolve_path(ROOT/'docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md')
        original = article.read_text(encoding='utf-8')
        backup = output/'manuscript_before_update.md'
        if not backup.exists():
            backup.write_text(original, encoding='utf-8')
        start, end = '<!-- BASELINE256_RESULTS_START -->', '<!-- BASELINE256_RESULTS_END -->'
        import os
        section = '\n'.join(text)
        for name in figures:
            link = os.path.relpath(output/name, article.parent).replace('\\', '/')
            section = section.replace(']('+name+')', ']('+link+')')
        section = start+'\n\n'+section+'\n'+end+'\n\n'
        if start in original:
            left, remainder = original.split(start, 1)
            _, right = remainder.split(end, 1)
            updated = left+section+right.lstrip('\n')
        else:
            marker = '## 10 Experiments'
            if marker not in original:
                raise ValueError('Manuscript section anchor changed; refusing blind append')
            updated = original.replace(marker, section+marker, 1)
        article.write_text(updated, encoding='utf-8')
    print(output)


if __name__ == '__main__':
    main()
