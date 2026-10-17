"""Collect real metric outputs into paper tables and PNG panels; missing evidence stays missing."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.training import file_hash

RATE_KEYS = ('legacy_structural_valid', 'composition_valid', 'joint_valid', 'exact_target_group')
RATE_LABELS = ('Geometry', 'SMACT', 'Joint validity', 'Exact group')
DATASETS = ('mp_20', 'mpts_52', 'c2db_51')
PROPERTIES = ('formation_energy', 'band_gap', 'energy_above_hull')
COLORS = ('#437da6', '#d68939', '#4d9b87', '#b36a88', '#8276ad')


def wilson(passed, total):
    if not total:
        return None, None
    z = 1.959963984540054
    p = passed/total
    center = (p+z*z/(2*total))/(1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total+z*z/(4*total*total))/(1+z*z/total)
    return max(0, float(center-half)), min(1, float(center+half))


def collect(manifest):
    tables, samples, provenance = [], [], []
    seen = set()
    for experiment in manifest['experiments']:
        path = root_path(experiment['metrics'])
        summary = json.loads(path.read_text(encoding='utf-8'))
        if summary.get('state', 'complete') != 'complete' or not summary.get('jobs'):
            raise ValueError('Only completed nonempty standard-metric summaries may be collected')
        selected = experiment.get('jobs', list(summary['jobs']))
        for name in selected:
            report = summary['jobs'][name]
            # Same samples under different labels would create a false replication.
            fingerprint = report['samples_sha256']
            if fingerprint in seen:
                raise ValueError('Duplicate sample file across paper experiments')
            seen.add(fingerprint)
            dataset = report.get('dataset') or next(d for d in DATASETS if name.startswith(d+'_'))
            sampling = report.get('sampling', {})
            mode = ('prior' if sampling.get('prior') else 'fixed_lattice' if 'fixedlattice' in name
                    else 'fixed_descriptor' if sampling.get('fixed_descriptor') else
                    'conditional' if report.get('requested_properties') else 'unconditional')
            row = {'model': experiment['label'], 'job': name, 'dataset': dataset, 'mode': mode,
                   'model_seed': experiment.get('model_seed'), 'sampling_seed': sampling.get('seed'),
                   'checkpoint_sha256': summary.get('checkpoint_sha256'),
                   'case': name.split('_case')[1].split('_')[0] if '_case' in name else None,
                   'reference_identity': summary['reference_identity'], 'attempted': report['attempted'],
                   'generated': report['generated'], 'guidance': sampling.get('guidance'),
                   'steps': sampling.get('steps'), 'seconds_per_attempt': sampling.get('seconds_per_attempt'),
                   'cov_precision': report['coverage'].get('cov_precision'),
                   'cov_recall': report['coverage'].get('cov_recall'),
                   'coverage_status': report['coverage'].get('status'),
                   'reference_fingerprints': report['coverage'].get('reference_fingerprints'),
                   'reference_requested': report['coverage'].get('reference_requested'),
                   'valid_unique_novel_rate': report['valid_unique_novel_rate_over_all_attempts'],
                   'stable_rate': report['external'].get('stable_rate_over_all_attempts'),
                   'sun_rate': report['external'].get('sun_rate_over_all_attempts')}
            for key in RATE_KEYS:
                metric = report[key]
                row[key] = metric['rate_over_all_attempts']
                row[key+'_passed'] = metric['passed']
                row[key+'_unknown'] = metric['unknown']
                row[key+'_ci_low'], row[key+'_ci_high'] = wilson(metric['passed'], report['attempted'])
            for prop in PROPERTIES:
                row['target_'+prop] = report.get('requested_properties', {}).get(prop)
                assessment = report['external'].get('properties', {}).get(prop, {})
                row[prop+'_valid_hit_rate'] = assessment.get('valid_hit_rate_over_all_attempts')
                row[prop+'_evaluated'] = assessment.get('evaluated')
                row[prop+'_mae'] = assessment.get('mae')
                row[prop+'_wasserstein'] = report.get('property_distributions', {}).get(prop, {}).get('wasserstein')
            for quantity, statistic in report['distributions'].items():
                row[quantity+'_wasserstein'] = statistic.get('wasserstein')
            tables.append(row)
            sample_path = path.parent/f'{name}.csv'
            if report.get('metrics_csv_sha256') and file_hash(sample_path) != report['metrics_csv_sha256']:
                raise ValueError('Per-attempt metrics file changed after evaluation')
            frame = pd.read_csv(sample_path)
            if len(frame) != report['attempted'] or frame['index'].duplicated().any():
                raise ValueError('Per-attempt table does not match summary accounting')
            for key in ('model', 'dataset', 'mode', 'job', 'model_seed', 'sampling_seed', 'guidance'):
                frame[key] = row[key]
            samples.append(frame)
        provenance.append({'metrics': str(path.relative_to(ROOT)), 'sha256': file_hash(path),
            'label': experiment['label'], 'model_seed': experiment.get('model_seed'),
            'checkpoint_sha256': summary.get('checkpoint_sha256'),
            'train_sha256': summary['train_sha256'], 'test_sha256': summary['test_sha256'],
            'evaluation_code_sha256': summary['evaluation_code_sha256'], 'versions': summary['versions'],
            'assessment_provenance': summary.get('assessment_provenance')})
    return pd.DataFrame(tables), pd.concat(samples, ignore_index=True), provenance


def grouped_rates(samples, by):
    rows = []
    for identity, frame in samples.groupby(by, dropna=False, observed=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        row = dict(zip(by, identity))
        row['attempted'] = len(frame)
        for key in RATE_KEYS:
            values = frame[key] if key in frame else pd.Series(index=frame.index, dtype=object)
            passed = int(values.eq(True).sum())
            row[key] = passed/len(frame)
            row[key+'_passed'] = passed
            row[key+'_unknown'] = int(values.isna().sum())
            row[key+'_ci_low'], row[key+'_ci_high'] = wilson(passed, len(frame))
        rows.append(row)
    return pd.DataFrame(rows)


def render(table, samples, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    created = []
    def save(fig, name):
        fig.savefig(output/name, dpi=220, bbox_inches='tight')
        plt.close(fig)
        created.append(name)
    uncond = samples[samples['mode'] == 'unconditional']
    if len(uncond):
        aggregated = grouped_rates(uncond, ['model', 'dataset'])
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), constrained_layout=True)
        for ax, dataset in zip(axes, DATASETS):
            selected = aggregated[aggregated.dataset == dataset]
            for i, (_, row) in enumerate(selected.iterrows()):
                y = np.arange(len(RATE_KEYS))+(i-(len(selected)-1)/2)*.13
                values = np.array([row[key] for key in RATE_KEYS])
                for k, key in enumerate(RATE_KEYS):
                    if row[key+'_unknown'] == row['attempted']:
                        values[k] = np.nan
                err = np.array([[row[key]-row[key+'_ci_low'] for key in RATE_KEYS],
                                [row[key+'_ci_high']-row[key] for key in RATE_KEYS]])
                ax.errorbar(values, y, xerr=np.maximum(err, 0), fmt='o', capsize=2,
                            label=f"{row['model']} (n={row['attempted']})", color=COLORS[i % len(COLORS)])
            ax.set_yticks(range(len(RATE_KEYS)), RATE_LABELS); ax.set_xlim(-.02, 1.03)
            ax.set_title(dataset); ax.set_xlabel('Passes / all attempts (95% Wilson interval)')
            if len(selected): ax.legend(fontsize=7, loc='lower left')
        save(fig, '03_shared_generation_validity.png')
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        selected_jobs = table[table['mode'] == 'unconditional']
        keys = ['cov_precision', 'cov_recall', 'valid_unique_novel_rate']
        for ax, dataset in zip(axes, DATASETS):
            frame = selected_jobs[selected_jobs.dataset == dataset]
            for i, (model, rows) in enumerate(frame.groupby('model')):
                # Display every sampling replicate; do not manufacture a training-replicate interval.
                for _, row in rows.iterrows():
                    ax.scatter(np.arange(3)+(i-.5)*.1, [row[k] for k in keys], s=25,
                               color=COLORS[i % len(COLORS)], label=model)
                    if (row.get('coverage_status') != 'complete' and pd.notna(row.get('reference_fingerprints'))
                            and pd.notna(row.get('reference_requested'))):
                        ax.annotate(f'Reference fingerprints: {int(row.reference_fingerprints)}/{int(row.reference_requested)}',
                                    (1+(i-.5)*.1, row.cov_recall), xytext=(0, 10), textcoords='offset points',
                                    ha='center', fontsize=7)
            ax.set_xticks(range(3), ['COV-P', 'COV-R', 'Valid unique novel'], rotation=12)
            ax.set_ylim(-.02, 1.04); ax.set_ylabel('Rate (each point = one sampling job)'); ax.set_title(dataset)
            handles, labels = ax.get_legend_handles_labels()
            distinct = dict(zip(labels, handles))
            if distinct: ax.legend(distinct.values(), distinct.keys(), fontsize=7)
        save(fig, '03_shared_generation_coverage.png')
        strata = uncond.copy()
        strata['dof_bin'] = pd.cut(strata.coordinate_dof, [-1, 0, 3, 9, np.inf],
                                   labels=['0', '1–3', '4–9', '10+'])
        by_dof = grouped_rates(strata, ['model', 'dataset', 'dof_bin'])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        for ax, dataset in zip(axes, DATASETS):
            for i, (model, frame) in enumerate(by_dof[by_dof.dataset == dataset].groupby('model')):
                frame = frame.set_index('dof_bin').reindex(['0', '1–3', '4–9', '10+'])
                ax.plot(range(4), frame.joint_valid, 'o-', label=model, color=COLORS[i % len(COLORS)])
                for x, (_, row) in enumerate(frame.iterrows()):
                    if pd.notna(row.attempted):
                        ax.annotate(f"n={int(row.attempted)}", (x, row.joint_valid), xytext=(0, 6),
                                    textcoords='offset points', ha='center', fontsize=6)
            ax.set_xticks(range(4), ['0', '1–3', '4–9', '10+']); ax.set_ylim(-.02, 1.12)
            ax.set_xlabel('Coordinate degrees of freedom'); ax.set_ylabel('Joint validity / all attempts')
            ax.set_title(dataset)
            if len(by_dof[by_dof.dataset == dataset]): ax.legend(fontsize=7)
        save(fig, '03_shared_generation_by_dof.png')
    controls = samples[samples['mode'].isin(['prior', 'fixed_descriptor', 'fixed_lattice'])]
    if len(controls):
        rates = grouped_rates(controls, ['model', 'dataset', 'mode'])
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), constrained_layout=True)
        modes = ['prior', 'fixed_descriptor', 'fixed_lattice']
        for ax, dataset in zip(axes, DATASETS):
            for model, frame in rates[rates.dataset == dataset].groupby('model'):
                frame = frame.set_index('mode').reindex(modes)
                ax.plot(range(3), frame.joint_valid, 'o-', label=model)
            ax.set_xticks(range(3), ['Legal prior', 'Learned flow', 'Fixed lattice'], rotation=15)
            ax.set_ylim(-.02, 1.08); ax.set_ylabel('Joint validity / all attempts'); ax.set_title(dataset)
            if len(rates[rates.dataset == dataset]): ax.legend(fontsize=7)
        save(fig, '04_continuous_controls.png')
        if 'observation_max_residual' in controls:
            observed = controls[controls.observation_max_residual.notna()]
            if len(observed):
                fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
                for dataset, group in observed.groupby('dataset'):
                    values = np.sort(np.maximum(group.observation_max_residual.to_numpy(), 1e-15))
                    ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                            label=f'{dataset}, n={len(values)}, max={group.observation_max_residual.max():.2g}')
                ax.set_xscale('log'); ax.set_xlabel('Max |Az − b| (zeros plotted at 1e−15)')
                ax.set_ylabel('Cumulative fraction'); ax.legend(fontsize=7)
                save(fig, '04_observation_residuals.png')
    predicted = [p for p in PROPERTIES if f'predicted_{p}' in samples and samples[f'predicted_{p}'].notna().any()]
    if predicted:
        fig, axes = plt.subplots(len(predicted), 3, figsize=(12, 3.5*len(predicted)),
                                  squeeze=False, constrained_layout=True)
        for axes_row, prop in zip(axes, predicted):
            for ax, dataset in zip(axes_row, DATASETS):
                selected = samples[(samples.dataset == dataset) & (samples['mode'] == 'conditional')
                    & samples.joint_valid.eq(True)]
                for i, ((model, guidance), frame) in enumerate(selected.groupby(['model', 'guidance'])):
                    values = frame.groupby('target_'+prop)['predicted_'+prop].agg(
                        median='median', low=lambda x: x.quantile(.25), high=lambda x: x.quantile(.75)).dropna()
                    if len(values):
                        ax.errorbar(values.index.astype(float), values['median'],
                            yerr=np.array([values['median']-values.low, values.high-values['median']]),
                            fmt='o-', label=f'{model}, CFG={guidance}', color=COLORS[i % len(COLORS)], capsize=2)
                if ax.lines:
                    low, high = ax.get_xlim(); ax.plot([low, high], [low, high], ':', color='#777777', lw=1)
                    ax.legend(fontsize=6)
                unit = 'eV' if prop == 'band_gap' else 'eV/atom'
                ax.set_xlabel(f'Requested {prop} ({unit})'); ax.set_ylabel(f'Independent value ({unit}): median / IQR')
                ax.set_title(dataset)
        save(fig, '05_condition_response.png')
        fig, axes = plt.subplots(1, len(predicted), figsize=(5*len(predicted), 3.6), squeeze=False,
                                 constrained_layout=True)
        for ax, prop in zip(axes[0], predicted):
            values = table[table[prop+'_valid_hit_rate'].notna()]
            for (model, dataset), frame in values.groupby(['model', 'dataset']):
                ax.scatter(frame.joint_valid, frame[prop+'_valid_hit_rate'], label=f'{model}: {dataset}', s=20)
            ax.set_xlim(-.02, 1.02); ax.set_ylim(-.02, 1.02); ax.set_title(prop)
            ax.set_xlabel('Joint validity / all attempts'); ax.set_ylabel('Valid property hits / all attempts')
            if len(values): ax.legend(fontsize=6)
        save(fig, '05_condition_valid_yield.png')
    physical = table[table.stable_rate.notna() | table.sun_rate.notna()]
    if len(physical):
        fig, ax = plt.subplots(figsize=(max(7, .7*len(physical)), 3.8), constrained_layout=True)
        x = np.arange(len(physical))
        ax.bar(x-.18, pd.to_numeric(physical.stable_rate, errors='coerce'), .35, label='Relaxed hull screen')
        ax.bar(x+.18, pd.to_numeric(physical.sun_rate, errors='coerce'), .35, label='Relaxed stable + unique + novel')
        ax.set_xticks(x, [f'{r.model}\n{r.job}' for r in physical.itertuples()], rotation=35, ha='right', fontsize=7)
        ax.set_ylabel('Success / all attempts'); ax.set_ylim(0, 1.03); ax.legend()
        save(fig, '06_physical_screening.png')
    if 'relaxed_recovered_group' in samples and 'recovered_group' in samples:
        known = samples[samples.relaxed_recovered_group.notna() & samples.recovered_group.notna()]
        if len(known):
            fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
            for i, (dataset, frame) in enumerate(known.groupby('dataset')):
                count = int((frame.recovered_group == frame.relaxed_recovered_group).sum())
                ax.bar(i, count/len(frame), color=COLORS[i % len(COLORS)])
                ax.text(i, count/len(frame)+.02, f'{count}/{len(frame)}', ha='center')
            ax.set_xticks(range(known.dataset.nunique()), list(known.groupby('dataset').groups))
            ax.set_ylim(0, 1.12); ax.set_ylabel('Same recovered group before/after relaxation')
            save(fig, '06_relaxation_symmetry.png')
    return created


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='JSON experiments: label, metrics (summary.json), optional model_seed/jobs')
    parser.add_argument('--output', default='assets/paper')
    args = parser.parse_args()
    manifest_path = root_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    table, samples, provenance = collect(manifest)
    table.to_csv(output/'main_results.csv', index=False)
    samples.to_csv(output/'attempts.csv.gz', index=False, compression='gzip')
    strata = samples.copy()
    strata['coordinate_dof_bin'] = pd.cut(strata.coordinate_dof, [-1, 0, 3, 9, np.inf], labels=['0', '1-3', '4-9', '10+'])
    grouped_rates(strata, ['model', 'dataset', 'mode', 'coordinate_dof_bin']).to_csv(output/'dof_results.csv', index=False)
    grouped_rates(samples, ['model', 'dataset', 'mode', 'target_group']).to_csv(output/'group_results.csv', index=False)
    # Do not treat sampling seeds as separately trained models.
    numeric = ['joint_valid', 'exact_target_group', 'valid_unique_novel_rate', 'cov_precision', 'cov_recall']
    repeat_keys = ['model', 'dataset', 'mode', 'guidance', 'steps', 'case', *['target_'+p for p in PROPERTIES]]
    table.groupby(repeat_keys, dropna=False)[numeric].agg(['mean', 'std', 'count']).to_csv(output/'repeat_summary.csv')
    created = render(table, samples, output)
    report = {'manifest_sha256': file_hash(manifest_path), 'inputs': provenance, 'png_files': created,
              'attempts': len(samples), 'metric_jobs': len(table),
              'pending': [name for name, present in (
                  ('conditional property assessment', '05_condition_response.png' in created),
                  ('physical relaxation/hull assessment', '06_physical_screening.png' in created),
                  ('matched continuous controls', '04_continuous_controls.png' in created)) if not present],
              'scope': 'Wilson intervals are descriptive per-attempt binomial intervals; correlated same-descriptor '
                       'samples are not independent materials or model replicates. Repeat means/std summarize '
                       'listed jobs; compare equal budgets and distinguish sampling from training seeds. '
                       'Pooling is only within source/mode; no 2D/3D density pooling. '
                       'Raw provenance is retained for comparison review; collecting files does not establish '
                       'matched splits or fair external baseline settings.'}
    (output/'manifest.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'figures': created, 'pending': report['pending']}))


if __name__ == '__main__':
    main()
