"""Paper figures from audited inputs, with explicit placeholders for unavailable results."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy
from unifiedgroupgen.symmetry import Descriptor
from unifiedgroupgen.training import file_hash

OUT = ROOT/'assets/paper'
DATASETS = [('mp_20', 'MP20', '#437da6'), ('mpts_52', 'MPTS52', '#d68939'), ('c2db_51', 'C2DB', '#4d9b87')]


def save(fig, name):
    fig.savefig(OUT/name, dpi=260, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def examples():
    from unifiedgroupgen.structure_rendering import render_structure, element_color
    records = json.loads((Path(__file__).parent/'material_examples.json').read_text(encoding='utf-8'))['examples']
    fig = plt.figure(figsize=(12, 6.5), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[4.4, 1.8])
    for column, record in enumerate(records):
        c = compiled_chart(Descriptor.from_dict(record['descriptor']))
        lattice, coords, _ = expand_numpy(c, record['state'])
        axis = fig.add_subplot(grid[0, column])
        basis = c.coordinate_basis.reshape(c.num_atoms, 3, c.coordinate_dof)
        motion = np.einsum('ijq,q->ij', basis, [.07 if column == 0 else .6]) @ lattice.T
        image = render_structure(lattice.T, coords, c.elements, periodic=tuple(c.periodic),
            repeat=(1, 1, 1) if column == 0 else (2, 2, 1), size=(1350, 850), vectors=4*motion, vector_color='#167d8d')
        axis.imshow(image); axis.set_axis_off()
        symbols = [('Ti', 22), ('O', 8)] if column == 0 else [('Mo', 42), ('S', 16)]
        for symbol, number in symbols:
            axis.scatter([], [], color=np.array(element_color(number))/255, s=65, label=symbol)
        axis.legend(loc='upper right', frameon=False, fontsize=10)
        if column == 0:
            axis.set_title('(a) Rutile TiO$_2$ · MP20 mp-2657', loc='left', fontsize=13)
            text = ('SG 136 · Ti 2a + O 4f · (d, p) = (3, 3)\n'
                    r'$z=(k_1,k_2,u)$: 2 lattice + 1 coordinate DOF'+'\n'
                    r'$a=b=4.6533\ \AA,\ c=2.9692\ \AA,\ u=0.30458$'+'\n'
                    '6 atoms per cell; closed-face periodic images shown')
        else:
            axis.set_title('(b) Monolayer MoS$_2$ · C2DB 1MoS2-1', loc='left', fontsize=13)
            text = ('LG 78 · Mo 1a + S 2f · (d, p) = (3, 2)\n'
                    r'$z=(k,h)$: 1 in-plane lattice + 1 height DOF'+'\n'
                    r'$a=b=3.1841\ \AA,\ \gamma=120^\circ,\ h=1.56356\ \AA$'+'\n'
                    '3 atoms per cell (2 × 2 display); no normal repetition')
        note = fig.add_subplot(grid[1, column]); note.set_axis_off()
        note.text(0, .98, text, ha='left', va='top', fontsize=11, linespacing=1.65, transform=note.transAxes)
    fig.suptitle('Real encoded examples · shared expansion and velocity interfaces', fontsize=14)
    save(fig, 'fig01_material_examples.png')


def representation():
    frame = pd.read_csv(ROOT/'assets/representation/records.csv.gz')
    catalog = pd.read_csv(ROOT/'assets/representation/catalog.csv')
    source = pd.read_csv(ROOT/'assets/representation/source_reencoding.csv')
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.1), layout='constrained')
    ax = axes[0, 0]
    for i, (dataset, label, color) in enumerate(DATASETS):
        selected = frame[frame.dataset == dataset]
        passed = int(selected.passed.sum())
        ax.bar(i, len(selected)/1000, color=color, width=.62)
        ax.text(i, len(selected)/1000+.8, f'{passed:,}/{len(selected):,}', ha='center', fontsize=10)
    ax.set_xticks(range(3), [x[1] for x in DATASETS]); ax.set_ylabel('Prepared records (thousands)')
    ax.set_ylim(0, 52); ax.set_title('(a) All-record algebraic checks', loc='left')
    ax = axes[0, 1]
    for dataset, label, color in DATASETS:
        values = np.sort(frame.loc[frame.dataset == dataset, 'state_dof'])
        ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=label)
    ax.set_xlabel('Independent state dimension: lattice + coordinates'); ax.set_ylabel('Cumulative fraction')
    ax.set_xscale('log'); ax.set_ylim(0, 1.03); ax.legend(frameon=False)
    ax.set_title('(b) Variable-dimensional legal states', loc='left')
    ax = axes[1, 0]
    for dataset, label, color in DATASETS:
        values = np.sort(np.maximum(frame.loc[frame.dataset == dataset, 'inverse_state_max_error'], 1e-16))
        ax.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=label)
    ax.set_xscale('log'); ax.set_ylim(0, 1.03); ax.set_xlabel('Independent-state round-trip max error')
    ax.set_ylabel('Cumulative fraction'); ax.legend(frameon=False)
    ax.text(.02, .03, 'Errors below 10$^{-16}$ shown at 10$^{-16}$', transform=ax.transAxes, fontsize=9)
    ax.set_title('(c) Encode–expand consistency', loc='left')
    ax = axes[1, 1]; ax.set_axis_off()
    lines = ['Rule catalog: all single-orbit types passed']
    for kind, label in [('space', '230 SG'), ('layer', '80 LG'), ('plane', '17 PG')]:
        group = catalog[catalog.kind == kind]
        lines.append(f'{label}: {int(group.passed.sum()):,} / {len(group):,} orbit types')
    lines += ['', 'Source-CIF re-encoding: sampled checks']
    for dataset, label, _ in DATASETS:
        group = source[source.dataset == dataset]
        lines.append(f'{label}: {int(group.source_reencode_passed.sum())}/{len(group)} checked')
    lines += ['', 'Full source-CIF re-encoding: pending',
              'Orbit-instance lists with repetition are not exhaustively enumerated.']
    ax.text(.02, .97, '\n'.join(lines), va='top', transform=ax.transAxes, fontsize=10.5, linespacing=1.65)
    ax.set_title('(d) Catalog scope and source-check scope', loc='left')
    save(fig, 'fig02_representation.png')
    return {'prepared_records': len(frame), 'passed': int(frame.passed.sum()),
            'single_orbit_catalog_types': len(catalog), 'source_reencoded': len(source),
            'source_reencoded_passed': int(source.source_reencode_passed.sum())}


def placeholder(number, panels):
    names = {3: 'fig03_shared_generation.png', 4: 'fig04_continuous_controls.png',
             5: 'fig05_property_control.png', 6: 'fig06_physical_validation.png'}
    if (OUT/names[number]).exists():
        return  # Preserve subsequently filled results or user edits.
    fig, axes = plt.subplots(1, len(panels), figsize=(5.1*len(panels), 3.4), squeeze=False, layout='constrained')
    for i, (axis, (title, missing)) in enumerate(zip(axes[0], panels)):
        axis.set_axis_off()
        axis.add_patch(plt.Rectangle((.01, .05), .98, .84, transform=axis.transAxes,
            facecolor='#f5f6f7', edgecolor='#aab1ba', linestyle='--', linewidth=1.3))
        axis.text(.04, .97, f'({chr(97+i)}) {title}', transform=axis.transAxes, va='top', fontsize=12)
        axis.text(.5, .65, 'Data pending', transform=axis.transAxes, ha='center', color='#697582', fontsize=14)
        axis.text(.5, .39, missing, transform=axis.transAxes, ha='center', va='center',
                  color='#697582', fontsize=10, linespacing=1.7)
    save(fig, names[number])


def joint_results(run):
    from unifiedgroupgen.evaluation import summarize_attempts
    state = json.loads((run/'pipeline.json').read_text(encoding='utf-8'))
    if state['state'] != 'complete':
        raise ValueError('Only complete joint results can enter manuscript figures')
    overview = json.loads((run/'summary.json').read_text(encoding='utf-8'))
    table, attempts, timings = [], [], []
    for job, item in state['jobs'].items():
        for name, checksum in item['output_sha256'].items():
            if file_hash(run/item['directory']/name) != checksum:
                raise ValueError('Archived generation outputs changed')
        if job == 'test':
            continue
        records = [json.loads(s) for s in (run/item['directory']/'samples.jsonl').read_text(encoding='utf-8').splitlines() if s]
        report = summarize_attempts(records)
        if any(overview['generation'][job][key] != value for key, value in report.items()):
            raise ValueError('Stored summary differs from actual attempt records')
        dataset = next(d for d, _, _ in DATASETS if job.startswith(d+'_'))
        row = {'job': job, 'dataset': dataset, 'mode': job[len(dataset)+1:],
               'attempted': report['attempted'], 'generated': report['generated'],
               'target_operation_closure': report['counts']['group_operation_closure'],
               'exact_target_group': report['exact_group_success'],
               'group_unidentified': report['exact_group_unidentified_generated'],
               'group_mismatch': report['exact_group_identified']-report['exact_group_success'],
               'distance_0p6': report['counts']['distance_valid'],
               'no_orbit_overlap': report['counts']['no_coincident_orbits'],
               'geometry_and_exact_group': report['geometry_and_exact_group_success'],
               'unique_requested_groups': len({r['descriptor']['number'] for r in records if r['status'] == 'generated'}),
               'zero_coordinate_dof': sum(r['validation']['coordinate_dof'] == 0 for r in records if r['status'] == 'generated')}
        table.append(row)
        for r in records:
            v = r.get('validation', {})
            attempts.append({'job': job, 'dataset': dataset, 'mode': row['mode'], 'index': r['index'],
                'target_group': r.get('descriptor', {}).get('number'),
                'recovered_group': v.get(f'recovered_{"layer" if dataset == "c2db_51" else "space"}_group'),
                'exact_target_group': v.get('exact_target_group'), 'atoms': v.get('atoms'),
                'group_identification_error': v.get('group_identification_error'),
                'coordinate_dof': v.get('coordinate_dof'), 'metric_residual': v.get('metric_residual'),
                'minimum_distance_angstrom': v.get('minimum_distance_angstrom')})
        timing = json.loads((run/item['directory']/'sampling.status.json').read_text(encoding='utf-8'))
        timings.append({'job': job, 'attempted': timing['attempted'], 'seconds': timing['seconds'],
                        'seconds_per_attempt': timing['seconds']/timing['attempted'],
                        'estimated_4096_seconds_same_settings': timing['seconds']*4096/timing['attempted']})
    frame, raw = pd.DataFrame(table), pd.DataFrame(attempts)
    tables = OUT/'tables'; tables.mkdir(exist_ok=True)
    frame.to_csv(tables/'joint_generation_preliminary.csv', index=False)
    raw.to_csv(tables/'joint_generation_attempts.csv.gz', index=False, compression='gzip')
    pd.DataFrame(timings).to_csv(tables/'joint_sampling_time.csv', index=False)
    fig, axis = plt.subplots(figsize=(7.5, 3.7), layout='constrained')
    for i, (mode, color) in enumerate([('unconditional', '#437da6'), ('conditional', '#d68939')]):
        means = [next(t['seconds_per_attempt'] for t in timings if t['job'] == f'{d}_{mode}') for d, _, _ in DATASETS]
        positions = np.arange(3)+(i-.5)*.34
        axis.bar(positions, means, width=.32, color=color, label=mode.capitalize())
        for x, y in zip(positions, means):
            axis.text(x, y+.035, f'{y:.3f}', ha='center', fontsize=10)
    axis.set_xticks(range(3), [x[1] for x in DATASETS]); axis.set_ylim(0, 2.5)
    axis.set_ylabel('Seconds per attempt (mean over 256)'); axis.legend(frameon=False, loc='upper left')
    axis.set_title('Recorded sampling wall time · 64 Heun steps · includes identification / I/O')
    save(fig, 'figS_joint_sampling_cost.png')
    test = json.loads((run/state['jobs']['test']['directory']/'result.json').read_text(encoding='utf-8'))
    test_rows = [{'dataset': dataset, 'requested': v['requested_records'], 'evaluated': v['evaluated_records'],
                  **v['loss_on_supported_records'], **{k+'_labels': n for k, n in v['property_label_counts'].items()}}
                 for dataset, v in test['source_breakdown'].items()]
    pd.DataFrame(test_rows).to_csv(tables/'joint_heldout_test.csv', index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.3), layout='constrained')
    metrics = ['target_operation_closure', 'exact_target_group', 'distance_0p6', 'no_orbit_overlap']
    labels = ['Target\noperations', 'Exact\ngroup', 'Distance\n≥0.6 Å', 'No orbit\noverlap']
    for axis, (dataset, label, _) in zip(axes, DATASETS):
        for i, (mode, color) in enumerate([('unconditional', '#437da6'), ('conditional', '#d68939')]):
            row = frame[(frame.dataset == dataset) & (frame['mode'] == mode)].iloc[0]
            x = np.arange(4)+(i-.5)*.35
            axis.bar(x, [row[k]/row.attempted for k in metrics], width=.33, color=color,
                     label=f'{mode.capitalize()}, n={row.attempted}')
            for j, key in enumerate(metrics):
                axis.text(x[j], row[key]/row.attempted+.02, str(row[key]), ha='center', fontsize=8)
        axis.set_xticks(range(4), labels, fontsize=9); axis.set_ylim(0, 1.15)
        axis.set_ylabel('Passed / all attempts'); axis.set_title(label)
    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_legend, frameon=False, fontsize=10, loc='lower center',
               bbox_to_anchor=(.5, -.06), ncol=2)
    fig.suptitle('One shared checkpoint · preliminary symmetry and geometry screens', fontsize=13)
    save(fig, 'fig03_shared_generation.png')
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.7), layout='constrained')
    for axis, (dataset, label, _) in zip(axes, DATASETS):
        for mode, color in [('unconditional', '#437da6'), ('conditional', '#d68939')]:
            values = np.sort(raw[(raw.dataset == dataset) & (raw['mode'] == mode)].coordinate_dof.dropna())
            axis.step(values, np.arange(1, len(values)+1)/len(values), where='post', color=color, label=mode.capitalize())
        axis.set_title(label); axis.set_xlabel('Coordinate DOF'); axis.set_ylabel('Cumulative fraction')
        axis.set_ylim(0, 1.02); axis.legend(frameon=False, fontsize=9)
    save(fig, 'figS_joint_generated_dof.png')
    imported = run/'import_verification.json'
    verification = json.loads(imported.read_text(encoding='utf-8')) if imported.exists() else None
    report = {'checkpoint_sha256': state['identity']['checkpoint_sha256'], 'checkpoint_epoch': state['checkpoint_epoch'],
        'pipeline_sha256': file_hash(run/'pipeline.json'), 'summary_sha256': file_hash(run/'summary.json'),
        'test_result_sha256': file_hash(run/state['jobs']['test']['directory']/'result.json'),
        'archive_sha256': verification['archive_sha256'] if verification else None,
        'test_split_sha256': test['test_split_sha256'], 'test': test_rows, 'generation': table,
        'scope': 'Raw recorded symmetry and 0.6 Å geometric diagnostics, not SMACT/COV/physical/property certification. '
                 'Unknown group identifications remain in the denominator. Timing includes identification and I/O.'}
    (tables/'joint_preliminary_summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def plot_layer_connectivity(frame):
    fig, axis = plt.subplots(figsize=(7.4, 4.1), layout='constrained')
    bottom = np.zeros(len(frame))
    for key, label, color in [('rank_2', 'Rank 2', '#4d9b87'), ('rank_1', 'Rank 1', '#d68939'),
                              ('rank_0', 'Rank 0', '#8996a3'), ('unknown', 'Unknown', '#b36a88')]:
        values = frame[key].to_numpy()/frame.attempted.to_numpy()
        axis.bar(range(len(frame)), values, bottom=bottom, label=label, color=color, width=.62)
        for i, (count, value) in enumerate(zip(frame[key], values)):
            if value > .035:
                axis.text(i, bottom[i]+value/2, str(count), ha='center', va='center', fontsize=11)
        bottom += values
    axis.set_xticks(range(len(frame)), [f'{r.label}\n(n={r.attempted})' for r in frame.itertuples()])
    axis.set_ylim(0, 1.02); axis.set_ylabel('Fraction of structures')
    axis.set_title('Layer connectivity · bonds within 1.2 × covalent-radius sum', fontsize=12)
    axis.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')
    save(fig, 'figS_layer_connectivity.png')


def standard_results(path, joint):
    if joint is None:
        raise ValueError('Standard results require the verified joint run for identity checks')
    report = json.loads(path.read_text(encoding='utf-8'))
    if report['state'] != 'complete':
        raise ValueError('Standard evaluation is not complete')
    if (report['checkpoint_sha256'] != joint['checkpoint_sha256'] or
            report['generation_pipeline_sha256'] != joint['pipeline_sha256'] or
            report['test_sha256'] != joint['test_split_sha256']):
        raise ValueError('Standard metrics are not bound to this joint generation and test split')
    expected = {row['job']: row for row in joint['generation']}
    if set(report['jobs']) != set(expected):
        raise ValueError('Standard metrics must cover all six joint generation jobs')
    for name, item in report['jobs'].items():
        if item['attempted'] != expected[name]['attempted']:
            raise ValueError('Standard attempt count differs from the archived generation')
        if file_hash(path.parent/f'{name}.csv') != item['metrics_csv_sha256']:
            raise ValueError('Per-attempt standard metrics changed after evaluation')
    tables = OUT/'tables'
    for name in report['jobs']:
        shutil.copyfile(path.parent/f'{name}.csv', tables/f'{name}.csv')
    destination = tables/'joint_standard_summary.json'
    shutil.copyfile(path, destination)
    manifest = Path(__file__).parent/'standard_results_manifest.json'
    manifest.write_text(json.dumps({'experiments': [{'label': 'UnifiedGroupGen (shared)',
        'model_seed': 42, 'metrics': str(destination.relative_to(ROOT)).replace('\\', '/')}]}, indent=2),
        encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT/'scripts/build_paper_results.py'),
                    '--manifest', str(manifest), '--output', 'assets/paper/standard'], check=True)
    reference = json.loads((path.parent/'c2db_51_reference.json').read_text(encoding='utf-8'))
    if (list(reference['identity']['files'].values()) != report['test_sha256'] or
            reference['identity']['metric_code_sha256'] != report['evaluation_code_sha256']['standard_metrics.py']):
        raise ValueError('Layer connectivity reference differs from the bound test split')
    populations = [('Test reference', pd.DataFrame(reference['rows']))]
    for mode in ('unconditional', 'conditional'):
        populations.append((mode.capitalize(), pd.read_csv(tables/f'c2db_51_{mode}.csv')))
    counts = []
    for label, data in populations:
        ranks = data.periodic_connectivity_rank
        counts.append({'label': label, 'attempted': len(data),
                       **{f'rank_{i}': int(ranks.eq(i).sum()) for i in range(3)},
                       'unknown': int(ranks.isna().sum())})
    counts = pd.DataFrame(counts)
    counts.to_csv(tables/'layer_connectivity_counts.csv', index=False)
    plot_layer_connectivity(counts)
    return {'summary': str(destination.relative_to(ROOT)).replace('\\', '/'),
            'summary_sha256': file_hash(destination), 'jobs': len(report['jobs'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--examples-only', action='store_true')
    parser.add_argument('--joint-run', help='Verified imported evaluate_joint directory, relative to repository')
    parser.add_argument('--standard-metrics', help='Complete standard-evaluation summary bound to --joint-run')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'mathtext.fontset': 'stix', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
    examples()
    if args.examples_only:
        return
    statistics = representation()
    joint = joint_results((ROOT/args.joint_run).resolve()) if args.joint_run else None
    standard = standard_results((ROOT/args.standard_metrics).resolve(), joint) if args.standard_metrics else None
    placeholder(3, [('Shared 3D / 2D quality', 'Three-source test values\nValidity · exact group · COV · VUN'),
                    ('Matched comparison', 'Same-split separate models\nExternal baselines · budgets · repeat seeds')])
    placeholder(4, [('Learned continuous distribution', 'Matched descriptor / seed\nLegal prior versus trained flow'),
                    ('Additional hard constraints', 'Fixed lattice / partial coordinates\nObservation residual · quality · diversity')])
    placeholder(5, [('Independent property response', 'TRAIN-quantile targets · CFG scan\nIndependent predictions or calculations'),
                    ('Valid target yield', 'Property tolerances · all-attempt denominator\nJoint hits · diversity · uncertainty')])
    placeholder(6, [('Relaxed candidate quality', 'Compatible energy references · reference phases\nRelaxation · hull · unique / novel'),
                    ('Physical verification', 'Representative candidate structures\nIndependent DFT / phonons as required')])
    inputs = [Path(__file__), Path(__file__).parent/'material_examples.json', ROOT/'scripts/build_paper_results.py',
              ROOT/'assets/representation/summary.json', ROOT/'assets/representation/records.csv.gz',
              ROOT/'assets/representation/catalog.csv', ROOT/'assets/representation/source_reencoding.csv']
    report = {'input_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'):
                  hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                  if p.suffix in ('.py', '.json', '.csv') else file_hash(p) for p in inputs},
              'input_hash_scope': 'Text authoring inputs use LF-normalized SHA256 for cross-platform replay; '
                                  'compressed inputs and imported metric artifacts retain exact-byte SHA256.',
              'representation': statistics,
              'completed_remote_jobs_reported_by_user': ['test', 'mp_20_unconditional', 'mp_20_conditional',
                  'mpts_52_unconditional', 'mpts_52_conditional', 'c2db_51_unconditional', 'c2db_51_conditional'],
              'remote_outcome_metrics_imported': joint is not None,
              'joint_outcome_summary': 'assets/paper/tables/joint_preliminary_summary.json' if joint else None,
              'standard_outcome': standard,
              'scope': 'Representation uses audited local records. Joint outcome values, when supplied, are '
                       'verified against archived attempt files. Encoded examples are not generated candidates; '
                       'symmetry and distance screens do not certify physical stability or target properties.'}
    (OUT/'figure_data_manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(statistics))


if __name__ == '__main__':
    main()
