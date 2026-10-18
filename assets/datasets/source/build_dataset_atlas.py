"""Reproduce PNG dataset statistics and seeded real-structure galleries.

Uses the prepared source records and their verified statistics, without model
sampling, training, or alteration of source structures. See ../ATLAS_README.md.
"""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
from pymatgen.core import Composition, Element

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT/'assets/datasets'
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy
from unifiedgroupgen.structure_rendering import element_color, render_structure
from unifiedgroupgen.symmetry import Descriptor, group_table

DATASETS = [('MP20', 'mp_20', '#3468A5'), ('MPTS52', 'mpts_52', '#C96C46'),
            ('C2DB', 'c2db_51', '#167D8D')]
PROPERTIES = [('formation_energy', 'Formation energy (eV/atom)'),
              ('band_gap', 'Band gap (eV)'),
              ('energy_above_hull', 'Energy above hull (eV/atom)')]


def save(fig, name):
    target = OUT/name
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=300, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)
    return target.relative_to(OUT).as_posix()


def axes_style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#E5E8EC', linewidth=.55)
    ax.set_axisbelow(True)


def formula_of(elements):
    composition = Composition({Element.from_Z(int(z)).symbol: n
                               for z, n in Counter(elements).items()})
    formula = composition.reduced_formula
    return formula, re.sub(r'(\d+)', r'$_{\1}$', formula)


def load_sources(seed, frame):
    """Verify the existing statistics provenance and sample without replacement."""
    provenance = json.loads((OUT/'tables/statistics.json').read_text(encoding='utf-8'))
    samples, counts, dimension_rows, hashes = {}, {}, [], []
    for dataset_index, (name, key, _) in enumerate(DATASETS):
        total = int((frame.dataset == name).sum())
        indices = np.random.default_rng(seed+dataset_index).choice(total, 20, replace=False)
        chosen, index = {}, 0
        occurrence, dimensions = Counter(), Counter()
        paths = [x for x in provenance['source_files'] if f'/{key}/' in x['path']]
        assert len(paths) == 3
        for item in paths:
            path = ROOT/item['path']
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != item['sha256']:
                raise ValueError(f'Statistics/source hash mismatch: {item["path"]}')
            hashes.append({'path': item['path'], 'sha256': digest})
            for line in raw.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if index in indices:
                    chosen[index] = record
                descriptor = record['descriptor']
                table = {w.letter: w for w in group_table(descriptor['kind'], descriptor['number'])}
                present = set()
                for orbit in descriptor['orbits']:
                    dimension = table[orbit['letter']].get_dof()
                    dimensions[dimension] += 1
                    present.add(dimension)
                occurrence.update(present)
                index += 1
        assert index == total
        samples[name] = [chosen[int(i)] for i in indices]
        counts[name] = total
        for dimension in range(4):
            dimension_rows.append({'dataset': name, 'coordinate_dof_per_orbit': dimension,
                'structures_with_dimension': occurrence[dimension], 'total_structures': total,
                'orbit_instances': dimensions[dimension]})
        print(f'[DATA] {name}: {total:,} verified records; 20 random examples', flush=True)
    dimensions = pd.DataFrame(dimension_rows)
    dimensions.to_csv(OUT/'tables/wyckoff_dimension_counts.csv', index=False)
    return samples, counts, dimensions, hashes


def group_distributions(frame):
    fig, axes = plt.subplots(3, 1, figsize=(13.8, 8.2), layout='constrained')
    space = [(1, 2, 'Triclinic'), (3, 15, 'Monoclinic'), (16, 74, 'Orthorhombic'),
             (75, 142, 'Tetragonal'), (143, 167, 'Trigonal'), (168, 194, 'Hexagonal'),
             (195, 230, 'Cubic')]
    layer = [(1, 2, 'Triclinic'), (3, 7, 'Mon. oblique'), (8, 18, 'Mon. rectangular'),
             (19, 48, 'Orthorhombic'), (49, 64, 'Tetragonal'), (65, 80, 'Hexagonal')]
    for row, (ax, (name, _, color)) in enumerate(zip(axes, DATASETS)):
        data = frame[frame.dataset == name]
        maximum = 80 if name == 'C2DB' else 230
        bins = data.group.value_counts().reindex(range(1, maximum+1), fill_value=0)
        heights = 100*bins/len(data)
        upper = heights.max()*1.35
        for band_index, (lo, hi, label) in enumerate(layer if name == 'C2DB' else space):
            ax.axvspan(lo-.5, hi+.5, color='#F1F4F7', alpha=.7 if band_index % 2 == 0 else .2, zorder=0)
            if lo > 1:
                ax.axvline(lo-.5, color='#9CA8B4', linestyle=':', linewidth=.8)
            ax.text((lo+hi)/2, upper*.96, label, ha='center', va='top', fontsize=9,
                    rotation=90 if (hi-lo+1)/maximum < .09 else 0, color='#596574')
        ax.bar(bins.index, heights, width=.85, color=color, zorder=2)
        ax.set(xlim=(.4, maximum+.6), ylim=(0, upper), ylabel='Structures (%)',
               xlabel='Layer-group number' if name == 'C2DB' else 'Space-group number')
        ax.set_xticks([1, 10, 20, 30, 40, 50, 60, 70, 80] if maximum == 80
                      else [1, 25, 50, 75, 100, 125, 150, 175, 200, 230])
        ax.set_title(f'{chr(97+row)}  {name}   |   N = {len(data):,}   |   {(bins > 0).sum()}/{maximum} groups observed', loc='left')
        axes_style(ax)
    return save(fig, '09_group_distribution.png')


def periodic_position(number):
    if 57 <= number <= 71:
        return number-54, 8
    if 89 <= number <= 103:
        return number-86, 9
    element = Element.from_Z(number)
    return element.group, element.row


def periodic_tables(counts):
    elements = pd.read_csv(OUT/'tables/element_prevalence.csv')
    files = []
    norm, cmap = LogNorm(vmin=.001, vmax=100), plt.get_cmap('Blues')
    for name, key, _ in DATASETS:
        lookup = elements[elements.dataset == name].set_index('atomic_number').records_with_element
        fig, ax = plt.subplots(figsize=(13.8, 7.8), layout='constrained')
        for number in range(1, 119):
            column, row = periodic_position(number)
            x, y = column-1, 10-row
            count = int(lookup.get(number, 0)); fraction = 100*count/counts[name]
            color = cmap(norm(fraction)) if count else '#F1F2F4'
            ax.add_patch(Rectangle((x, y), .94, .92, facecolor=color, edgecolor='white', linewidth=.8))
            ink = 'white' if count and np.dot(color[:3], [.2126, .7152, .0722]) < .55 else '#25364A'
            ax.text(x+.08, y+.76, str(number), fontsize=6.5, color=ink)
            ax.text(x+.47, y+.49, Element.from_Z(number).symbol, fontsize=13, ha='center', va='center', color=ink, weight='bold')
            ax.text(x+.47, y+.15, f'{count:,}' if count else '—', fontsize=7, ha='center', color=ink)
        for row, label in [(6, '57–71'), (7, '89–103')]:
            ax.text(2.47, 10-row+.45, label, ha='center', va='center', fontsize=8, color='#6B7280')
        for column in range(1, 19):
            ax.text(column-.53, 10.1, str(column), ha='center', fontsize=8, color='#6B7280')
        ax.text(2.6, 8.3, f'{name}   |   N = {counts[name]:,}\n{len(lookup)} elements observed', fontsize=18, linespacing=1.6)
        ax.text(2.6, 7.3, 'Cell labels: number of structures containing the element\nOne count per element per structure; all reference splits', fontsize=10, color='#596574', linespacing=1.5)
        ax.set(xlim=(-.2, 18), ylim=(.7, 10.5), aspect='equal')
        ax.set_axis_off()
        colorbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                               orientation='horizontal', fraction=.035, pad=.035, aspect=45)
        colorbar.set_label('Structures containing the element (%) · shared logarithmic scale')
        colorbar.set_ticks([.001, .01, .1, 1, 10, 100], labels=['0.001', '0.01', '0.1', '1', '10', '100'])
        files.append(save(fig, f'10_periodic_table_{key}.png'))
    return files


def chemical_complexity(frame):
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 9.4), layout='constrained')
    rows = []
    for row, (name, _, color) in enumerate(DATASETS):
        data = frame[frame.dataset == name]
        for column, (field, label) in enumerate([
            ('elements', 'Elements per structure'), ('source_atoms', 'Atoms in source cell'),
            ('expanded_atoms', 'Atoms in standardized cell')]):
            ax = axes[row, column]
            frequencies = data[field].value_counts().sort_index()
            ax.bar(frequencies.index, 100*frequencies/len(data), width=.86, color=color)
            median = data[field].median()
            ax.axvline(median, color='#263444', linestyle='--', linewidth=1)
            ax.text(.97, .94, f'Median = {median:g}\nRange = {data[field].min():g}–{data[field].max():g}',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
            ax.set(xlabel=label, ylabel='Structures (%)', xlim=(.2, data[field].max()+.8))
            ax.set_title(f'{chr(97+row*3+column)}  {name}   |   N = {len(data):,}', loc='left')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
            axes_style(ax)
            rows.extend({'dataset': name, 'field': field, 'value': int(value), 'count': int(count),
                         'percent': 100*count/len(data)} for value, count in frequencies.items())
    pd.DataFrame(rows).to_csv(OUT/'tables/chemistry_cell_size_counts.csv', index=False)
    return save(fig, '11_chemistry_cell_size.png')


def property_distributions(frame):
    fig, axes = plt.subplots(3, 3, figsize=(13.8, 9.4), layout='constrained')
    bins_rows = []
    for row, (name, _, color) in enumerate(DATASETS):
        data = frame[frame.dataset == name]
        for column, (field, label) in enumerate(PROPERTIES):
            ax = axes[row, column]; values = data[field].dropna().to_numpy()
            ax.set_title(f'{chr(97+row*3+column)}  {name} · {field.replace("_", " ")}', loc='left', fontsize=11)
            ax.set(xlabel=label, ylabel='Labelled structures per bin (%)')
            axes_style(ax)
            if not len(values):
                ax.text(.5, .5, 'No labels in the prepared dataset\nMissing values are not zeros', transform=ax.transAxes,
                        ha='center', va='center', color='#6B7280', fontsize=11, linespacing=1.6)
                ax.set_xticks([]); ax.set_yticks([])
                continue
            counts, edges = np.histogram(values, bins=60)
            ax.stairs(100*counts/len(values), edges, fill=True, color=color, alpha=.8, linewidth=.6)
            median = np.median(values)
            ax.axvline(median, color='#263444', linestyle='--', linewidth=.9)
            ax.text(.97, .94, f'n = {len(values):,}/{len(data):,}\nMedian = {median:.3g}\nZero labels = {100*np.mean(values == 0):.1f}%',
                    transform=ax.transAxes, ha='right', va='top', fontsize=9)
            ax.set_xlim(edges[0], edges[-1])
            ax.set_ylim(0, 100*counts.max()/len(values)*1.4)
            bins_rows.extend({'dataset': name, 'property': field, 'bin_left': left, 'bin_right': right,
                              'count': int(count), 'percent_of_finite_labels': 100*count/len(values)}
                             for left, right, count in zip(edges[:-1], edges[1:], counts))
    pd.DataFrame(bins_rows).to_csv(OUT/'tables/property_histogram_bins.csv', index=False)
    return save(fig, '12_property_distributions.png')


def wyckoff_dimensions(dimensions):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout='constrained')
    for ax, (name, _, color) in zip(axes, DATASETS):
        rows = dimensions[dimensions.dataset == name]
        values = 100*rows.structures_with_dimension/rows.total_structures
        ax.bar(range(4), values, color=color, width=.65)
        for dimension, value in enumerate(values):
            ax.text(dimension, value+1.8, f'{value:.1f}%', ha='center', fontsize=10)
        ax.set(xticks=range(4), xlabel='Coordinate DOF of a Wyckoff orbit',
               ylabel='Structures containing this DOF (%)', ylim=(0, 110))
        ax.set_title(f'{name}   |   N = {int(rows.total_structures.iloc[0]):,}', loc='left')
        axes_style(ax)
    return save(fig, '13_wyckoff_dimensions.png')


def structure_view(record, size=(750, 650), view='isometric', arrows=False):
    chart = compiled_chart(Descriptor.from_dict(record['descriptor']))
    lattice, coords, _ = expand_numpy(chart, record['state'])
    rows = lattice.T.copy(); fractional = coords.copy()
    layer = record['descriptor']['kind'] == 'layer'
    if layer:
        # Display the in-plane cell outline 1 Angstrom below the lowest atom.
        fractional[:, 2] += 1-coords[:, 2].min()
    vectors = None
    if arrows:
        vectors = chart.coordinate_basis[:, :, 0] @ lattice.T
        vectors *= 3/max(np.linalg.norm(vectors, axis=1).max(), 1e-12)
    image = render_structure(rows, fractional, chart.elements, periodic=tuple(chart.periodic),
        repeat=(2, 2, 1) if layer else (1, 1, 1), size=size, view=view,
        vectors=vectors, vector_color='#167D8D')
    return image, chart


def symmetry_examples():
    records = json.loads((ROOT/'assets/paper/source/material_examples.json').read_text(encoding='utf-8'))['examples']
    fig = plt.figure(figsize=(13.5, 7.4), layout='constrained')
    grid = fig.add_gridspec(2, 3, width_ratios=[1.1, 1.1, .95])
    for row, record in enumerate(records):
        for column, view in enumerate(['isometric', 'top' if row == 0 else 'side']):
            image, chart = structure_view(record, arrows=True, view=view)
            ax = fig.add_subplot(grid[row, column]); ax.imshow(image); ax.set_axis_off()
            ax.set_title(f'{chr(97+row*3+column)}  '+('TiO$_2$ · SG 136' if row == 0 else 'MoS$_2$ · LG 78')+f' · {view}', loc='left')
            if column == 0:
                for number in sorted(set(chart.elements)):
                    ax.scatter([], [], s=65, color=np.array(element_color(number))/255,
                               label=Element.from_Z(int(number)).symbol)
                ax.legend(loc='upper right', frameon=False)
        ax = fig.add_subplot(grid[row, 2]); ax.set_axis_off()
        ax.set_title(f'{chr(99+row*3)}  Independent orbit parameters', loc='left')
        if row == 0:
            text = ('Rutile TiO$_2$ · mp-2657\n\nTi 2a: $(0,0,0)$\nCoordinate DOF: 0; multiplicity: 2\n\n'
                    'O 4f: $(u,u,0)$\nCoordinate DOF: 1; multiplicity: 4\n\n'
                    '$u=0.30458$\n$z=(k_1,k_2,u)$\n2 lattice + 1 coordinate DOF\n\n6 atoms per standardized cell')
        else:
            text = ('Monolayer MoS$_2$ · 1MoS2-1\n\nMo 1a: $(0,0,0)$\nCoordinate DOF: 0; multiplicity: 1\n\n'
                    'S 2f: $(2/3,1/3,\pm h)$\nCoordinate DOF: 1; multiplicity: 2\n\n'
                    '$h=1.56356$ Å\n$z=(k_1,h)$\n1 lattice + 1 height DOF\n\n3 atoms per cell; 2 × 2 display')
        ax.text(.02, .98, text, transform=ax.transAxes, va='top', fontsize=10.5, linespacing=1.4)
    fig.supxlabel('Green arrows: the allowed coordinate direction, rescaled for visibility; real encoded dataset examples', fontsize=10)
    return save(fig, '08_tio2_mos2_symmetry.png')


def galleries(samples, seed):
    files, selected = [], []
    for dataset_index, (name, key, _) in enumerate(DATASETS):
        fig, axes = plt.subplots(4, 5, figsize=(15, 12.4), layout='constrained')
        for index, (record, ax) in enumerate(zip(samples[name], axes.flat), 1):
            image, chart = structure_view(record, size=(720, 650))
            formula, label = formula_of(chart.elements)
            group = record['descriptor']['number']; kind = 'LG' if key == 'c2db_51' else 'SG'
            ax.imshow(image); ax.set_axis_off()
            ax.set_title(label, fontsize=13, pad=3)
            ax.text(.5, -.015, f'{record["id"]}  |  {kind} {group}  |  N = {chart.num_atoms}',
                    transform=ax.transAxes, ha='center', va='top', fontsize=8)
            individual, single = plt.subplots(figsize=(3.6, 3.7), layout='constrained')
            single.imshow(image); single.set_axis_off(); single.set_title(label, fontsize=16)
            single.text(.5, -.015, f'{record["id"]}  |  {kind} {group}  |  N = {chart.num_atoms}',
                        transform=single.transAxes, ha='center', va='top', fontsize=8.5)
            path = save(individual, f'structures/{key}/{index:02d}.png'); files.append(path)
            selected.append({'dataset': name, 'id': record['id'], 'formula': formula,
                'group_kind': record['descriptor']['kind'], 'group': group, 'atoms': chart.num_atoms,
                'seed': seed+dataset_index, 'file': path, 'source_cif_sha256': record['provenance']['source_cif_sha256'],
                'descriptor': record['descriptor'], 'state': record['state']})
        fig.suptitle(f'{name} · 20 real dataset structures sampled without replacement · seed {seed+dataset_index}', fontsize=17)
        files.append(save(fig, f'14_random_structures_{key}.png'))
        print(f'[PNG] {name}: 20 individual views and one gallery', flush=True)
    (OUT/'tables/random_structure_examples.json').write_text(json.dumps(selected, indent=2), encoding='utf-8')
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261018)
    args = parser.parse_args()
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
        'axes.titlesize': 12, 'axes.labelsize': 10, 'xtick.labelsize': 9,
        'ytick.labelsize': 9, 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
    frame = pd.read_csv(OUT/'tables/record_statistics.csv.gz')
    samples, counts, dimensions, hashes = load_sources(args.seed, frame)
    files = [symmetry_examples(), group_distributions(frame), *periodic_tables(counts),
             chemical_complexity(frame), property_distributions(frame), wyckoff_dimensions(dimensions)]
    files += galleries(samples, args.seed)
    manifest = {'seed': args.seed, 'counts': counts, 'source_files': hashes,
        'statistics_sha256': hashlib.sha256((OUT/'tables/record_statistics.csv.gz').read_bytes()).hexdigest(),
        'files': files, 'scope': 'All reference splits; real encoded structures, no model-generated structures',
        'style_references': ['SGEquiDiff Figure 1', 'SLayerGen Figure 6', 'CrystalFlow Supplementary Figure S3-1', 'DiffCSP Figure 12'],
        'structure_style': 'Existing Crystal Toolkit inspired renderer: matte spheres, contact cylinders, translucent coordination hulls',
        'layer_display': '2x2 in plane; basal outline 1 Angstrom below lowest atom via display-only translation; no normal repetition',
        'units': 'TiO2: fractional coordinates; MoS2: fractional in-plane coordinates and Cartesian height in Angstrom',
        'renderer_sha256': hashlib.sha256((ROOT/'src/unifiedgroupgen/structure_rendering.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest()}
    (OUT/'tables/atlas_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(f'[COMPLETE] {len(files)} PNG files; statistics and sample provenance saved.', flush=True)


if __name__ == '__main__':
    main()
