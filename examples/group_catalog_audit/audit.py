"""Read-only catalog/data audit; writes reports beside this script, never training data."""
from unifiedgroupgen.common.configuration import project_root, resolve_path
import csv
import hashlib
import itertools
import json
import sys
from dataclasses import asdict
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
import spglib
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from unifiedgroupgen.group_constraints.compiler import (Descriptor, compile_descriptor, group_table,
                                      metric_chart, wp_arrays)

OUT = Path(__file__).resolve().parent


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def table(name, rows, fields):
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean(a):
    a = np.asarray(a).copy()
    a[np.abs(a) < 1e-10] = 0
    return np.round(a, 10).tolist()


def metric_constraints(rotations, dim):
    """Exact rational RREF for constraints on raw symmetric metric entries."""
    indices = [(i, j) for i in range(dim) for j in range(i, dim)]
    standard = []
    for i, j in indices:
        mat = np.zeros((dim, dim))
        mat[i, j] = mat[j, i] = 1
        standard.append(mat)
    rows = set()
    for r in rotations:
        system = np.stack([(r.T @ b @ r - b).ravel() for b in standard], axis=1)
        for row in system:
            rational = tuple(Fraction(float(x)).limit_denominator(48) for x in row)
            if any(rational):
                rows.add(rational)
    matrix = [list(row) for row in sorted(rows)]
    pivot = 0
    for col in range(len(indices)):
        found = next((i for i in range(pivot, len(matrix)) if matrix[i][col]), None)
        if found is None:
            continue
        matrix[pivot], matrix[found] = matrix[found], matrix[pivot]
        scale = matrix[pivot][col]
        matrix[pivot] = [v / scale for v in matrix[pivot]]
        for i in range(len(matrix)):
            if i != pivot and matrix[i][col]:
                scale = matrix[i][col]
                matrix[i] = [a - scale * b for a, b in zip(matrix[i], matrix[pivot])]
        pivot += 1
    names = [f'h{i+1}{j+1}' for i, j in indices]
    equations = []
    for row in matrix[:pivot]:
        parts = [f'({coef})*{name}' for coef, name in zip(row, names) if coef]
        equations.append(' + '.join(parts) + ' = 0')
    return equations


def bounded_count(wps, max_atoms, max_orbits):
    """Unordered nonempty WP multisets; fixed sites once, free sites repeat."""
    dp = np.zeros((max_orbits + 1, max_atoms + 1), dtype=object)
    dp[0, 0] = 1
    for wp in wps:
        mult = wp['multiplicity']
        if mult > max_atoms:
            continue
        ks = range(1, max_orbits + 1) if wp['dof'] else range(max_orbits, 0, -1)
        for k in ks:
            dp[k, mult:] += dp[k-1, :-mult]
    return int(dp.sum()) - 1


def main():
    assert bounded_count([{'multiplicity': 1, 'dof': 3}], 6, 4) == 4
    toy = [{'multiplicity': 1, 'dof': 0}] * 3 + [{'multiplicity': 2, 'dof': 3}]
    brute = sum(0 < sum(v) <= 4 and sum(v[:3]) + 2*v[3] <= 6
                for v in itertools.product(range(2), range(2), range(2), range(4)))
    assert bounded_count(toy, 6, 4) == brute
    catalog, lookup, classes = [], {}, defaultdict(list)
    max_residual = 0.0
    for kind, total in [('space', 230), ('layer', 80), ('plane', 17)]:
        for number in range(1, total+1):
            group = group_table(kind, number)
            root, _, basis = metric_chart(kind, number)
            dim = root.shape[0]
            rotations = [op.rotation_matrix[:dim, :dim] for op in group[0].ops]
            constraints = metric_constraints(rotations, dim)
            wps = []
            for wp in group:
                b, offset, axes = wp_arrays(kind, number, wp.letter)
                wps.append({'letter': wp.letter, 'multiplicity': len(b), 'dof': len(axes)})
            record = {'kind': kind, 'number': number, 'table_symbol': group.symbol,
                      'hall_number': getattr(group, 'hall_number', None) if kind == 'space' else None,
                      'lattice_dof': len(basis), 'H0': clean(root @ root),
                      'E': clean(basis), 'constraints': constraints, 'wyckoff': wps}
            h = root @ torch.matrix_exp(torch.tensor(np.einsum('k,kij->ij', np.arange(len(basis))*.13, basis))).numpy() @ root
            residual = max(np.max(np.abs(r.T @ h @ r - h)) for r in rotations)
            assert np.linalg.eigvalsh(h).min() > 0 and residual < 1e-7
            max_residual = max(max_residual, float(residual))
            assert len(basis) == dim*(dim+1)//2 - len(constraints)
            catalog.append(record)
            lookup[kind, number] = record
            classes[kind, json.dumps(record['E']), tuple(constraints)].append(number)
    save('catalog_all_327_groups.json', catalog)
    class_rows = []
    for (kind, bases, constraints), nums in classes.items():
        class_rows.append({'kind': kind, 'groups': nums, 'E': json.loads(bases), 'constraints': constraints})
    save('basis_classes.json', class_rows)
    table('all_group_lattice_bases.csv', [
        {'kind': r['kind'], 'number': r['number'], 'symbol': r['table_symbol'],
         'hall': r['hall_number'], 'dof': r['lattice_dof'], 'H0': json.dumps(r['H0']),
         'E': json.dumps(r['E']), 'constraints': '; '.join(r['constraints']) or 'none'}
        for r in catalog], ['kind', 'number', 'symbol', 'hall', 'dof', 'H0', 'E', 'constraints'])
    wyckoff136 = []
    for wp in reversed(list(group_table('space', 136))):
        wp.get_site_symmetry()
        wyckoff136.append({'letter': wp.letter, 'multiplicity': int(wp.multiplicity),
                          'dof': int(wp.get_dof()), 'site_symmetry': str(wp.site_symm),
                          'positions': [op.as_xyz_str() for op in wp.ops]})
    save('space136_wyckoff_full.json', {'hall': asdict(spglib.get_spacegroup_type(419)), 'wyckoff': wyckoff136})
    lines = ['# Space group 136, default Hall setting 419', '',
             'Coordinates below are affine parameter maps, not invertible spatial group operations.', '']
    for w in wyckoff136:
        lines += [f"## {w['multiplicity']}{w['letter']} — DOF {w['dof']}; site symmetry {w['site_symmetry']}", '',
                  *[f'- `{p}`' for p in w['positions']], '']
    (OUT/'space136_wyckoff_full.md').write_text('\n'.join(lines), encoding='utf-8')
    summary, group_rows, wp_rows, pattern_rows, missing_rows = {}, [], [], [], []
    missing_examples = []
    space_union_groups, space_union_types, space_union_patterns = set(), set(), set()
    selected_mpts = None
    for config_name in ['mp20_local', 'mpts52_local', 'c2db_official_local']:
        config = yaml.safe_load((resolve_path(ROOT/'configs'/f'{config_name}.yaml')).read_text(encoding='utf-8'))
        name, kind = config['dataset'], config['kind']
        path = resolve_path(ROOT/config['prepared_root'])
        groups, pairs, patterns, decorated = Counter(), Counter(), Counter(), Counter()
        split_pairs = defaultdict(set)
        split_patterns = defaultdict(set)
        inputs = []
        for split in ['train', 'val', 'test']:
            source = path/f'{split}.jsonl'
            inputs.append({'path': str(source), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()})
            with source.open(encoding='utf-8') as stream:
                for line in stream:
                    record = json.loads(line)
                    d = record['descriptor']
                    assert d['kind'] == kind
                    number = d['number']
                    groups[number] += 1
                    letters = tuple(sorted(o['letter'] for o in d['orbits']))
                    pattern = (number, letters)
                    patterns[pattern] += 1
                    decorated[number, tuple(sorted((o['letter'], o['atomic_number']) for o in d['orbits']))] += 1
                    split_patterns[split].add(pattern)
                    wpinfo = {w['letter']: w for w in lookup[kind, number]['wyckoff']}
                    counts = Counter(letters)
                    for letter, amount in counts.items():
                        assert letter in wpinfo
                        assert wpinfo[letter]['dof'] > 0 or amount == 1
                    atoms = sum(wpinfo[l]['multiplicity'] for l in letters)
                    assert atoms <= config['max_atoms'] and len(letters) <= config['max_orbits']
                    assert len(record['state']) == lookup[kind, number]['lattice_dof'] + sum(wpinfo[l]['dof'] for l in letters)
                    for letter in set(letters):
                        pairs[number, letter] += 1
                        split_pairs[split].add((number, letter))
                    if name == 'mpts_52' and selected_mpts is None and 1 < sum(wpinfo[l]['dof'] for l in letters) <= 5 and atoms <= 12:
                        selected_mpts = record
        bound_total = 0
        pair_total = 0
        if kind == 'space':
            space_union_groups.update(groups)
            space_union_types.update(pairs)
            space_union_patterns.update(patterns)
        available = [r for r in catalog if r['kind'] == kind]
        for cat in available:
            number = cat['number']
            bounded = bounded_count(cat['wyckoff'], config['max_atoms'], config['max_orbits'])
            bound_total += bounded
            observed = sum(1 for g, _ in patterns if g == number)
            assert observed <= bounded
            group_rows.append({'dataset': name, 'kind': kind, 'group': number, 'records': groups[number],
                               'observed_unordered_patterns': observed, 'bounded_formal_patterns': bounded,
                               'bounded_unobserved_patterns': bounded-observed})
            by_letter = {w['letter']: w for w in cat['wyckoff']}
            examples_found = 0
            for size in range(1, min(3, config['max_orbits'])+1):
                for combination in itertools.combinations_with_replacement(sorted(by_letter), size):
                    n = sum(by_letter[l]['multiplicity'] for l in combination)
                    repeated_fixed = any(by_letter[l]['dof'] == 0 and count > 1
                                         for l, count in Counter(combination).items())
                    if n > config['max_atoms'] or repeated_fixed or (number, combination) in patterns:
                        continue
                    missing_examples.append({'dataset': name, 'kind': kind, 'group': number,
                                             'orbit_letters_with_repeats': ' '.join(combination),
                                             'atoms': n, 'group_seen': bool(groups[number])})
                    examples_found += 1
                    if examples_found == 5:
                        break
                if examples_found == 5:
                    break
            for wp in cat['wyckoff']:
                pair_total += 1
                row = {'dataset': name, 'kind': kind, 'group': number, **wp,
                       'records_containing': pairs[number, wp['letter']],
                       'within_atom_budget': wp['multiplicity'] <= config['max_atoms']}
                wp_rows.append(row)
                if not row['records_containing']:
                    missing_rows.append(row)
        for (number, letters), count in sorted(patterns.items()):
            pattern_rows.append({'dataset': name, 'kind': kind, 'group': number,
                                 'orbit_letters_with_repeats': ' '.join(letters), 'records': count,
                                 'in_train': (number, letters) in split_patterns['train'],
                                 'in_val': (number, letters) in split_patterns['val'],
                                 'in_test': (number, letters) in split_patterns['test']})
        summary[name] = {'records': sum(groups.values()), 'groups_seen': len(groups),
                         'groups_total': len(available), 'missing_groups': [r['number'] for r in available if not groups[r['number']]],
                         'group_wp_types_seen': len(pairs), 'group_wp_types_total': pair_total,
                         'group_wp_types_missing': pair_total-len(pairs),
                         'observed_unordered_orbit_patterns': len(patterns),
                         'observed_element_decorated_patterns': len(decorated),
                         'max_atoms': config['max_atoms'], 'max_orbits': config['max_orbits'],
                         'bounded_formal_patterns': bound_total, 'bounded_unobserved_patterns': bound_total-len(patterns),
                         'val_patterns_absent_from_train': len(split_patterns['val']-split_patterns['train']),
                         'test_patterns_absent_from_train': len(split_patterns['test']-split_patterns['train']),
                         'input_files': inputs}
        print(name, json.dumps({k: v for k, v in summary[name].items() if k != 'input_files'}), flush=True)
    table('group_coverage.csv', group_rows, list(group_rows[0]))
    table('wyckoff_type_coverage.csv', wp_rows, list(wp_rows[0]))
    table('missing_wyckoff_types.csv', missing_rows, list(missing_rows[0]))
    table('observed_orbit_patterns.csv', pattern_rows, list(pattern_rows[0]))
    table('missing_orbit_pattern_examples.csv', missing_examples, list(missing_examples[0]))
    union = {'groups_seen': len(space_union_groups), 'group_wp_types_seen': len(space_union_types),
             'observed_unordered_orbit_patterns': len(space_union_patterns),
             'missing_groups': sorted(set(range(1, 231))-space_union_groups)}
    save('coverage_summary.json', {'datasets': summary, 'mp20_mpts52_union': union,
         'catalog_checks': {'groups': 327, 'max_metric_residual': max_residual},
         'scope': 'Counts refer to current prepared data and default catalog settings. Patterns are unordered WP multisets, ignore elements and continuous parameters, allow repeated positive-DOF WPs, allow zero-DOF WPs once, and obey configured atom/orbit budgets. Formal counts are not counts of exact-symmetry distinct materials or chemically feasible structures. No exhaustive materialization of missing multiset patterns; missing totals are computed by exact generating-function DP. Full per-group/type absence lists and observed pattern lists are exported.'})
    examples = [json.loads((OUT.parent/'mp20_walkthrough'/'record.json').read_text()),
                json.loads((OUT.parent/'c2db_walkthrough'/'record.json').read_text()), selected_mpts]
    demonstrations = []
    for rec in examples:
        c = compile_descriptor(Descriptor.from_dict(rec['descriptor']))
        end = torch.tensor(rec['state'], dtype=torch.float64)
        start = torch.zeros_like(end)
        start[c.lattice_dof:] = .1 if c.descriptor.kind == 'space' else .5
        periodic = torch.tensor(c.state_periodic)
        delta = torch.where(periodic, (end-start+.5)%1-.5, end-start)
        middle = start+.5*delta
        geometry = c.expand(middle)
        candidate = torch.zeros((c.num_atoms, 3), dtype=torch.float64)
        for i in range(c.num_atoms):
            if np.linalg.norm(c.coordinate_basis[i]) > 0:
                candidate[i] = torch.tensor([.1, .2, .3])
                break
        qdot = c.reduce_velocity(candidate, geometry['lattice'])
        legal = c.coordinate_velocity(qdot)
        constraints = c.constraint_matrix(geometry['coordinates'].numpy())
        assert np.max(np.abs(constraints@legal.numpy().ravel())) < 1e-7
        kdot = np.einsum('kij,ij->k', c.metric_basis, np.diag([1., 2., 3.])[:c.root.shape[0], :c.root.shape[0]])
        demonstrations.append({'id': rec['id'], 'descriptor': rec['descriptor'], 'state': rec['state'],
            'atoms': c.num_atoms, 'directed_edges': len(c.edge_index[0]), 'B': clean(c.coordinate_basis), 'b': clean(c.b),
            'illustrative_start_not_random_training_draw': start.tolist(), 'midpoint': middle.tolist(),
            'training_target_velocity': delta.tolist(), 'illustrative_candidate_not_trained_prediction': candidate.tolist(),
            'legal_q_velocity': qdot.tolist(), 'legal_atom_velocity': legal.tolist(),
            'illustrative_lattice_matrix': np.diag([1., 2., 3.]).tolist(), 'legal_k_velocity': kdot.tolist(),
            'constraint_residual': float(np.max(np.abs(constraints@legal.numpy().ravel())))})
    save('three_sample_flow_examples.json', demonstrations)
    readme = ['# 群目录、轨道覆盖与连续参数审计', '',
              '日期：2026-10-17。读取当前本地 prepared 数据；没有训练、修改模型或评估材料稳定性。', '',
              '## 统计口径', '',
              '- 群—Wyckoff 类型：一个群编号加一个字母；完整目录有限。',
              '- 轨道组合：忽略元素与轨道顺序，保留同一字母的重复次数。',
              '- 无原子/轨道上限时，正自由度轨道可以任意重复，组合总数无限。',
              '- 有界计数使用各配置 max_atoms/max_orbits，零自由度轨道至多一次；用整数动态规划精确计数。',
              '- 有界计数不是物理非等价材料数，不保证精确恢复目标群、无碰撞、化学合理或稳定。',
              '- 覆盖统计合并 train/val/test，只是描述性审计，不能据此说训练集已覆盖全部组合。',
              '- 不同数据集上限不同，其有界组合总数不能直接用于生成能力排名。', '',
              '## 数据覆盖', '',
              '| 数据 | 记录 | 群 | 群-WP类型 | 已见轨道组合 | 有界形式组合总数 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, s in summary.items():
        readme.append(f"| {name} | {s['records']} | {s['groups_seen']}/{s['groups_total']} | {s['group_wp_types_seen']}/{s['group_wp_types_total']} | {s['observed_unordered_orbit_patterns']} | {s['bounded_formal_patterns']} |")
    readme += ['', '## 文件', '',
               '- `catalog_all_327_groups.json`：每个群的 H0、全部 E 基底、原始度量的线性约束、Wyckoff 重数/自由度。',
               '- `basis_classes.json`：相同基底和约束的群编号分类。',
               '- `all_group_lattice_bases.csv`：327 个群的晶格信息。plane 的 symbol 来自嵌入层群表，非独立平面群名称。',
               '- `space136_wyckoff_full.md/json`：Hall 419 的完整 136 号 Wyckoff 表。',
               '- `group_coverage.csv`：每个数据集每个群的记录数及有界组合计数。',
               '- `wyckoff_type_coverage.csv`：所有群-WP类型的覆盖；计数指包含该类型的记录数。',
               '- `missing_wyckoff_types.csv`：完全未见的群-WP类型列表。',
               '- `observed_orbit_patterns.csv`：所有已见组合及 train/val/test 存在性。',
               '- `missing_orbit_pattern_examples.csv`：每群至多五个未见形式组合，来自最多三个轨道的搜索；不是全部缺失组合。',
               '- `coverage_summary.json`：汇总、输入文件 SHA256、两套三维数据并集和验证范围。',
               '- `three_sample_flow_examples.json`：真实终点、人工指定起点、训练目标速度和示意候选输出的合法投影。候选不是训练模型预测。', '',
               '## 本次验证', '',
               f'- 327 个群的一个确定性参数点检查正定性和操作不变，最大度量残差 {max_residual:.3e}。',
               '- 逐记录核对字母合法性、零维轨道不重复、状态维数和配置预算。',
               '- 动态规划用两个可直接计数的小实例交叉检查；组合总数未通过逐项穷举复核。',
               '- 三个样本的候选速度转换后，检查完整群约束残差。',
               '- 此审计不是全数据逐结构的独立空间群重识别，也没有重新认证原始数据来源。', '',
               '复现：使用项目已有依赖环境运行本目录 `audit.py`。']
    (OUT/'README.md').write_text('\n'.join(readme), encoding='utf-8')
    print('DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
