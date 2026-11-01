"""Export completed HPC results without importing PyTorch or recalculating metrics."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

DEFAULT_MODELS = ('mp20_u', 'c2db51_u', 'joint_u', 'mp20_all', 'c2db51_all', 'joint_all')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def export(manifest, output, models=DEFAULT_MODELS, cifs_per_task=5):
    if cifs_per_task < -1:
        raise ValueError('cifs_per_task must be -1 (all), 0, or a positive integer')
    campaign = read(manifest)
    folder = Path(campaign['folder'])
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(set(models)) != len(models):
        raise ValueError('Select each model only once')
    selected = []
    for model in models:
        if Path(model).name != model or any(c in model for c in '/\\:') or model in ('.', '..'):
            raise ValueError('Invalid model name')
        phase = 'legacy' if model in campaign['legacy_models'] else 'new'
        phase_file = folder/'state'/f'{phase}_{model}.json'
        evaluation = folder/'evaluation'/model
        pipeline = read(evaluation/'pipeline.json')
        if read(phase_file).get('state') != 'complete' or pipeline.get('state') != 'complete':
            raise ValueError('Evaluation is not complete: '+model)
        expected = {model+'/'+job['name'] for job in pipeline['identity']['jobs']}
        if ({entry['name'] for entry in pipeline['jobs']} != expected
                or any(entry.get('state') != 'complete' for entry in pipeline['jobs'])):
            raise ValueError('Incomplete task receipts: '+model)
        selected.append((model, phase_file, evaluation, pipeline))
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix(output.suffix+'.partial')
    inventory = {'campaign': campaign['name'], 'source_revision': campaign['revision'],
                 'models': list(models), 'cifs_per_task': cifs_per_task, 'tasks': [], 'files': {}}
    tables, cif_rows = [], []
    allowed = {'.json', '.jsonl', '.csv', '.png', '.cif', '.log', '.yaml', '.yml', '.txt'}
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        def add_bytes(name, content):
            archive.writestr(name, content)
            inventory['files'][name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}

        def add_file(path, name, expected=None):
            value = hashlib.sha256(); size = 0
            with path.open('rb') as source, archive.open(name, 'w', force_zip64=True) as target:
                for block in iter(lambda: source.read(1024*1024), b''):
                    value.update(block); size += len(block); target.write(block)
            if expected and value.hexdigest() != expected:
                raise ValueError('Saved output missing or modified: '+str(path))
            inventory['files'][name] = {'sha256': value.hexdigest(), 'bytes': size}

        def add_tree(root, prefix, expected=None):
            root = root.resolve()
            hashes = {}
            for name, value in (expected or {}).items():
                path = (root/name).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError('Missing or external receipt output: '+str(path))
                hashes[path.relative_to(root).as_posix()] = value
            for path in sorted(root.rglob('*')):
                if path.is_file() and (path.suffix in allowed or path.relative_to(root).as_posix() in hashes):
                    relative = path.relative_to(root).as_posix()
                    add_file(path, prefix+'/'+relative, hashes.get(relative))

        add_file(Path(manifest), 'campaign/campaign.json')
        for name in ('legacy_catalog.yaml', 'new_catalog.yaml', 'legacy_budget.json', 'profile.json'):
            if (folder/name).is_file():
                add_file(folder/name, 'campaign/'+name)
        for model, phase_file, evaluation, pipeline in selected:
            print('EXPORT', model, flush=True)
            add_file(phase_file, f'models/{model}/phase.json')
            add_tree(evaluation, f'models/{model}/campaign_evaluation')
            with (evaluation/'summary.csv').open(encoding='utf-8-sig', newline='') as stream:
                tables.extend(csv.DictReader(stream))
            for entry in pipeline['jobs']:
                task = entry['name'].split('/', 1)[1]
                if '/' in task or '\\' in task or ':' in task or task in ('.', '..'):
                    raise ValueError('Invalid task name')
                prefix = f'models/{model}/tasks/{task}'
                generated = Path(entry['directory'])
                receipt = entry.get('generation', entry)
                add_tree(generated, prefix+'/generation', receipt['outputs'])
                item = {'model': model, 'task': task, 'generation': prefix+'/generation'}
                metrics = entry.get('metrics')
                if metrics:
                    directory = Path(metrics['metric_directory'])
                    add_tree(directory, prefix+'/metrics', metrics['outputs'])
                    item['metrics'] = prefix+'/metrics'
                ledger = generated/'samples.jsonl'
                if ledger.is_file():
                    attempted = successful = exported = 0
                    indices = set()
                    with ledger.open(encoding='utf-8-sig') as stream:
                        for line in stream:
                            if not line.strip(): continue
                            row = json.loads(line); index = int(row['index'])
                            if index in indices or index < 0:
                                raise ValueError('Invalid or duplicate sample index: '+str(ledger))
                            indices.add(index); attempted += 1
                            if row['status'] != 'generated': continue
                            successful += 1
                            if 'cif' in row:
                                content = row['cif'].encode('utf-8')
                            else:
                                raise ValueError('Completed campaign sample lacks portable inline CIF: '+str(ledger))
                            checksum = hashlib.sha256(content).hexdigest()
                            if not content.strip() or row.get('cif_sha256', checksum) != checksum:
                                raise ValueError('CIF hash mismatch: '+str(ledger)+' index='+str(index))
                            if cifs_per_task == -1 or exported < cifs_per_task:
                                name = prefix+f'/cifs/sample_{index:07d}.cif'
                                add_bytes(name, content); exported += 1
                                cif_rows.append({'model': model, 'task': task, 'index': index,
                                    'case_id': row.get('case_id', row.get('reference_id')),
                                    'candidate': row.get('candidate'), 'cif': name, 'cif_sha256': checksum,
                                    'metrics_csv': item.get('metrics', '')+'/samples.csv'})
                    item.update(attempted=attempted, generated=successful, failed=attempted-successful,
                                exported_cifs=exported, samples=item['generation']+'/samples.jsonl')
                inventory['tasks'].append(item)
        for name, rows in (('summary.csv', tables), ('cif_index.csv', cif_rows)):
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}) or ['model'])
            writer.writeheader(); writer.writerows(rows)
            add_bytes(name, stream.getvalue().encode('utf-8-sig'))
        add_bytes('README.txt', (
            '本包只导出已完成的生成与评估结果，不重新计算指标。\n'
            'summary.csv：所选模型各任务的原始汇总指标，保留分母和缺失值。\n'
            'models/<模型>/tasks/<任务>/generation/samples.jsonl：完整尝试记录，含失败尝试和成功结构。\n'
            '同任务 metrics/samples.csv：逐结构指标；按模型、任务、index 对应。CSP 还可结合 case_id 和 candidate。\n'
            'cif_index.csv：独立 CIF 与原始记录的对应表；默认每任务前5个成功结构，不代表质量排名。\n'
            '所有成功结构的精确 CIF 均已校验；未导出独立 CIF 的结构仍在完整记录中。\n'
            'manifest.json：批次、原代码版本、路径映射、任务计数及压缩包内文件 SHA256。\n'
            '原报告内服务器绝对路径作为历史证据保留；本地查找请使用 manifest.json 的相对路径。\n'
            '性质预测沿用原报告适用域与限制；数据表示指标完成，不等于DFT或实验验证。\n'
        ).encode('utf-8'))
        archive.writestr('manifest.json', json.dumps(inventory, ensure_ascii=False, indent=2)+'\n')
    partial.replace(output)
    checksum = digest(output)
    output.with_suffix(output.suffix+'.sha256').write_text(checksum+'  '+output.name+'\n', encoding='utf-8')
    print('ZIP:', output)
    print('SIZE MiB:', round(output.stat().st_size/1024**2, 2))
    print('SHA256:', checksum)
    return inventory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--models', nargs='+', default=DEFAULT_MODELS)
    parser.add_argument('--cifs-per-task', type=int, default=5, help='-1: all successful CIFs; 0: original records only')
    args = parser.parse_args()
    export(args.campaign, args.output, args.models, args.cifs_per_task)
