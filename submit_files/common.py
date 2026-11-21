"""Shared immutable-snapshot and JSON helpers for current flow campaigns."""
import json
from pathlib import Path
import subprocess
import tarfile

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)

def snapshot(root, folder, revision):
    destination = folder/'source'
    destination.mkdir()
    archive = subprocess.Popen(['git', 'archive', revision, 'unifiedgroupgen', 'configs', 'scripts', 'submit_files'],
                               cwd=root, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout, mode='r|') as stream:
        stream.extractall(destination)
    if archive.wait():
        raise RuntimeError('Cannot freeze tracked source')
    return destination
