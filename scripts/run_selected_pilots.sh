#!/usr/bin/env bash
# Three representative models, five epochs each. Never runs main15 training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONIOENCODING=utf-8
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
pilot_root=${1:?Provide a new output directory under artifacts/runtime_pilots}

python - "$pilot_root" <<'PY'
import json, subprocess, sys
from pathlib import Path
root = Path(sys.argv[1]).resolve()
if not root.is_relative_to((Path.cwd()/'artifacts/runtime_pilots').resolve()) or root.exists():
    raise SystemExit('Use a NEW directory under artifacts/runtime_pilots; nothing was started')
state = json.loads(Path('artifacts/loss_balance_4090_seed42/status.json').read_text())
if state.get('state') != 'complete':
    raise SystemExit('Previous evaluation is not complete; nothing was started')
gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'],
                     capture_output=True, text=True, check=True)
if any(line.strip().isdigit() for line in gpu.stdout.splitlines()):
    raise SystemExit('GPU 0 is busy; no process was stopped and no pilot was started')
PY

mkdir -p "$(dirname "$pilot_root")"
mkdir "$pilot_root"
cp -- "${BASH_SOURCE[0]}" "$pilot_root/launcher.sh"
for setting in mp20_all:256:400000 c2db51_all:768:200000 joint_all:256:400000; do
  IFS=: read -r model batch budget <<< "$setting"
  echo "[START] $model batch=$batch edges=$budget scale=3A epochs=5"
  python scripts/train_timing_pilot.py \
    --model "$model" --threads 4 --batch-size "$batch" --edge-budget "$budget" \
    --coordinate-length-scale 3 --no-checkpoint --edge-chunk-size 32768 \
    --output "$pilot_root/$model" --plan > "$pilot_root/${model}.plan.json"
  python -u scripts/train_timing_pilot.py \
    --model "$model" --threads 4 --batch-size "$batch" --edge-budget "$budget" \
    --coordinate-length-scale 3 --no-checkpoint --edge-chunk-size 32768 \
    --output "$pilot_root/$model" > "$pilot_root/${model}.log" 2>&1
  echo "[DONE] $model"
done

python - "$pilot_root" <<'PY'
import csv, hashlib, json, sys, zipfile
from pathlib import Path
root = Path(sys.argv[1])
reports, rows = {}, []
for model in ('mp20_all', 'c2db51_all', 'joint_all'):
    report = json.loads((root/model/'timing.json').read_text())
    state = json.loads((root/model/'pilot.json').read_text())
    assert state['state'] == 'complete' and report['epochs'] == 5
    assert report['coordinate_length_scale_angstrom'] == 3
    reports[model] = report
    rows.extend({'model': model, **row} for row in report['rows'])
(root/'summary.json').write_text(json.dumps({'models': reports, 'formal15_started': False,
    'scope': 'Three representative models, five epochs each; speed/loss check, not convergence'}, indent=2), encoding='utf-8')
with (root/'summary.csv').open('w', encoding='utf-8', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
archive = Path(str(root)+'.zip')
with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.suffix.lower() != '.pt':
            bundle.write(path, root.name+'/'+path.relative_to(root).as_posix())
    driver = Path(str(root)+'.driver.log')
    if driver.exists():
        bundle.write(driver, root.name+'/driver.log')
with zipfile.ZipFile(archive) as bundle:
    assert bundle.testzip() is None
sha = hashlib.sha256(archive.read_bytes()).hexdigest()
Path(str(archive)+'.sha256').write_text(sha+'  '+archive.name+'\n', encoding='utf-8')
print('[COMPLETE] three five-epoch pilots; no main15 training')
print('[AUDIT PACKAGE]', archive.resolve())
PY
