#!/usr/bin/env bash
# Twelve runtime benchmarks only. Never launches formal training.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONIOENCODING=utf-8
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0

python -m pip install --no-deps --no-build-isolation -e '.[test]'
python -m pip install --no-deps 'pyxtal==1.1.3'
python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.is_bf16_supported(); print(torch.__version__, torch.cuda.get_device_name(0))"
nvidia-smi
if nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]*$'; then
  echo 'GPU 0已有计算进程，请确认其他训练已退出后再测速。没有自动停止任何进程。'
  exit 1
fi

python scripts/unpack_training_data.py --bundle datasets/source_integrity_v1
python scripts/prepare_main15.py
python scripts/prepare_main15.py --verify-only

export UGG_TUNE_DIR="artifacts/benchmarks/4090_tuning_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$UGG_TUNE_DIR"
printf 'model\tmode\tchunk\texit_code\n' > "$UGG_TUNE_DIR/status.tsv"

for tune_model in mp20_all c2db51_all joint_all; do
  for tune_setting in checkpoint:4096 no-checkpoint:4096 no-checkpoint:16384 no-checkpoint:32768; do
    tune_mode=${tune_setting%:*}
    tune_chunk=${tune_setting#*:}
    tune_path="$UGG_TUNE_DIR/${tune_model}_${tune_mode}_${tune_chunk}"
    echo "[START] $tune_model $tune_setting"
    if python -u scripts/benchmark.py \
      --config "configs/main15/${tune_model}.yaml" --device cuda \
      --seed 42 --batch-size 64 --threads 4 \
      --selection random --batches 6 --warmup 6 --steps 24 \
      "--$tune_mode" --edge-chunk-size "$tune_chunk" \
      --output "$tune_path.json" > "$tune_path.log" 2>&1; then
      tune_exit=0
      echo '[DONE]'
    else
      tune_exit=$?
      echo "[FAILED] exit=$tune_exit; log=$tune_path.log"
      tail -n 5 "$tune_path.log"
    fi
    printf '%s\t%s\t%s\t%s\n' "$tune_model" "$tune_mode" "$tune_chunk" "$tune_exit" >> "$UGG_TUNE_DIR/status.tsv"
  done
done

python - <<'PY'
import csv, json, os
from pathlib import Path
root = Path(os.environ['UGG_TUNE_DIR'])
with (root/'status.tsv').open(encoding='utf-8') as stream:
    statuses = list(csv.DictReader(stream, delimiter='\t'))
rows, best = [], {}
print('\n模型 | 重计算 | 边块 | 状态 | 平均秒/步 | 中位秒/步 | 样本/秒 | 分配GiB | 保留GiB | 加速比')
for model in ('mp20_all', 'c2db51_all', 'joint_all'):
    entries = [s for s in statuses if s['model'] == model]
    reports = {}
    for s in entries:
        path = root/f"{model}_{s['mode']}_{s['chunk']}.json"
        if s['exit_code'] == '0' and path.exists():
            reports[(s['mode'], int(s['chunk']))] = json.loads(path.read_text(encoding='utf-8'))
    values = list(reports.values())
    if values:
        keys = ('record_ids','atoms','batch_sizes','config_sha256','fitted_config_sha256',
                'train_split_sha256','val_split_sha256','seed','precision','total_parameters')
        assert all(all(r[k] == values[0][k] for k in keys) for r in values), f'{model}: 配对输入不一致'
    baseline = reports.get(('checkpoint', 4096))
    for s in entries:
        r = reports.get((s['mode'], int(s['chunk'])))
        row = {'model': model, 'recompute': s['mode'] == 'checkpoint',
               'chunk': int(s['chunk']), 'status': 'OK' if r else 'FAILED',
               'mean_s': None, 'median_s': None, 'samples_s': None,
               'allocated_GiB': None, 'reserved_GiB': None, 'speedup': None,
               'exit_code': int(s['exit_code'])}
        if r:
            row.update(mean_s=r['seconds_per_step'], median_s=r['median_seconds_per_step'],
                       samples_s=r['records_per_second'], allocated_GiB=r['peak_cuda_bytes']/2**30,
                       reserved_GiB=r['peak_cuda_reserved_bytes']/2**30,
                       speedup=r['records_per_second']/baseline['records_per_second'] if baseline else None)
        rows.append(row)
        fields = ('model','recompute','chunk','status','mean_s','median_s','samples_s',
                  'allocated_GiB','reserved_GiB','speedup')
        print(' | '.join(f'{row[k]:.4f}' if isinstance(row[k], float) else
                         ('NA' if row[k] is None else str(row[k])) for k in fields))
    if values:
        winner = max(values, key=lambda r: r['records_per_second'])
        best[model] = {k: winner[k] for k in ('checkpoint_layers','edge_chunk_size',
                     'records_per_second','median_seconds_per_step','peak_cuda_bytes','peak_cuda_reserved_bytes')}
        print(f"[最快候选] {model}: 重计算={winner['checkpoint_layers']}, 边块={winner['edge_chunk_size']}")
    else:
        best[model] = None
        print(f'[无有效候选] {model}: 检查失败日志')
with (root/'summary.csv').open('w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
(root/'summary.json').write_text(json.dumps({'rows':rows, 'fastest_candidates':best,
    'scope':'Short training-step timing only; no model weights saved; no formal training started; high-edge and full-epoch checks pending'},
    indent=2, allow_nan=False), encoding='utf-8')
print('\n汇总目录:', root.resolve())
print('生成 summary.csv、summary.json；失败日志及全部12项状态均保留。')
print('最快候选还需高边数批次和整轮复验，本脚本在这里结束。')
PY
