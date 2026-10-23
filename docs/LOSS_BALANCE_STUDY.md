# Coordinate-loss normalization and paired regression study

## Objective

The existing flow loss uses the mean squared error of log-metric lattice velocities
and the Cartesian coordinate-velocity error per independent coordinate degree of freedom:

`raw_flow = lattice + coordinates_raw`.

The optional `coordinate_length_scale` is a fixed length in angstrom:

`flow = lattice + coordinates_raw / coordinate_length_scale**2`.

The default is 1.0 and reproduces the old numerical objective. A length of 3.0
converts physical coordinate errors to squared units of a common 3 angstrom length.
This is a fixed unit choice, not a fitted optimum or a claim of equal gradients.
The group compiler, legal projection, prior, backbone and sampling process are unchanged.

History retains `coordinates` as the raw physical error and adds
`coordinates_dimensionless` and `flow_raw`. `flow` is the actual weighted flow
objective. Both training and held-out evaluation use the checkpoint's scale.
Totals from different loss definitions must not be compared directly.

## Gradient diagnostics

`gradient_diagnostics_interval: 0` disables diagnostics (default). Positive values
sample the first, last and every specified training batch. The output
`model.gradients.jsonl` contains raw and weighted component gradient norms,
shared-backbone lattice/coordinate cosine, reconstructed total norm and the
actual pre-clipping total norm. `model.history.jsonl` also records the fraction
of all training batches triggering global clipping.

Diagnostics call `autograd.grad(..., retain_graph=True)`, do not fill `.grad`,
and consume no random draws. They add diagnostic backward work on selected
batches; timing with diagnostics is not a pure throughput benchmark.
Shared-parameter comparisons exclude the two task heads. Zero-gradient cosine
is reported as null. The two modules still share the original global clipping
and scheduler behavior unless the isolated paired protocol states otherwise.

## Local paired protocol

Run `python scripts/loss_balance_study.py --prepare-only` to freeze the protocol,
current source including the composition-fingerprint scaler resource, full repaired data hashes/scalers, and one identical initialization
per dataset. Preparation uses CPU and does not allocate CUDA model memory.

`python -u scripts/loss_balance_study.py --wait-for artifacts/paired_training_20261023`
queues behind the existing acceleration study, then runs only these six ablations:

| Dataset | Before | After |
|---|---|---|
| MP-20 | length scale 1 | length scale 3 angstrom |
| C2DB-51 | length scale 1 | length scale 3 angstrom |
| MP-20+C2DB-51 | length scale 1 | length scale 3 angstrom |

Both arms use the complete repaired TRAIN/VAL files, full main15 `_all` architecture,
seed 42, batch size 64, BF16, four CPU threads, no activation recomputation,
edge chunks 32768 and validation every five epochs. Default budget is 20 epochs.
The learning rate stays at the original initial rate for this short study:
plateau scheduler patience is set above the budget and early stopping is disabled
in both arms. This isolates the objective change from adaptive LR/early stopping.
Diagnostics use interval 100 in both arms. No formal main15 YAML is modified.

The order alternates before/after between datasets. Saved initial weight hashes,
per-epoch batch-order hashes and RNG-state hashes must match within each pair.
Partial interrupted runs are preserved and require review rather than silently
reusing a different training budget. The queue waits for the entire previous
study, including generation evaluation, to finish; a failed dependency stops it.

## Evaluation and interpretation

Use final equal-budget checkpoints. Evaluate full audited TEST losses and,
per source domain, 256 unconditional plus 128 property-conditioned samples,
seed 142, 64 Heun steps, CFG 2 for properties and 0 for unconditional sampling.
Property targets are complete tuples selected deterministically from TRAIN.
The joint model reports MP-20 and C2DB results separately.

Reuse the standard 3D/2D evaluator: group containment, exact group recovery,
orbit regularity, geometry, chemistry, uniqueness, novelty, fingerprints,
coverage and distribution discrepancies. Full same-kind TRAIN is the novelty
reference and full source-specific TEST is the distribution reference.
All failed and unknown attempts remain in the corresponding denominators.
Independent property hit rates and physical stability remain unassessed.

Compare descriptor, raw lattice, raw coordinate and the common diagnostic
`descriptor + lattice + coordinates_raw/9` for BOTH arms. Compare weighted shared
component gradients and clipping frequency. Arm-specific weighted totals alone
cannot establish an improvement.

The report flags >10% increases in final raw component validation loss and
>10 percentage-point drops in validity/exact-group/uniqueness/novelty rates.
These are regression warnings, not significance tests or noninferiority margins.
Single-seed short training can screen obvious regression, not certify final quality.
The report preserves all other quality metrics for review before selecting any
formal loss configuration. It never launches main15 or automatically promotes
an ablation into a formal configuration.

## Outputs

Under `artifacts/loss_balance_20261023/`:

- `protocol.json`: hashes, scales, full configurations, counts, evaluation settings.
- `status.json`: queued/running/failed/complete, active job, process ID.
- `runs/{dataset}/{before,after}/`: checkpoint, loss/gradient history and pairing trace.
- `evaluations/`: complete TEST and generation metrics, per-attempt CSV and PNG.
- `figures/loss_comparison.png`: three datasets, four common-scale loss panels.
- `figures/gradient_comparison.png`: shared gradients and clipping fractions.
- `comparison.json`, `REPORT.md`: matched inputs, warnings and remaining decisions.

All figures created by this protocol are PNG.


## Remote 4090 execution

Use the existing `cgdit` environment (Python >=3.10, PyTorch >=2.2); do not reinstall
`requirements-remote.txt` for this study because that snapshot pins a different
Torch version. Editable installation with `--no-deps` preserves the working runtime.
Required evaluation imports are `smact`, `matminer`, and `matplotlib`.
The training and evaluation settings match the local loss study (four threads).
Do not pass the local `--wait-for` directory on the server.

From the repository root after activating `cgdit`:

```bash
(
set -e

git pull --ff-only
export MKL_THREADING_LAYER=SEQUENTIAL
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONPATH="$PWD/src"
export CUDA_VISIBLE_DEVICES=0
export PYTHONIOENCODING=utf-8

python -m pip install --no-deps --no-build-isolation -e '.[test,evaluation]'
python - <<'PY'
import sys
import torch
import importlib.metadata as metadata
assert sys.version_info >= (3, 10)
assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.is_bf16_supported(), 'BF16 unavailable'
for package in ('torch', 'pyxtal', 'smact', 'matminer', 'matplotlib'):
    print(package, metadata.version(package))
assert metadata.version('pyxtal') == '1.1.3'
print('GPU:', torch.cuda.get_device_name(0))
PY

python -m pytest tests/test_loss_balance.py tests/test_loss_study_archive.py -q

python -u scripts/loss_balance_study.py \
  --output artifacts/loss_balance_4090_seed42 \
  --epochs 20 --count 256 --conditional-count 128 --steps 64 \
  --prepare-only

CUDA_VISIBLE_DEVICES=0 nohup python -u scripts/loss_balance_study.py \
  --output artifacts/loss_balance_4090_seed42 \
  --epochs 20 --count 256 --conditional-count 128 --steps 64 \
  > artifacts/loss_balance_4090_seed42/driver.log 2>&1 &

echo "Driver PID: $!"
)
```

Preparation verifies full repaired main15 TRAIN/VAL gates, hashes source data,
freezes code and the fingerprint scaler, fits identical property scalers, and
saves identical starting weights. It does not start the formal main15 runner.
The exclusive `pipeline.lock` prevents duplicate study controllers.

Expected TRAIN/VAL/TEST counts:

| Dataset | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| MP-20 | 27136 | 9047 | 9046 |
| C2DB-51 | 10143 | 3381 | 3381 |
| Joint | 37279 | 12428 | 12427 |

Progress:

```bash
tail -n 20 -F artifacts/loss_balance_4090_seed42/driver.log
# Ctrl+C leaves the nohup worker running.
watch -n 5 'python -m json.tool artifacts/loss_balance_4090_seed42/status.json'
tail -n 5 -F artifacts/loss_balance_4090_seed42/train_mp20_before.log
```

The sequence is MP20 before/after, C2DB after/before, joint before/after,
then six evaluation jobs (the joint model is evaluated in both source domains).
`state=complete` means the training/evaluation/report stages finished.
The driver then prints the automatic ZIP archive path and SHA256.

## Return the results for local analysis

On completion an evidence archive is automatically written to:

`artifacts/exports/loss_balance_4090_seed42_audit_YYYYMMDD_HHMMSS.zip`

A matching `.zip.sha256` file is written beside it. The ZIP is CRC-checked
and contains `audit_manifest.json` with each included file's size and SHA256.
It includes protocol/scalers/initial weight hashes, source, per-epoch losses and
timings, gradient samples, batch/RNG pairing traces, status and logs, full TEST
results, generation attempt records and CIFs, standard metric CSV/JSON, and PNG.
It excludes checkpoints/initial weights, raw data and large reference caches.
All model weights remain in `runs/` on the server.

Find the finished archive:

```bash
ls -lh artifacts/exports/loss_balance_4090_seed42_audit_*.zip*
```

If packaging needs to be repeated after successful completion:

```bash
python scripts/pack_loss_study.py --study artifacts/loss_balance_4090_seed42
```

For a failed or still-running job, preserve evidence with an explicitly marked
incomplete snapshot rather than treating it as a completed experiment:

```bash
python scripts/pack_loss_study.py \
  --study artifacts/loss_balance_4090_seed42 --allow-incomplete
```

Download the ZIP and SHA256 through the existing server file transfer tool.
Provide the local path for analysis of losses, gradients, constraints and quality.
A short single-seed ablation does not establish final convergence or property/stability performance.


## Read-only live summary

```bash
python scripts/loss_study_status.py --study artifacts/loss_balance_4090_seed42
watch -n 30 'python scripts/loss_study_status.py --study artifacts/loss_balance_4090_seed42'
```

This standard-library-only command reads the protocol, six histories and status
files, diagnostic gradient samples and evaluation receipts. It does not start,
stop or modify training. It shows saved-epoch train/validation/checkpoint time,
raw component losses, both arms on one common loss scale, validation epoch,
weighted shared gradients, clipping fraction, historical allocated/reserved GPU
peaks and current whole-GPU usage. The last three lines of the active child log
help track CPU reference-fingerprint work as well as GPU sampling.

Training ETA uses recent noninitial epoch medians, available validation timings
and the current batch fraction. Pending datasets are scaled from measured MP20
throughput; otherwise the earlier 4090 pilot is the fallback. Sampling ETA uses
current observed seconds/attempt once at least ten attempts are available,
otherwise the earlier 4090 64-step sampling timings. The CPU standard-metric
allowance is 1-3 hours apportioned over unfinished metric jobs, not an observed
ETA or a confidence interval. Queued waiting time is excluded. If the Linux
controller PID no longer exists, ETA represents remaining work after recovery.
The summary preserves the distinction between finished execution and quality.
