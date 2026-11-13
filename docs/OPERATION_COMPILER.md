# Operation-compiled joint generation

Current theory: [THEORY.md](THEORY.md), joint-theory-2026-11-12. The default task is noise-to-crystal generation without a target property. Structure type and group operations are hard specifications, not property conditions. An unspecified group is sampled externally from TRAIN frequencies within the structure type; no learned type/group head is planned. New task_* checkpoints include a TRAIN-frequency prior for no-group-input sampling; legacy checkpoints without it still need an explicit spec. Only target properties enter the property-conditioning channel; CSP composition is a separate hard task input.

Numbered catalog lookup and cached constraints are the normal engineering route. This opt-in compiler derives orbit charts from operations and returns the same constraint contract; it is not rerun at every denoising step. Standard table labels, computed chart labels and arbitrary coordinate settings must not be conflated.

This is an opt-in extension of `joint_symmetry_v1`. Existing experiments, default
model parameters and old checkpoint keys are unchanged. Enable both
`model.operation_compiler` and `model.constraint_layers`, or select
`operation_mp20`, `operation_c2db51`, or `operation_joint`.

## What is computed

`group_constraints/operations.py` accepts nonmagnetic spatial generators, closes
them exactly with rational arithmetic modulo the reference translation lattice,
and computes fixed loci `(R-I)x=n-t`. RREF and Hermite normal form identify affine
components modulo lattice translations. Intersections produce the stabilizer
strata; group action identifies equivalent components and expands complete orbit
charts. No special-position/Wyckoff row is used by this computation. The optional
numbered adapter reads only the general-position operations from PyXtal.

The current domain is space, layer and plane families in an adapted frame:
periodic coordinates first, fractional periodic coordinates, real nonperiodic
coordinates. Input coefficients must be rational with denominator <=96; closure
is bounded at 384 operations. Plane operations preserve z=0 and have identity
normal action. Internally produced orbit names `o000`, ... are not Wyckoff letters.
They are canonical for a fixed operation setting, not invariant to arbitrary
changes of origin or lattice basis. Lower-dimensional strata are included;
chart boundaries can still cause coincident atoms and must pass final validation.

`compile_operations(operations, kind)` returns the existing `SymmetrySpec` contract
with number 0, computed templates and closed operations. The existing invariant
SPD metric and hard coordinate decoders work on this spec. The required group is
contained in the output symmetry; an exact maximal group is not implied.

## Network and probability process

`pl_modules/constraint_layers.py` derives two typed feature fields after each
attention block: scalar fields tied across an orbit, and Cartesian vector fields
projected through the current orbit tangent basis. The latter solves at most a
3x3 system per orbit, retaining gradients through the current lattice. Packed
vector components remain available to the next proposal block and coordinate
head. An atomwise proposal residual is also retained: fixed-position orbits have
zero tangent space, yet their public-state teacher bridges still contain unequal
offsets and normal residuals. Removing this channel makes ambient anchor matching
inexpressive. Scalar pooling alone therefore cannot replace the public context.

Typed fields satisfy their constraints. Packed proposal contexts and raw public
anchor heads are not claimed to be globally equivariant. Physical structures
remain hard-decoded from the public continuous state. In particular, the
derivative of a lattice-dependent decoder includes its lattice dependence; no
unsupported assertion that the raw drift is tangent is made.

`pl_data/operation_path.py` matches existing orbit endpoints to computed charts,
reorders atoms, and re-encodes the physical lattice in memory. Full periodic set
matching is required before accepting a conversion. Dataset files are not edited.
The standard-setting dataset adapter searches neighboring periodic lifts; a
nonmatching external setting is rejected rather than silently distorted.

The existing conditional PDMP, rate-weighted loss and event-aware Heun sampler
are reused. Birth, split, delete, element, replace and swap use the same complete
orbit kernels in training and inference. Geometry and chemistry condition the
joint action logits, while the target operation group stays fixed. N changes by
orbit multiplicities. END, FAIL and finite-epsilon failures retain their original
meaning; failed attempts are never silently converted into valid outputs.

## Usage

Compile a numbered adapter, reading operations only:

```powershell
python -m unifiedgroupgen.joint compile --kind space --group 225 --output intermediate/fcc_operations.json
```

Or supply a request JSON; no templates or group number are required:

```json
{"kind":"plane","generators":[{"rotation":[[-1,0,0],[0,1,0],[0,0,1]],"translation":[0,0,0]}]}
```

```powershell
python -m unifiedgroupgen.joint compile --symmetry request.json --output intermediate/computed.json
python -m unifiedgroupgen.joint smoke --operation-compiler --records 2 --steps 4 --device cpu --output intermediate/checks/operation_compiler_v1/cpu/model.pt
python -m unifiedgroupgen.joint sample --checkpoint intermediate/checks/operation_compiler_v1/cpu/model.pt --symmetry request.json --count 1 --steps 32 --output intermediate/checks/operation_compiler_v1/samples.jsonl
```

For normal audited training, first create the matching data audit, then use the
existing Hydra entry:

```powershell
python -m unifiedgroupgen.joint audit --experiment operation_joint
python -m unifiedgroupgen.run experiment=operation_joint
```

Explicit operator JSON can also be embedded in a training record's `symmetry`
with `public_state`; its coordinates/elements must follow the computed atom order.
The CLI persists the full compiled spec with every operation-mode result. A
numbered request can additionally use `--exact-group`; an operation-only request
checks containment rather than inventing an exact group number.

## Scope and validation

Tests cross-check every catalog orbit for all 230/80/17 families, including actual
generic coordinate-set reconstruction, and exercise invalid requests, redundant
generators, field covariance/gradients, all legal edits, real MP-20/C2DB paths,
three-family event sampling and native checkpoint resume. These are mathematical
and execution checks, not trained material-quality or DFT-stability results.

Run the focused implementation checks with:

```powershell
python -m pytest unifiedgroupgen/tests/unit/test_operation_compiler.py -q
```

This prioritized implementation does not add a learned mixture over completed
groups, operation exclusion sampling, dynamic group changes, automatic magnetic
orbit compilation or a universal group catalog. Unsupported exclusions/spin
requests fail explicitly. Existing explicitly templated magnetic paths remain
available with the compiler flag off; magnetic validation is outside this change.

## Verified local run (2026-11-11)

The combined, deduplicated JUnit ledger contains 920 passed cases and 3 skipped
CPU-bf16 cases (CUDA mixed precision is used instead), with no unresolved failure.
This includes 356 new checks and the existing regression coverage. An initial
integration failure detected a changing package-source fingerprint; its frozen
retry passed. The final residual-channel correction was followed by focused
checks and 33 legacy generation-parity checks.

All 2,280 catalog orbit types passed computed-chart coordinate-set reconstruction.
Another 128 real MP-20/C2DB train/validation endpoints passed complete conditional
paths, with maximum operation residual 4.91e-15 in float64. All six dataset hashes
were rechecked after testing. Formal data audits and the training gates pass for
all three new experiment selectors, covering 62,134 input records.

Final CPU and CUDA short runs each completed four optimizer steps and validation;
native checkpoint continuation is covered by the focused tests. These weights
are smoke artifacts: all three bounded learned-sampling trials exhausted their
eight-event budget. Successful three-family oracle sampling validates the process,
not learned material quality. Pre-correction smoke weights are retained only under
`intermediate/checks/operation_compiler_v1/superseded/`.

The full evidence and source hashes are in
`intermediate/checks/operation_compiler_v1/validation_report.json`; raw JUnit
reports and `workflow.json` are beside it. Rebuild the report with
`python intermediate/checks/operation_compiler_v1/build_report.py` after rerunning
the recorded checks.

默认“条件生成”只指目标性质；类型／群／操作是硬规范，未指定群按TRAIN频率外部采样，不学习群概率。新task_*入口已接入外部初始化，见[TASK_DEFINITIONS.md](TASK_DEFINITIONS.md)。
