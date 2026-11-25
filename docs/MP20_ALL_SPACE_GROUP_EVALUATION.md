# MP-20 backbone evaluation across all space groups

Use the best checkpoints from the frozen MP-20 backbone training campaign. The launcher reads the campaign's cache audits, verifies TRAIN and TEST cache hashes, and records the checkpoint hashes. The workers import the campaign's frozen code.

## Protocol

| Evaluation | Workload per backbone | Outputs |
| --- | --- | --- |
| CSP training-set reconstruction | 1,024 shared cases, 10 candidates per case | Match@1/5/10 from prefixes of the same ledger |
| DNG free generation | 1,024 attempts | Validity, uniqueness, novelty against TRAIN and coverage against TEST |
| CSP specified symmetry | 230 groups × 32 attempts × two input modes | Inventory accuracy, operation retention, validity, exact group, strict supergroup and reference matching where available |
| DNG specified symmetry | 230 groups × 32 attempts × two input modes | Operation retention, validity, exact group and strict supergroup |

The specified inputs are the space-group number and the equivalent complete operation set. The same group, inventory and seed schedule are used in both input modes and on both backbones. Group scoring checks the exported and reloaded CIF. The denominator includes every attempted sample. A valid strict supergroup passes all-operation retention. Terminal contact repair is disabled.

For CSP with a specified group, one fixed TRAIN reference is selected per observed group. Its exact per-cell inventory is used for all 32 attempts. For a group absent from TRAIN, the probe uses Si in the smallest permitted Wyckoff orbit. These probes measure inventory and symmetry retention. Structure matching is restricted to the groups with a real TRAIN reference. Keep the reference and probe results separate.

The default configuration reserves 4 jobs for the main workload and 40 jobs for the specified symmetry workload. The 230 groups are split into five shards of 46 for each backbone, task and input mode.

## Run on 5090

Run from the server project root with the existing project Python environment. The root checkout and frozen training source can stay at the training revision. You can extract the launcher, configuration and batch script into an evaluation-specific directory and pass the launcher's absolute path.

```bash
python -B submit_files/mp20_backbone_evaluation.py submit
```

The launcher writes into the campaign's `evaluation/mp20_train1024_k10_dng1024_space230_x32_s42` directory. The verified plan and accepted scheduler job IDs are stored in `evaluation_plan.json` and `submission.jsonl`. A repeat call with the same launcher and configuration submits entries missing from that ledger. Each worker uses an exclusive results directory.

After all jobs finish, run the launcher with `collect --plan <absolute-evaluation-plan-path>`. It verifies all shards and writes `all_groups_summary.json` with per-group and pooled results. CSP training reconstruction and DNG free-generation metrics remain in their own job result directories.

