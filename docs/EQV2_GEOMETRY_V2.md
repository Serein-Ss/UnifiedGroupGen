# EquiformerV2 training on MP-20 and MP-20+C2DB-51

The four `*_geometry_v2` experiments start with fresh weights and use the same joint orbit/geometry objective. Compare the MP-20 portion of the two test sets directly, and report the C2DB-51 portion of the mixed runs separately. The fixed FP32 calibration candidates are 8 for MP-20 and 4 for the mixed dataset. These batches differ, so report optimizer steps, examples processed and GPU time with the evaluation.

For blind CSP, the checkpoint stores the exact composition histograms fitted on TRAIN. At generation time, the current composition score gets 75% of the probability mass and the nearest-composition vote gets 25%. Neighbors are selected separately for each group kind and infeasible groups are filtered out. The features encode square-root elemental fractions, cell atom count and number of elements. The neighborhood uses up to 128 training observations, including duplicate compositions, with distance-weighted votes. An explicit group request bypasses this ranking. Use `--csp-group-policy composition_ranked` for the original ordering with the same checkpoint.

All four experiments default to bounded terminal contact repair. It moves only the internal coordinates of the current Wyckoff orbits and rechecks the requested operations after CIF readback. It can be disabled with `--no-repair-short-contacts` for paired evaluation. The optional differentiable short-contact loss stays disabled in these runs.

## Local evidence

| Check | Original | New |
| --- | ---: | ---: |
| MP-20 validation reference group coverage, top 1 / 20 | 105 / 314 of 384 | 145 / 341 of 384 |
| C2DB-51 validation reference group coverage, top 1 / 20 | 62 / 328 of 384 | 92 / 348 of 384 |
| MP-20 validation Match@1 on 16 cases with changed top-ranked groups | 2/16 | 6/16 |
| MP-20 archived DNG candidates accepted after CIF checks | 971/1,024 | 1,012/1,024 |
| MP-20 archived training-set CSP Match@1 after contact repair | 455/1,024 | 464/1,024 |

The generation pilot used the same archived EquiformerV2 CSP weights, seeds and 32 integration-step setting in both arms. The selected 16 cases measure a change on cases affected by ranking. The contact-repair comparison reused the exact archived candidate endpoints. The 24-step paired optimization pilot showed inconsistent short-contact loss changes across tasks and datasets. These checks do not estimate full test set performance after retraining.
