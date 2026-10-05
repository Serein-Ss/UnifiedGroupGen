# Full training bundle

The three `full_training.tar.gz.partNNN` files contain **all 102,610 source records** (MP20 45,229; MPTS52 40,476; C2DB 16,905), full grouped joint splits, raw CSVs, conversion/audit manifests, and the original C2DB recovery archive.

Run from the repository root:

```bash
python scripts/unpack_training_data.py
python scripts/verify_training_data.py
```

`manifest.json` binds each archive part, the combined compressed archive and every extracted file to SHA256. Archive size: 104,214,509 bytes; files: 68. Checkpoints and credentials are excluded.

Only JSON metadata paths are rebound to the current checkout. Structure record bytes and property labels are preserved. Original Windows provenance strings in the records remain historical evidence.

For sources, complete counts, installation, training/resume commands and scope, see [REMOTE_TRAINING.md](../docs/REMOTE_TRAINING.md).
