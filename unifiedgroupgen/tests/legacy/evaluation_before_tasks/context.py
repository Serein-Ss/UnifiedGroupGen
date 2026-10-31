"""Inputs shared by per-sample checks and population aggregation."""
from dataclasses import dataclass
from pathlib import Path
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import ReferenceBank


@dataclass(frozen=True)
class EvaluationContext:
    task: str
    kind: str
    dataset: str | None
    request: dict
    refs: ReferenceBank
    train: ReferenceBank
    use_fingerprints: bool
    reidentify: bool
    novelty_files: tuple | list
    samples: Path
