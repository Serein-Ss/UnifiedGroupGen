"""Local path/logging settings and dated training directories."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import uuid


@dataclass(frozen=True)
class RunSettings:
    project_root: Path
    jobs: Path
    wandb_dir: Path
    wandb_mode: str

    @classmethod
    def load(cls, root, mode=None):
        from dotenv import dotenv_values
        root = Path(root).resolve()
        values = {**dotenv_values(root/'.env'), **os.environ}
        flag = str(values.get('USE_WANDB_LOGGING', '0'))
        if flag not in ('0', '1'):
            raise ValueError('USE_WANDB_LOGGING must be 1 (online) or 0 (offline)')
        def directory(key, default):
            value = Path(values.get(key) or default).expanduser()
            return (value if value.is_absolute() else root/value).resolve()
        return cls(root, directory('HYDRA_JOBS', 'outputs'), directory('WANDB_DIR', 'wandb'),
                   mode or ('online' if flag == '1' else 'offline'))


def training_output(settings, experiment, output=None, resume=None):
    """Explicit historical outputs stay explicit; new default runs get a real start date."""
    if not experiment or Path(experiment).name != experiment or experiment in ('.', '..') or ':' in experiment or '\\' in experiment:
        raise ValueError('Experiment must be a single directory name')
    if output:
        output = Path(output).expanduser()
        output = (output if output.is_absolute() else settings.project_root/output).resolve()
    elif resume:
        output = Path(resume).resolve().parent/'model.pt'
    else:
        started = datetime.now(timezone(timedelta(hours=8)))
        directory = settings.jobs/experiment/started.strftime('%y-%m-%d-%H-%M-%S')
        directory.mkdir(parents=True, exist_ok=False)
        output = directory/'model.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.parent/'run.json'
    if resume and not metadata.exists():
        raise ValueError('Resume needs a Lightning run.json; use --initialize-from to start from legacy weights')
    if not metadata.exists():
        started = datetime.now(timezone(timedelta(hours=8)))
        metadata.write_text(json.dumps({'experiment': experiment, 'started_at': started.isoformat(),
                            'directory_date_kind': 'training_run_start', 'trainer': 'pytorch-lightning',
                            'wandb_mode': settings.wandb_mode, 'wandb_id': uuid.uuid4().hex,
                            'output': str(output)}, indent=2)+'\n', encoding='utf-8')
    elif not resume:
        raise FileExistsError('Training directory already has run.json; use --resume: '+str(output.parent))
    return output
