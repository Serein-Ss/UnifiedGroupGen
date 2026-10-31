"""CSV sessions and optional online/offline W&B loggers."""
import json
import uuid
from pytorch_lightning.loggers import CSVLogger, WandbLogger


def build_loggers(args, output, settings, resume):
    # Lightning's CSV writer overwrites an existing metrics.csv. Keep every resume session.
    version = 'resume_'+uuid.uuid4().hex[:12] if resume else ''
    csv_logger = CSVLogger(save_dir=str(output.parent), name='logs', version=version)
    loggers = [csv_logger]
    if settings.wandb_mode != 'disabled':
        settings.wandb_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output.parent/'run.json'
        metadata = json.loads(metadata_file.read_text(encoding='utf-8')) if metadata_file.exists() else {}
        loggers.append(WandbLogger(project=getattr(args, 'wandb_project', None) or 'UnifiedGroupGen',
                                  name=metadata.get('experiment', output.parent.name),
                                  id=metadata.get('wandb_id'), save_dir=str(settings.wandb_dir),
                                  offline=settings.wandb_mode == 'offline', mode=settings.wandb_mode,
                                  log_model=False))
    return loggers
