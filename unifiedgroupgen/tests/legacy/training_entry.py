"""Archived training adapter; new public training uses unifiedgroupgen.run."""
from pathlib import Path
import json
from omegaconf import OmegaConf
from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path
from unifiedgroupgen.common.configuration import RunSettings
from unifiedgroupgen.run import training_output
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.conditional.properties import attach_dataset_domains
from unifiedgroupgen.pl_data.encoding import read_jsonl
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.run import fit_training_config
from unifiedgroupgen.tests.preflight import check_training_gate

def train_args(args, config=None, resolved_config=None):
    """Common Python entry; no training logic is duplicated in command adapters."""
    from unifiedgroupgen.run import run_lightning

    root = project_root()
    if config is None:
        config = OmegaConf.to_container(OmegaConf.load(resolve_path(args.config, root)), resolve=True)
    if getattr(args, 'trainer', 'lightning') == 'legacy':
        # Only historical frozen studies use this numerical reference backend.
        from unifiedgroupgen.tests.reference_training import run_training
        backend = run_training
    else:
        backend = None
    prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
    paths = {split: [resolve_path(p, root) for p in (getattr(args, split, None)
             or config.get(split+'_files') or [f'{prepared}/{split}.jsonl'])] for split in ('train', 'val')}
    check_training_gate(config, paths['train'], paths['val'], root)
    records = attach_dataset_domains([r for p in paths['train'] for r in read_jsonl(p)], config)
    validation = attach_dataset_domains([r for p in paths['val'] for r in read_jsonl(p)], config)
    if not records or not validation:
        raise ValueError('Training and validation records must both be nonempty')
    if {(r['dataset'], r['id']) for r in records} & {(r['dataset'], r['id']) for r in validation}:
        raise ValueError('Training/validation identifier overlap')
    config = fit_training_config(config, records)
    args.resume = resolve_path(args.resume, root) if args.resume else None
    if getattr(args, 'initialize_from', None):
        args.initialize_from = resolve_path(args.initialize_from, root)
    proposal, flow = build_models(config, args.device if backend else 'cpu')
    if backend:
        if getattr(args, 'initialize_from', None):
            raise ValueError('The historical reference trainer cannot initialize from pretrained weights')
        output = resolve_path(args.output or f"artifacts/{config['dataset']}/model.pt", root)
        output.parent.mkdir(parents=True, exist_ok=True)
        return backend(args, config, records, validation, proposal, flow, paths['train'], paths['val'], output)
    settings = RunSettings.load(root, getattr(args, 'wandb_mode', None))
    output = training_output(settings, args.experiment or config.get('experiment_id', 'training'), args.output, args.resume)
    if not args.resume:
        meta = output.parent/'run.json'
        metadata = json.loads(meta.read_text(encoding='utf-8'))
        metadata.update(seed=args.seed)
        if getattr(args, 'config', None):
            metadata['source_config_sha256'] = file_hash(resolve_path(args.config, root))
        meta.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    if resolved_config is not None:
        # Preserve launch settings separately for each resume, too.
        from datetime import datetime
        conf = output.parent/'.hydra'
        conf.mkdir(exist_ok=True)
        filename = 'config.yaml' if not args.resume else 'resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.yaml'
        OmegaConf.save(resolved_config, conf/filename, resolve=True)
        OmegaConf.save(OmegaConf.create(config), conf/'training_config.yaml')
    print(json.dumps({'event': 'run_directory', 'trainer': 'lightning', 'output': str(output),
                      'wandb_mode': settings.wandb_mode}), flush=True)
    return run_lightning(args, config, records, validation, proposal, flow,
                         paths['train'], paths['val'], output, settings)
