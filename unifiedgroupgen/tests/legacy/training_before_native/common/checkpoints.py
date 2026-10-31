from unifiedgroupgen.common.configuration import resolve_path
import hashlib
import json
import torch

def file_hash(path):
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)

def save_status(output, status):
    path = output.with_suffix('.status.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(path)


def load_pretrained_models(proposal, flow, path):
    """Weights only: accept inference .pt or native Lightning .ckpt, strictly."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'proposal' in saved and 'flow' in saved:
        proposal_weights, flow_weights = saved['proposal'], saved['flow']
    elif 'state_dict' in saved and 'ugg_state' in saved:
        weights = saved['state_dict']
        proposal_weights = {k.removeprefix('proposal.'): v for k, v in weights.items() if k.startswith('proposal.')}
        flow_weights = {k.removeprefix('flow.'): v for k, v in weights.items() if k.startswith('flow.')}
    else:
        raise ValueError('Pretrained weights must be a UnifiedGroupGen inference .pt or native .ckpt')
    proposal.load_state_dict(proposal_weights, strict=True)
    flow.load_state_dict(flow_weights, strict=True)
