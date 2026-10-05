"""Count inspected reference constructors without importing PyG/scatter runtimes.

Reference Python files must first be obtained from the author repositories into
--reference-dir. Only module constructors are executed; no forward or training
code runs. This is a parameter audit, not reproduction of the reference models.
"""

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
from torch import nn
from unifiedgroupgen.cli import artifact_path


def constructors(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == '__init__']
            if node.body:
                classes.append(node)
    scope = {'torch': torch, 'nn': nn, 'math': math, 'MAX_ATOMIC_NUM': 100,
             'Any': Any, 'Dict': Dict, 'List': List, 'Optional': Optional}
    exec(compile(ast.Module(body=classes, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    reference = Path(args.reference_dir)
    paths = {'CrystalFlow_decoder': reference/'crystalflow_cspnet.py',
             'CrysVCD_decoder': reference/'crysvcd_cspnet.py',
             'CGDiT_decoder': Path(__file__).resolve().parents[2]/'cgdit/pl_modules/decoder/cspnet.py'}
    overrides = {'CrystalFlow_decoder': {'lattice_dim': 9, 'latent_dim': 256},
                 'CrysVCD_decoder': {'latent_dim': 512},
                 'CGDiT_decoder': {'latent_dim': 256, 'pred_type': True}}
    report = {}
    for name, path in paths.items():
        config = dict(hidden_dim=512, num_layers=6, num_freqs=128, ln=True, **overrides[name])
        model = constructors(path)['CSPNet'](**config)
        report[name] = {'parameters': sum(p.numel() for p in model.parameters()), 'overrides': config,
                        'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if name == 'CrystalFlow_decoder':
            inactive = sum(p.numel() for key, module in model.named_children() if key.startswith('cemb_')
                           for p in module.parameters())
            report[name]['parameters_used_without_cemb'] = report[name]['parameters']-inactive
    conditioner_path = Path(__file__).resolve().parents[2]/'cgdit/pl_modules/cfg_utils/conditional_embedding_utils.py'
    conditioner = constructors(conditioner_path)['ConditioningEncoder'](256,
                            {'formation_energy_per_atom': {'type': 'scalar'}, 'band_gap': {'type': 'scalar'}})
    report['CGDiT_two_scalar_core'] = {'parameters': report['CGDiT_decoder']['parameters']+
                                    sum(p.numel() for p in conditioner.parameters()),
                                    'scope': 'Decoder + two scalar property encoders + null embeddings; excludes optional multimodal/RL/predictor modules'}
    # GPT2's standard n_inner=4h block has 12h^2+13h parameters (two LNs included).
    electronic = ast.parse((reference/'crysvcd_electronic.py').read_text())
    raw = next(ast.literal_eval(n.value) for n in electronic.body if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'RAW_ELECTRON_CONFIG' for t in n.targets))
    for kind in ('ionic', 'alloy'):
        elements = ['PAD', 'START', 'END']+[e for e in raw if e not in ('PAD', 'START', 'END')
                    and (e.endswith('0') == (kind == 'alloy'))]
        h, vocab, count, electronic_dim = 128, len(elements)+21, 21, len(raw['PAD'])
        gpt = 3*(12*h*h+13*h)+2*h+vocab*h+10*h
        embedding = len(elements)*h+count*h+(electronic_dim+1)*h
        head = vocab*(h+1)
        report[f'CrysVCD_{kind}_front'] = {'parameters': gpt+embedding+head, 'elements': len(elements),
                                          'scope': 'Analytical constructor count; no optional property embeddings'}
    report['scope'] = 'Registered parameter counts; no checkpoint load, forward execution, or numerical reproduction.'
    artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
