"""Explicit real-weight integration check; never run by the default offline unit suite."""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.common.structures import load_structure, write_json
from unifiedgroupgen.common.predictors.matgl import MatGLPredictor, FIDELITIES
from unifiedgroupgen.common.predictors.structures import predictor_structure
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor


def check(cache, output):
    from unifiedgroupgen.evaluate import execute
    from omegaconf import OmegaConf
    root = project_root()
    predictor = MatGLPredictor(cache)
    cases, differences = {}, []
    for title, directory, kind in [('TiO2', 'mp20_walkthrough', 'space'), ('MoS2', 'c2db_walkthrough', 'layer')]:
        record = json.loads((root/'examples'/directory/'record.json').read_text(encoding='utf-8'))
        structure = load_structure(record, root)
        result = predictor.predict_structures([structure, structure], kind=kind, batch_size=2)
        assert all(not row['prediction_error'] for row in result), result
        graph_results = {}
        for name, model in predictor.models.items():
            converted = predictor_structure(structure, kind, model.model.cutoff)
            kwargs = {'state_feats': torch.tensor([0])} if name == 'band_gap' else {}
            with torch.no_grad(): native = float(model.predict_structure(converted, **kwargs))
            np.testing.assert_allclose([r[name] for r in result], native, rtol=1e-5, atol=1e-5)
            differences.extend(abs(r[name]-native) for r in result)
            graph = predictor.graph(structure, name, kind)
            graph_results[name] = {'nodes': graph.num_nodes(), 'edges': graph.num_edges(),
                'normal_image_edges': int((graph.edata['pbc_offset'][:, 2] != 0).sum()), 'native_prediction': native}
        compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        expanded = compiled.expand(torch.tensor(record['state'], dtype=torch.float64))
        from_graph = predictor.predict_graph(expanded, kind=kind)
        for name in predictor.models:
            np.testing.assert_allclose(from_graph[name], result[0][name], rtol=1e-4, atol=1e-4)
        if kind == 'layer':
            first = predictor_structure(structure, kind, 5.)
            from unifiedgroupgen.evaluation.utils import layer_structure
            other_vacuum = layer_structure(first, height=first.lattice.c+30.)
            other = predictor.predict_structures([other_vacuum], kind=kind)[0]
            for name in predictor.models:
                np.testing.assert_allclose(other[name], result[0][name], rtol=1e-5, atol=1e-5)
        fidelities = {}
        for fidelity, index in FIDELITIES.items():
            head = MatGLPredictor(cache, properties=['band_gap'], band_gap_fidelity=fidelity)
            value = head.predict_structures([structure], kind=kind)[0]['band_gap']
            converted = predictor_structure(structure, kind, head.models['band_gap'].model.cutoff)
            native = float(head.models['band_gap'].predict_structure(converted, state_feats=torch.tensor([index])))
            np.testing.assert_allclose(value, native, rtol=1e-5, atol=1e-5)
            fidelities[fidelity] = value
        job = output/title
        job.mkdir(parents=True, exist_ok=True)
        cif = structure.to(fmt='cif')
        samples, reference = job/'samples.jsonl', job/'reference.jsonl'
        # These are real example fixtures, not samples from a newly evaluated generator.
        samples.write_text(json.dumps({'index': 5, 'status': 'generated', 'cif': cif,
            'source': 'real_dataset_example_not_new_generation'})+'\n', encoding='utf-8')
        reference.write_text(json.dumps(record)+'\n', encoding='utf-8')
        properties=OmegaConf.load(root/'configs/property_predictor/default.yaml')
        properties.update(kind=kind,dataset=record['dataset'],samples=str(samples),reference_files=[str(reference)],
            model_cache=str(cache),reference_cache=str(output/'reference_cache'),output=str(job/'properties'),
            allow_proxy_transfer=kind=='layer')
        execute(OmegaConf.create({'task':'predict_properties','device':'cpu','threads':1,'seed':142,
                                 'property_predictor':properties}))
        metrics=OmegaConf.load(root/'configs/evaluation/default.yaml')
        metrics.update(kind=kind,dataset=record['dataset'],samples=str(samples),reference_files=[str(reference)],
            assessments=str(job/'properties/predictions.csv'),reference_assessments=str(job/'properties/reference_predictions.csv'),
            assessment_metadata=str(job/'properties/evaluator.json'),output=str(job/'metrics'),
            no_fingerprints=True,no_reidentify=True)
        metrics.conditional.enabled=True
        execute(OmegaConf.create({'task':'evaluate','device':'cpu','threads':1,'seed':142,'evaluation':metrics}))
        assert list((job/'metrics').glob('*.png'))
        with (job/'properties/predictions.csv').open(encoding='utf-8-sig') as stream:
            rows = list(csv.DictReader(stream))
        assert len(rows) == 1 and not rows[0]['prediction_error']
        cases[title] = {'dataset': record['dataset'], 'kind': kind, 'atoms': len(structure),
                       'predictions': result[0], 'graphs': graph_results, 'expanded_graph_agrees': True,
                       'band_gap_fidelities_verified': fidelities,
                       'vacuum_invariance_checked': kind == 'layer'}
    return {'passed': True, 'models': predictor.identity, 'cases': cases,
        'max_batch_vs_official_single_absolute_difference': max(differences),
        'scope': 'Two true dataset examples verify API and graph conversion, not predictor accuracy or generator quality; '
                 'no retraining, relaxation or server actions'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-cache', type=Path, default=project_root()/'data/scr/pretrained/matgl')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    report = check(args.model_cache, args.output)
    write_json(args.output/'integration.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
