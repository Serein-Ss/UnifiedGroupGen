"""Attempt-level chemical/scaffold checks and held-out reconstruction metrics."""

from collections import Counter
import numpy as np
from .chemical_controls import ChemicalRules
from .symmetry import Descriptor, compile_descriptor
from .standard_metrics import geometry_metrics, structure_match


def structure_from_record(record):
    import torch
    from .evaluation import to_structure
    compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
    return to_structure(compiled, compiled.expand(torch.tensor(record['state'], dtype=torch.float64)))


def evaluate_task(records, reference, kind, request=None, scaffold=None):
    request = request or {}
    chemical = ChemicalRules(request.get('allowed'), request.get('required'))
    expected = request.get('composition')
    expected = {int(z): int(n) for z, n in expected.items()} if expected else None
    rows = []
    target = structure_from_record(reference) if reference else None
    for record in records:
        row = {'index': record['index'], 'generated': record.get('status') == 'generated',
               'chemical_satisfied': False, 'reference_match': False, 'geometry_valid': False}
        if not row['generated']:
            row['reason'] = record.get('reason'); rows.append(row)
            continue
        try:
            structure = structure_from_record(record)
            counts = dict(Counter(structure.atomic_numbers))
            per_sample = record.get('chemical_request', {}).get('composition')
            exact = expected or ({int(z): n for z, n in per_sample.items()} if per_sample else None)
            row['chemical_satisfied'] = chemical.matches(counts) and (exact is None or counts == exact)
            row['composition_counts'] = counts
            row['geometry_valid'] = bool(geometry_metrics(structure, kind)['legacy_structural_valid'])
            row['reference_match'] = bool(target is not None and structure_match(structure, target, kind))
            validation = record.get('validation', {})
            row['group_operation_closure'] = validation.get('group_operation_closure')
            row['exact_group_match'] = validation.get(f'recovered_{kind}_group') == record['descriptor']['number']
            if scaffold is not None:
                compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
                observed = scaffold.report(compiled, record['state'])
                row['scaffold_preserved'] = observed['scaffold_preserved']
                row['maximum_observed_displacement_angstrom'] = observed['maximum_observed_displacement_angstrom']
        except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as error:
            row['error'] = str(error)
        rows.append(row)
    attempted = len(rows)
    def fraction(key):
        return sum(r.get(key) is True for r in rows)/attempted if attempted else None
    return {'attempted': attempted, 'generated': sum(r['generated'] for r in rows),
            'chemical_satisfaction_all_attempts': fraction('chemical_satisfied'),
            'geometry_validity_all_attempts': fraction('geometry_valid'),
            'scaffold_preservation_all_attempts': fraction('scaffold_preserved') if request.get('scaffold') else None,
            'match_at_1': bool(rows[0]['reference_match']) if rows and target is not None else None,
            'match_at_k': any(r['reference_match'] for r in rows) if target is not None else None,
            'k': attempted, 'rows': rows, 'property_accuracy': None, 'physical_stability': None,
            'scope': 'Constraint satisfaction and reference reconstruction; no conditional-density or stability guarantee'}
