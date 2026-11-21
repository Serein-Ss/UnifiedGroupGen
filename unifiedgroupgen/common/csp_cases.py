"""Geometry-free requests shared by CSP generation and matching."""
from dataclasses import dataclass
import hashlib
import json

from unifiedgroupgen.group_constraints.chemistry import ChemicalControl


@dataclass(frozen=True)
class CSPCase:
    dataset: str
    case_id: str
    kind: str
    chemistry: ChemicalControl

    @property
    def key(self):
        return self.dataset, self.case_id

    def to_dict(self):
        return dict(dataset=self.dataset, case_id=self.case_id, kind=self.kind,
                    counts=dict(self.chemistry.counts))


@dataclass(frozen=True)
class CSPManifest:
    candidates_per_case: int
    cases: tuple[CSPCase, ...]

    def to_dict(self):
        return dict(schema='ugg_csp_cases_v1', candidates_per_case=self.candidates_per_case,
                    cases=[case.to_dict() for case in self.cases])

    @property
    def digest(self):
        content = json.dumps(self.to_dict(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode('utf-8')).hexdigest()


def parse_csp_manifest(data):
    if (not isinstance(data, dict) or set(data) != {'schema', 'candidates_per_case', 'cases'}
            or data['schema'] != 'ugg_csp_cases_v1'):
        raise ValueError('Expected a ugg_csp_cases_v1 manifest')
    count, rows = data['candidates_per_case'], data['cases']
    if type(count) is not int or count < 1 or not isinstance(rows, list) or not rows:
        raise ValueError('A CSP manifest needs nonempty cases and a positive integer candidate count')
    cases = []
    seen = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'dataset', 'case_id', 'kind', 'counts'}:
            raise ValueError('CSP cases contain only dataset, case_id, kind and counts')
        dataset, case_id, kind = row['dataset'], row['case_id'], row['kind']
        if not all(isinstance(value, str) and value.strip() for value in (dataset, case_id)):
            raise ValueError('CSP cases require a source dataset and a nonempty case_id')
        if kind not in ('space', 'layer', 'plane'):
            raise ValueError(f'Unsupported CSP structure kind: {kind}')
        if (not isinstance(row['counts'], dict) or not row['counts']
                or any(type(n) is not int or n < 1 for n in row['counts'].values())):
            raise ValueError('CSP cases require nonempty per-cell counts')
        case = CSPCase(dataset, case_id, kind, ChemicalControl.from_inputs(counts=row['counts']))
        if case.key in seen:
            raise ValueError(f'Duplicate CSP case: {case.key}')
        seen.add(case.key)
        cases.append(case)
    return CSPManifest(count, tuple(cases))


def read_csp_manifest(path):
    return parse_csp_manifest(json.loads(path.read_text(encoding='utf-8')))
