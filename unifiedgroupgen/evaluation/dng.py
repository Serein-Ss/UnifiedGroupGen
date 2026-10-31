"""De novo novelty, uniqueness and distribution metrics."""
import json
from .utils import rate, population_metrics

def dng_summary(rows, reference, features, refs, train, kind, novelty_files):
    report = {}
    report['dng'] = {'unique_valid':rate(r.get('unique') for r in rows),
        'novel_valid':rate((r.get('novel_against_reference') for r in rows),bool(novelty_files)),
        'vun_yield':rate([bool(r.get('basic_valid') and r.get('unique') and r.get('novel_against_reference'))
                         if r.get('novel_against_reference') is not None else None for r in rows],requested=bool(novelty_files))}
    if refs.rows:
        report['dng'].update(population_metrics(rows, reference['rows'], features, reference['fingerprints'], len(rows), (.4,10.)))
    for row in rows:
        row['template_unique'] = None
    observed = set()
    reference_templates = {r.get('template') for r in reference['rows'] if r.get('template')}
    train_templates = {json.dumps([kind, r['descriptor']['number'], sorted(o['letter'] for o in r['descriptor']['orbits'])])
                       for r, _ in train.rows if r.get('descriptor')}
    for row in rows:
        template = row.get('template')
        if template and row.get('basic_valid'):
            row['template_unique'] = template not in observed; observed.add(template)
            row['template_novel_against_test'] = template not in reference_templates if reference_templates else None
            row['template_novel_against_train'] = template not in train_templates if train_templates else None
    report['dng']['template_unique'] = rate((r.get('template_unique') for r in rows),any(r.get('template') for r in rows))
    report['dng']['template_novel_against_test'] = rate((r.get('template_novel_against_test') for r in rows),bool(reference_templates))
    report['dng']['template_novel_against_train'] = rate((r.get('template_novel_against_train') for r in rows),bool(train_templates))
    return report['dng']
