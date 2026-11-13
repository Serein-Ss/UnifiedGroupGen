"""Synchronize the canonical joint theory into its manuscript and web views."""
import argparse
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / 'web/symmetry-lab'


def body():
    text = (ROOT / 'docs/THEORY.md').read_text(encoding='utf-8')
    return text[text.index('## 01 '):].strip() + '\n'


def replace_region(text, label, value):
    start, end = f'<!-- {label}_BEGIN -->', f'<!-- {label}_END -->'
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    return before + start + '\n\n' + value.strip() + '\n\n' + end + after


def synchronize(check=False):
    source = body()
    manuscript = ROOT / 'docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md'
    text = manuscript.read_text(encoding='utf-8')
    appendix = re.sub(r'^## (\d+) ', r'### H.\1 ', source, flags=re.M)
    updated = replace_region(text, 'JOINT_THEORY_APPENDIX', appendix)
    web_source = re.sub(r'\]\(([^:/()\s]+\.md)\)', r'](tables/\1)', source)
    for name in ['JOINT_SYMMETRY_V1.md', 'OPERATION_COMPILER.md', 'MULTITASK_SHARED_MODEL.md']:
        web_source = web_source.replace('](tables/' + name + ')', '](#math-31)')
    targets = {manuscript: updated, SITE / 'theory_content.md': web_source}
    copied = ['THEORY.md', 'RESEARCH_PLAN.md', 'IMPLEMENTATION.md', 'REFERENCES.md', 'TASK_DEFINITIONS.md', 'INDEPENDENT_TASKS.md', 'GENERIC_FEATURES_AND_BACKBONES.md', 'JOINT_FLOW_V2.md']
    for name in copied:
        content = (ROOT / 'docs' / name).read_text(encoding='utf-8')
        if name == 'THEORY.md':
            content = content.replace('、[JOINT_SYMMETRY_V1.md](JOINT_SYMMETRY_V1.md)、[OPERATION_COMPILER.md](OPERATION_COMPILER.md)', '')
        targets[SITE / 'dist/tables' / name] = content
    mismatches = []
    for path, content in targets.items():
        if path.exists() and path.read_text(encoding='utf-8') == content:
            continue
        mismatches.append(path.relative_to(ROOT).as_posix())
        if not check:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding='utf-8')
    if check and mismatches:
        raise SystemExit('Theory views are stale: ' + ', '.join(mismatches))
    print(('Checked' if check else 'Synchronized') + f' {len(targets)} theory views')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    synchronize(parser.parse_args().check)
