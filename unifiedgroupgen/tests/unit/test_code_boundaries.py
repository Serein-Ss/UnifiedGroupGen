"""Public boundaries retained while reorganizing runtime responsibilities."""

def test_reverse_coordinate_codec_remains_available_to_scaffold_users():
    from unifiedgroupgen.pl_data.encoding import _q_for_point
    from unifiedgroupgen.group_constraints.encoding import _q_for_point as implementation
    assert _q_for_point is implementation


def test_rule_and_flow_backbones_keep_their_existing_parameter_names():
    import torch
    from unifiedgroupgen.pl_modules.factory import build_models

    config = {'kind': 'space', 'groups': [('space', 2)], 'properties': {},
              'model': {'hidden': 16, 'heads': 4, 'proposal_layers': 1, 'flow_layers': 1},
              'max_atoms': 4, 'max_orbits': 1}
    torch.manual_seed(142)
    proposal, flow = build_models(config, 'cpu')
    assert any(k.startswith('transformer.') for k in proposal.state_dict())
    assert any(k.startswith('layers.') for k in flow.state_dict())


def test_runtime_functions_do_not_reference_undefined_globals():
    import builtins
    from pathlib import Path
    import symtable
    package = Path(__file__).resolve().parents[2]
    problems = []
    for path in package.rglob('*.py'):
        if path.is_relative_to(package/'tests') or any(p in path.parts for p in ('build', '__pycache__')) or path.name == 'setup.py':
            continue
        table = symtable.symtable(path.read_text(encoding='utf-8'), str(path), 'exec')
        known = set(table.get_identifiers()) | set(vars(builtins)) | {'__file__', '__name__', '__package__', '__path__', '__spec__'}
        def visit(scope):
            for name in scope.get_identifiers():
                symbol = scope.lookup(name)
                if symbol.is_global() and symbol.is_referenced() and name not in known:
                    problems.append((str(path), scope.get_name(), name))
            for child in scope.get_children():
                visit(child)
        for scope in table.get_children():
            visit(scope)
    assert not problems
