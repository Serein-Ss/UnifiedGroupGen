"""Archived source-path and command aliases; not used by new runtime commands."""
import os
import json
from pathlib import Path, PureWindowsPath
from unifiedgroupgen.common.configuration import resolve_path as current_resolve
from unifiedgroupgen.common.configuration import package_parent, runtime_pythonpath, project_root, source_key, artifact_path, root_path
DATASETS = ('mp_20', 'mpts_52', 'c2db_51', 'full', 'full_tol0p01', 'full_official_tol0p1', 'full_official_tol0p1_v7', 'full_joint_sourcegrouped', 'full_verified_v1', 'joint_verified_v1')

def resolve_path(value, root=None):
    """Resolve historical inputs and explicitly registered archived test outputs."""
    root=Path(root).resolve() if root is not None else project_root()
    if PureWindowsPath(str(value)).is_absolute():
        value=current_resolve(value, root)
    path=Path(value).expanduser()
    path=(path if path.is_absolute() else root/path).resolve()
    package=package_parent()/'unifiedgroupgen'
    from .layout import MODULE_PATHS
    for prefix in (root/'src/unifiedgroupgen', root/'unifiedgroupgen'):
        if path.is_relative_to(prefix):
            relative=path.relative_to(prefix).as_posix()
            if relative in MODULE_PATHS:
                return package/MODULE_PATHS[relative]
    code= root/'src/unifiedgroupgen'
    if path.is_relative_to(code) and code.exists(): return path
    aliases=[(root/'artifacts'/name,root/'data/dataset'/name) for name in DATASETS]
    aliases += [(root/'artifacts/source_recovery',root/'data/scr/source_recovery'),
                (root/'datasets',root/'data/scr/bundles'),
                (root/'raw_data',root/'data/scr'),
                (root.parent/'data',root/'data/scr'),
                (root/'tests/fixtures',root/'data/dataset/test_fixtures'),
                (root/'scripts/tests/fixtures',root/'data/dataset/test_fixtures'),
                (root/'tests',package/'tests/unit'),
                (root/'scripts/tests',package/'tests/unit'),
                (root/'src/unifiedgroupgen',package),
                (root/'unifiedgroupgen',package)]
    for before,after in aliases:
        before=before.resolve()
        if path.is_relative_to(before):
            target=after/path.relative_to(before)
            return path if path.exists() and not target.exists() else target
    registry=root/'outputs/tests/path_aliases.json'
    if path.is_relative_to(root/'artifacts') and registry.is_file():
        saved=json.loads(registry.read_text(encoding='utf-8'))
        for old,new in sorted(saved.items(),key=lambda item:-len(item[0])):
            before=(root/old).resolve();after=(root/new).resolve()
            if not before.is_relative_to(root/'artifacts') or not after.is_relative_to(root/'outputs/tests'):
                raise ValueError('Archived test path alias escapes its permitted directory')
            if path.is_relative_to(before):
                return path if path.exists() else after/path.relative_to(before)
    if path==root/'src': return root
    scripts=root/'scripts'
    if path != scripts and path.is_relative_to(scripts):
        from .workflow_registry import WORKFLOWS
        name=path.relative_to(scripts).with_suffix('').as_posix().replace('/','.')
        if name in WORKFLOWS:
            return package/(WORKFLOWS[name].replace('.','/')+'.py')
    return current_resolve(path, root)


def module_command(command):
    """Use installed workflow modules instead of depending on checkout script files."""
    if isinstance(command,(str,bytes)): return command
    values=list(command)
    from .workflow_registry import WORKFLOWS
    root=project_root()
    for i,value in enumerate(values):
        if not isinstance(value,(str,Path)): continue
        text=str(value).replace('\\','/')
        prefix=str(root/'scripts').replace('\\','/')+'/'
        relative=text[len(prefix):] if text.startswith(prefix) else text[8:] if text.startswith('scripts/') else None
        if relative and relative.endswith('.py'):
            name=relative[:-3].replace('/','.')
            if name in WORKFLOWS:
                return [*values[:i],'-m','unifiedgroupgen.'+WORKFLOWS[name],*values[i+1:]]
    return values
