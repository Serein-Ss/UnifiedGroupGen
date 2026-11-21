"""Strict .env settings without directory discovery or shell-value overrides."""
import pytest

from unifiedgroupgen.common import configuration


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.delenv('UGG_PROJECT_ROOT', raising=False)
    monkeypatch.setattr(configuration, 'package_parent', lambda: tmp_path)
    return tmp_path


def write_env(root, **changes):
    values = dict(PROJECT_ROOT=root.as_posix(), HYDRA_JOBS='${PROJECT_ROOT}/outputs',
                  WANDB_DIR='wandb', USE_WANDB_LOGGING='0')
    values.update(changes)
    (root/'.env').write_text(''.join(f'{k}="{v}"\n' for k, v in values.items() if v is not None),
                            encoding='utf-8')


def test_settings_use_env_and_ignore_shell_values_and_working_directory(workspace, monkeypatch):
    write_env(workspace)
    for key in ('PROJECT_ROOT', 'HYDRA_JOBS', 'WANDB_DIR', 'USE_WANDB_LOGGING'):
        monkeypatch.setenv(key, 'invalid-shell-setting')
    elsewhere = workspace/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    settings = configuration.RunSettings.load()
    assert settings.project_root == configuration.project_root() == workspace
    assert settings.jobs == workspace/'outputs'
    assert settings.wandb_dir == workspace/'wandb'
    assert settings.wandb_mode == 'offline'
    assert not settings.jobs.exists() and not settings.wandb_dir.exists()


def test_missing_env_does_not_fall_back_to_cwd_or_shell(workspace, monkeypatch):
    other = workspace/'other'
    other.mkdir()
    write_env(other)
    monkeypatch.chdir(other)
    monkeypatch.setenv('PROJECT_ROOT', str(other))
    with pytest.raises(FileNotFoundError, match=r'Required environment file.*\.env'):
        configuration.project_root()


@pytest.mark.parametrize('key', ['PROJECT_ROOT', 'HYDRA_JOBS', 'WANDB_DIR', 'USE_WANDB_LOGGING'])
@pytest.mark.parametrize('value', [None, '', '   '])
def test_required_keys_cannot_be_replaced_by_shell_defaults(workspace, monkeypatch, key, value):
    write_env(workspace, **{key: value})
    monkeypatch.setenv(key, str(workspace))
    with pytest.raises(ValueError, match='missing required settings: '+key):
        configuration.RunSettings.load()


@pytest.mark.parametrize('value', ['relative/path', '/missing_ugg_project_8912'])
def test_invalid_project_root_fails(workspace, value):
    write_env(workspace, PROJECT_ROOT=value)
    with pytest.raises(ValueError, match='PROJECT_ROOT must be an existing absolute directory'):
        configuration.project_root()


@pytest.mark.parametrize('key', ['HYDRA_JOBS', 'WANDB_DIR'])
@pytest.mark.parametrize('child', ['', '/child'])
def test_output_or_log_path_cannot_be_a_file_or_have_a_file_parent(workspace, key, child):
    (workspace/'occupied').write_text('fixture')
    write_env(workspace, **{key: 'occupied'+child})
    with pytest.raises(ValueError, match=key+' must point to a directory'):
        configuration.RunSettings.load()


def test_unknown_interpolation_cannot_take_values_from_shell(workspace, monkeypatch):
    monkeypatch.setenv('OTHER_ROOT', str(workspace))
    write_env(workspace, HYDRA_JOBS='${OTHER_ROOT}/outputs')
    with pytest.raises(ValueError, match='HYDRA_JOBS only supports'):
        configuration.RunSettings.load()


@pytest.mark.parametrize('flag', ['2', 'true', 'offline'])
def test_invalid_logging_flag_fails(workspace, flag):
    write_env(workspace, USE_WANDB_LOGGING=flag)
    with pytest.raises(ValueError, match='USE_WANDB_LOGGING must be'):
        configuration.RunSettings.load(mode='disabled')


@pytest.mark.parametrize('mode', ['online', 'offline', 'disabled'])
def test_explicit_logging_mode_still_works(workspace, mode):
    write_env(workspace, USE_WANDB_LOGGING='1')
    assert configuration.RunSettings.load(mode=mode).wandb_mode == mode


def test_invalid_explicit_logging_mode_fails(workspace):
    write_env(workspace)
    with pytest.raises(ValueError, match='Logging mode must be'):
        configuration.RunSettings.load(mode='invalid')


def test_installed_workspace_selector_selects_env_instead_of_bypassing_it(workspace, monkeypatch):
    selected = workspace/'selected'
    selected.mkdir()
    write_env(selected)
    monkeypatch.setenv('UGG_PROJECT_ROOT', str(selected))
    assert configuration.project_root() == selected
    (selected/'.env').unlink()
    with pytest.raises(FileNotFoundError):
        configuration.project_root()


def test_absolute_configured_output_paths_are_allowed_without_opening_arbitrary_paths(workspace):
    project = workspace/'project'
    project.mkdir()
    write_env(project, HYDRA_JOBS=(workspace/'results').as_posix(),
              WANDB_DIR=(workspace/'logs').as_posix())
    settings = configuration.RunSettings.load(project)
    assert settings.jobs == workspace/'results' and settings.wandb_dir == workspace/'logs'
    # Use this .env for the default helpers without changing the process environment.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(configuration, 'package_parent', lambda: project)
        output = configuration.artifact_path(settings.jobs/'fixture/report.json')
        log = configuration.artifact_path(settings.wandb_dir/'fixture/log.txt')
        assert output.parent.is_dir() and log.parent.is_dir()
        assert not output.exists() and not log.exists()
        with pytest.raises(ValueError, match='Generated artifacts must be inside'):
            configuration.artifact_path(workspace/'unrelated/report.json')


def test_env_changes_are_not_hidden_by_a_process_cache(workspace):
    write_env(workspace)
    assert configuration.RunSettings.load().wandb_mode == 'offline'
    write_env(workspace, USE_WANDB_LOGGING='1', HYDRA_JOBS='new_outputs')
    settings = configuration.RunSettings.load()
    assert settings.wandb_mode == 'online' and settings.jobs == workspace/'new_outputs'
