import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import yaml
import pytest

from cce_mobility.methods.factory import create_method

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_official_skillopt_loads_markdown_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "best_skill.md"
    artifact.write_text("# Recovery\nVerify current availability.\n", encoding="utf-8")

    method = create_method("official_skillopt")
    method.load_registry(artifact)

    assert "Verify current availability" in method.agent_guidance()
    assert method.report_metrics()["official_skillopt_artifact_loaded"] == 1


def test_only_official_skillopt_artifact_is_published() -> None:
    assert create_method("official_skillopt").name == "official_skillopt"
    with pytest.raises(ValueError, match="Unknown method"):
        create_method("skillopt_like")


def test_official_skillopt_runner_registers_mobilitybench_adapter() -> None:
    runner = (
        PROJECT_ROOT
        / "src"
        / "cce_mobility"
        / "workflows"
        / "baselines"
        / "run_official_skillopt.py"
    )
    text = runner.read_text(encoding="utf-8")

    assert "SKILLOPT_ROOT" in text
    assert "CCEMobilityBenchSkillOptAdapter" in text
    assert 'official_train._ENV_REGISTRY["cce_mobilitybench"]' in text


def test_official_skillopt_selects_only_configured_adapter(tmp_path: Path) -> None:
    script = PROJECT_ROOT / "src/cce_mobility/workflows/baselines" / "run_official_skillopt.py"
    spec = importlib.util.spec_from_file_location("official_skillopt_runner_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = tmp_path / "train.yaml"
    config.write_text(yaml.safe_dump({"env": {"name": "cce_mobilitybench"}}), encoding="utf-8")
    assert module._selected_environment(["--config", str(config)]) == "cce_mobilitybench"
    assert (
        module._selected_environment(
            ["--config", str(config), "--cfg-options", "env.name=cce_travelplanner"]
        )
        == "cce_travelplanner"
    )


def test_public_mb_skillopt_template_is_portable() -> None:
    path = PROJECT_ROOT / "configs" / "baselines" / "mobilitybench_skillopt.example.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    assert config["env"]["name"] == "cce_mobilitybench"
    assert config["env"]["judge_api_key_env"] == "CCE_JUDGE_API_KEY"
    assert (PROJECT_ROOT / config["env"]["skill_init"]).is_file()
    assert config["env"]["split_dir"].startswith("runs/local_data/")
    assert config["evaluation"]["eval_test"] is False


@pytest.mark.parametrize("environment_only", [False, True])
def test_skillopt_bridges_role_keys_without_requiring_provider_aliases(
    monkeypatch, environment_only
):
    from cce_mobility.workflows.baselines.run_official_skillopt import _bridge_model_keys

    names = (
        "CCE_MODEL_API_KEY",
        "CCE_JUDGE_API_KEY",
        "QWEN_CHAT_API_KEY",
        "OPTIMIZER_QWEN_CHAT_API_KEY",
        "TARGET_QWEN_CHAT_API_KEY",
        "JUDGE_API_KEY",
    )
    for name in names:
        monkeypatch.delenv(name, raising=False)
    local = ModuleType("cce_mobility.local_secrets")
    if environment_only:
        monkeypatch.setenv("CCE_MODEL_API_KEY", "example-model-token")
        monkeypatch.setenv("CCE_JUDGE_API_KEY", "example-judge-token")
    else:
        local.CCE_MODEL_API_KEY = "example-model-token"
        local.CCE_JUDGE_API_KEY = "example-judge-token"
    monkeypatch.setitem(sys.modules, local.__name__, local)

    _bridge_model_keys()

    import os

    assert os.environ["QWEN_CHAT_API_KEY"] == "example-model-token"
    assert os.environ["OPTIMIZER_QWEN_CHAT_API_KEY"] == "example-model-token"
    assert os.environ["TARGET_QWEN_CHAT_API_KEY"] == "example-model-token"
    assert os.environ["JUDGE_API_KEY"] == "example-judge-token"
