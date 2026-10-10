from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from cce_mobility.integrations.mobilitybench_continuation import (
    FORMAL_EXPERIMENT_SCRIPTS,
    cce_continuation_runtime_provenance,
    formal_experiment_code_provenance,
)
from cce_mobility.workflows.mobilitybench import run_mobilitybench_formal_coevolution_curve
from cce_mobility.workflows.travelplanner import (
    resume_travelplanner_memory_curation_ablation,
    run_travelplanner_memory_curation_ablation,
)


ROOT = Path(__file__).resolve().parents[2]


def test_scripts_directory_contains_only_the_shell_launcher() -> None:
    assert not list((ROOT / "scripts").rglob("*.py"))
    assert (ROOT / "scripts/run_mobilitybench_official_smoke.sh").is_file()


@pytest.mark.parametrize(
    "module",
    [
        "cce_mobility.workflows.public.baselines",
        "cce_mobility.workflows.travelplanner.run_travelplanner_memory_curation_ablation",
        "cce_mobility.workflows.travelplanner.resume_travelplanner_memory_curation_ablation",
        "cce_mobility.workflows.travelplanner.export_travelplanner_official_eval",
        "cce_mobility.workflows.mobilitybench.prepare_mobilitybench_continuation_protocol",
        "cce_mobility.workflows.mobilitybench.run_mobilitybench_formal_coevolution_train",
        "cce_mobility.workflows.mobilitybench.run_mobilitybench_formal_coevolution_curve",
        "cce_mobility.workflows.shared.export_agent_skill_prompt",
    ],
)
def test_advanced_workflow_module_help(module: str) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout.lower()


def test_runtime_fingerprints_use_existing_package_entrypoints() -> None:
    for name in FORMAL_EXPERIMENT_SCRIPTS:
        assert (ROOT / name).is_file()
        if name.endswith(".py"):
            assert name.startswith("src/cce_mobility/workflows/")
    runtime = cce_continuation_runtime_provenance(ROOT)
    formal = formal_experiment_code_provenance(ROOT)
    assert runtime["runtime_file_count"] > 0
    assert formal["formal_file_count"] >= runtime["runtime_file_count"]
    assert len(runtime["runtime_source_sha256"]) == 64
    assert len(formal["formal_source_sha256"]) == 64


def test_tp_resume_imports_the_packaged_runner() -> None:
    runner = resume_travelplanner_memory_curation_ablation._load_runner()
    assert runner is run_travelplanner_memory_curation_ablation


def test_mb_curve_launches_the_packaged_continuation(monkeypatch, tmp_path: Path) -> None:
    calls = []
    monkeypatch.setattr(
        run_mobilitybench_formal_coevolution_curve,
        "_run_command",
        lambda command, **kwargs: calls.append((command, kwargs)),
    )
    config = {
        "mobilitybench_python": sys.executable,
        "mobilitybench_root": str(tmp_path),
        "expected_count": 1,
    }
    point = {
        "name": "full",
        "eval_config": {"output_dir": str(tmp_path / "evaluation")},
        "eval_config_path": str(tmp_path / "evaluation.json"),
    }
    run_mobilitybench_formal_coevolution_curve.evaluate(config, [point], selected_names=set())
    assert calls[0][0] == [
        sys.executable,
        "-m",
        "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_continuation",
        point["eval_config_path"],
    ]
    assert calls[0][1]["cwd"] == tmp_path
