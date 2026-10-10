from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_name",
    ["travelplanner_offline.json", "mobilitybench_mock.json"],
)
def test_offline_smoke_entrypoint(config_name: str, tmp_path: Path) -> None:
    source = ROOT / "configs" / "smoke" / config_name
    config = json.loads(source.read_text(encoding="utf-8"))
    config["output_dir"] = str(tmp_path)
    local_config = tmp_path / config_name
    local_config.write_text(json.dumps(config), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "cce_mobility.cli", "run-config", str(local_config)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["total_samples"] == config["benchmark"]["limit"]
