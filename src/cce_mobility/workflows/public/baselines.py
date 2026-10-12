from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import artifact, read_json, write_locked


def prepare(config_path: Path, destination: Path, skillopt: Path | None = None) -> list[Path]:
    from cce_mobility.workflows.public.pipeline import _tp_challenges
    from cce_mobility.workflows.travelplanner.run_travelplanner_memory_curation_ablation import (
        _holdout_eval_config,
    )

    config = read_json(config_path)
    if "benchmark" not in config:
        raise ValueError("Use the MobilityBench baseline guide for official continuation baselines")
    if not (Path(config["output_root"]) / "test_manifest.json").is_file():
        raise FileNotFoundError("Complete the CCE frozen test before preparing matched baselines")
    config = _tp_challenges(config, "test", require_existing=True)
    created = []
    for method in ("no_evolution", "reflexion", "voyager", "official_skillopt"):
        if method == "official_skillopt" and skillopt is None:
            continue
        value = _holdout_eval_config(
            config,
            split="test",
            label=method,
            method=method,
            frozen_interventions=Path(config["test"]["frozen_intervention_path"]),
        )
        value["output_dir"] = str(Path(config["output_root"]) / "baselines")
        value["extra"]["freeze_method"] = method in {"no_evolution", "official_skillopt"}
        value["extra"]["save_skill_memory"] = method in {"reflexion", "voyager"}
        value["extra"]["agent_owned_retrieval"] = False
        if method == "official_skillopt":
            value["extra"]["base_agent_guidance_path"] = str(skillopt.resolve())
            value["extra"]["external_artifact"] = artifact(skillopt.resolve())
        output = destination / f"{method}.json"
        write_locked(output, value)
        created.append(output)
    return created


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare named TP baselines on the shared frozen test challenges"
    )
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skillopt-artifact", type=Path)
    args = parser.parse_args()
    for path in prepare(args.experiment, args.output, args.skillopt_artifact):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
