from __future__ import annotations

import copy
import csv
import json
import os
import subprocess
from pathlib import Path

from cce_mobility.core.config import ExperimentConfig, ModelConfig
from cce_mobility.core.llm import ChatClient, _get_secret
from cce_mobility.core.schema import EvaluationRecord
from .artifacts import artifact, read_json, sha256, source_fingerprint, stage_lock, write_locked
from .configuration import validate_models
from .data import pin_checkout

ROOT = Path(__file__).resolve().parents[4]


def check_models(path: Path) -> dict:
    payload = read_json(path)
    configs = validate_models({r: payload[r] for r in ("agent", "world_model", "judge")})
    rows = []
    for role, value in configs.items():
        client = ChatClient(ModelConfig.model_validate(value))
        try:
            response = client.chat(
                system_prompt="Return only OK.", user_prompt="Connectivity check.", max_tokens=32
            )
            if (
                not response.ok
                or not response.text.strip()
                or response.text.startswith("[LLM_ERROR]")
            ):
                raise RuntimeError("Empty or failed model response")
            rows.append(
                {
                    "role": role,
                    "ok": True,
                    "configured_model": value["model"],
                    "transport": client.transport_metrics(),
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "role": role,
                    "ok": False,
                    "configured_model": value["model"],
                    "error_type": type(exc).__name__,
                }
            )
    return {"all_ok": all(row["ok"] for row in rows), "results": rows}


def run_stage(path: Path, stage: str) -> dict:
    config = read_json(path)
    models = validate_models({r: config[r] for r in ("agent", "world_model", "judge")})
    config.update(models)
    if not config.get("public_profile"):
        raise ValueError("Use cce prepare-config to generate a portable experiment configuration")
    benchmark = "travelplanner" if "benchmark" in config else "mobilitybench"
    root = Path(config["output_root"])
    with stage_lock(root):
        _verify_input_provenance(config, benchmark)
        write_locked(root / "experiment_config.json", config)
        write_locked(root / "source_manifest.json", {"package_sha256": source_fingerprint()})
        if benchmark == "travelplanner":
            return _tp(config, path, stage)
        return _mb(config, path, stage)


def _verify_input_provenance(config: dict, benchmark: str) -> None:
    provenance = config["input_provenance"]
    for key in ("model_config", "data_manifest"):
        item = provenance[key]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Input changed since prepare-config: {key}")
    if config.get("base_agent_guidance"):
        item = config["base_agent_guidance"]
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Frozen external guidance changed")
    manifest = read_json(Path(provenance["data_manifest"]["path"]))
    for item in manifest["files"].values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Prepared data changed: {item['path']}")
    repo = Path(
        config["benchmark"]["repo_path"]
        if benchmark == "travelplanner"
        else config["mobilitybench_root"]
    )
    pin_checkout(benchmark, repo, provenance["upstream_code"]["revision"])


def _tp(config: dict, path: Path, stage: str) -> dict:
    from cce_mobility.workflows.travelplanner import (
        run_travelplanner_memory_curation_ablation as runner,
    )

    runner._validate_config(config)
    arm = config["variant"]
    root = Path(config["output_root"])
    if stage == "train":
        runner.prepare(config, config_path=path)
        manifests = [
            runner.run_training_cycle(config, arm=arm, cycle=i)
            for i in range(1, config["cycles"] + 1)
        ]
        return {"stage": "train", "completed_cycles": len(manifests), "output_root": str(root)}
    if stage == "validate":
        effective = _tp_challenges(config, "validation")
        result = runner.run_validation(effective, arms=(arm,))
        chosen = result["selected"]["selected"][arm]
        selected = {
            "selection_split": "validation",
            "variant": arm,
            "selection_rule": result["matrix"]["selection_rule"],
            **chosen,
            "validation_challenges": artifact(
                Path(effective["validation"]["frozen_intervention_path"])
            ),
            "config_sha256": sha256(root / "experiment_config.json"),
        }
        selected["world_artifacts"] = _world_artifacts(root, arm, chosen["completed_cycle"])
        write_locked(root / "selected_checkpoint.json", selected)
        return selected
    selected = _selected(root, config)
    if stage == "test":
        effective = _tp_challenges(config, "test")
        before = sha256(Path(selected["agent_memory_path"]))
        result = runner.run_test(effective, arms=(arm,))
        if sha256(Path(selected["agent_memory_path"])) != before:
            raise RuntimeError("Test modified the selected memory")
        write_locked(
            root / "test_manifest.json",
            {
                "selection": artifact(root / "selected_checkpoint.json"),
                "challenge_set": artifact(Path(effective["test"]["frozen_intervention_path"])),
                "memory_sha256_before": before,
                "memory_sha256_after": before,
                "method_updates": 0,
            },
        )
        return result
    # Official factual planning must be newly executed with this method, rather
    # than exporting the shared no-evolution context used for dynamic pairing.
    effective = _tp_challenges(config, "test", require_existing=True)
    evaluation = runner._holdout_eval_config(
        effective,
        split="test",
        label="official_factual",
        method="agent_skill_only",
        frozen_interventions=Path(effective["test"]["frozen_intervention_path"]),
        agent_memory=Path(selected["agent_memory_path"]),
    )
    evaluation["intervention"]["selected_budget"] = 0
    evaluation["extra"].pop("frozen_intervention_path", None)
    evaluation["extra"]["skip_factual_judge"] = True
    report = runner._run_or_resume(
        evaluation,
        expected_samples=config["test"]["sample_count"],
        expected_counterfactual=0,
        quality=config["quality_gate"],
    )
    from .travelplanner_official import evaluate

    return evaluate(config, Path(report["records_path"]), root / "official")


def _tp_challenges(config: dict, split: str, *, require_existing: bool = False) -> dict:
    from cce_mobility.workflows.travelplanner.generate_intervention_set import (
        generate_intervention_set,
    )

    effective = copy.deepcopy(config)
    path = Path(config[split]["frozen_intervention_path"])
    manifest_path = path.with_suffix(".manifest.json")
    if not path.exists():
        if require_existing:
            raise FileNotFoundError("Run cce test before evaluate-official")
        seed_config = ExperimentConfig.model_validate(
            {
                "experiment_name": f"{config['experiment_name']}_{split}_challenge_seed",
                "output_dir": str(Path(config["output_root"]) / "challenge_seed"),
                "seed": config["seed"],
                "method": "no_evolution",
                "benchmark": {
                    "name": "travelplanner",
                    "repo_path": config["benchmark"]["repo_path"],
                    "data_path": config["benchmark"][f"{split}_data_path"],
                    "split": split,
                    "limit": config[split]["sample_count"],
                    "extra": {"planner_mode": "adaptive_rule"},
                },
                **{r: config[r] for r in ("agent", "world_model", "judge")},
                "intervention": {
                    **config["train_intervention"],
                    "policy": "fixed",
                    "selected_budget": config[split]["selected_budget"],
                },
            }
        )
        candidates = generate_intervention_set(seed_config)
        if not candidates:
            raise RuntimeError("No valid held-out challenges generated")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in candidates), encoding="utf-8"
        )
        write_locked(
            manifest_path,
            {
                "split": split,
                "challenge_set": artifact(path),
                "source_data": artifact(Path(config["benchmark"][f"{split}_data_path"])),
                "generation": "fixed_seed_ontology_grid_grounded_in_factual_trace",
                "historical_paper_artifact": False,
            },
        )
    manifest = read_json(manifest_path)
    if sha256(path) != manifest["challenge_set"]["sha256"]:
        raise ValueError("Held-out challenge set changed")
    count = sum(bool(line.strip()) for line in path.read_text(encoding="utf-8").splitlines())
    effective[split]["expected_counterfactual_count"] = count
    return effective


def _selected(root: Path, config: dict) -> dict:
    path = root / "selected_checkpoint.json"
    if not path.exists():
        raise FileNotFoundError("Run cce validate before cce test")
    selected = read_json(path)
    if selected["selection_split"] != "validation" or selected["config_sha256"] != sha256(
        root / "experiment_config.json"
    ):
        raise ValueError("Selection protocol differs from current experiment")
    if (
        selected["variant"] != config["variant"]
        or sha256(Path(selected["agent_memory_path"])) != selected["agent_memory_sha256"]
    ):
        raise ValueError("Selected memory missing or changed")
    for item in selected.get("world_artifacts", {}).values():
        if sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError("Selected world memory or ontology changed")
    return selected


def _world_artifacts(root: Path, arm: str, cycle: int) -> dict:
    directory = root / arm / f"round_{cycle}"
    return {
        name: artifact(directory / name) for name in ("world_memory.jsonl", "world_ontology.jsonl")
    }


def _mb(config: dict, path: Path, stage: str) -> dict:
    from cce_mobility.workflows.mobilitybench import (
        run_mobilitybench_formal_coevolution_train as runner,
    )

    runner._validate_config(config)
    root = Path(config["output_root"])
    arm = config["variant"]
    if stage == "train":
        _mb_runtime(config)
        _mb_factual(config, "train", config["train_task_count"] * config["cycles"])
        runner.prepare(config, config_path=path)
        manifests = [
            runner.run_cycle(config, arm=arm, cycle=i) for i in range(1, config["cycles"] + 1)
        ]
        return {"stage": "train", "completed_cycles": len(manifests), "output_root": str(root)}
    if stage == "evaluate-official":
        _selected(root, config)
        if not (root / "test_manifest.json").exists():
            raise FileNotFoundError("Run cce test before evaluate-official")
        factual = root / "factual" / "test" / "official"
        ground_truth = root / "factual" / "test" / "dataset.csv"
        command = [
            config["mobilitybench_python"],
            "-m",
            "cce_mobility.workflows.mobilitybench.run_mobilitybench_official_eval",
            "--mobilitybench-root",
            config["mobilitybench_root"],
            "--run-dir",
            str(factual),
            "--ground-truth",
            str(ground_truth),
            "--output-dir",
            str(root / "official"),
        ]
        _mb_command(config, command)
        return {
            "official_output": str(root / "official"),
            "scope": "Official factual metrics; continuation recovery reported separately",
        }
    split = "validation" if stage == "validate" else "test"
    if stage == "test":
        _selected(root, config)
    for cycle in range(1, config["cycles"] + 1):
        manifest = read_json(root / arm / f"round_{cycle}" / "round_manifest.json")
        runner._validate_completed_manifest(manifest, root / arm / f"round_{cycle}")
    factual = _mb_factual(config, split, config[split]["sample_count"])
    cases = _mb_challenges(config, split, factual)
    if stage == "validate":
        rows = []
        for cycle in range(1, config["cycles"] + 1):
            memory = root / arm / f"round_{cycle}" / "agent_memory.jsonl"
            report = _mb_holdout(config, split, cases, memory, f"round_{cycle}")
            from cce_mobility.workflows.travelplanner.run_travelplanner_memory_curation_ablation import (
                _selection_score,
            )

            rows.append(
                {
                    "run_name": f"{arm}_round_{cycle}",
                    "completed_cycle": cycle,
                    "agent_memory_path": str(memory),
                    "agent_memory_sha256": sha256(memory),
                    "validation_metrics": report["metrics"],
                    "selection_score": _selection_score(report["metrics"]),
                }
            )
        best = max(
            rows,
            key=lambda row: (
                row["selection_score"],
                row["validation_metrics"].get("counterfactual_recovery_avg", 0),
                -row["completed_cycle"],
            ),
        )
        selection = {
            "selection_split": "validation",
            "variant": arm,
            **best,
            "selection_rule": "recovery - 0.7*low - 0.8*unavailable + 0.2*final; recovery then earlier cycle",
            "validation_cases": artifact(cases),
            "config_sha256": sha256(root / "experiment_config.json"),
        }
        selection["world_artifacts"] = _world_artifacts(root, arm, best["completed_cycle"])
        write_locked(root / "validation" / "matrix_report.json", {"runs": rows})
        write_locked(root / "selected_checkpoint.json", selection)
        return selection
    chosen = _selected(root, config)
    memory = Path(chosen["agent_memory_path"])
    before = sha256(memory)
    report = _mb_holdout(config, split, cases, memory, "selected")
    if sha256(memory) != before:
        raise RuntimeError("Test modified selected memory")
    write_locked(
        root / "test_manifest.json",
        {
            "selection": artifact(root / "selected_checkpoint.json"),
            "cases": artifact(cases),
            "memory_sha256_before": before,
            "memory_sha256_after": before,
            "method_updates": 0,
        },
    )
    return report


def _mb_env(config: dict) -> dict:
    env = os.environ.copy()
    env.update(config.get("official_runtime_env", {}))
    key = _get_secret(config["agent"]["api_key_env"])
    if not key:
        raise RuntimeError("Missing acting key in local_secrets.py or environment")
    env["LLM_API_KEY"] = key
    env["CCE_MODEL_EXTRA_BODY"] = json.dumps(config["agent"].get("extra", {}).get("extra_body", {}))
    if config["agent"].get("extra", {}).get("enable_thinking") is False:
        env["CCE_MOBILITYBENCH_DISABLE_THINKING"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(ROOT / "patches/mobilitybench"),
            str(ROOT / "src"),
            str(Path(config["mobilitybench_root"]) / "src"),
        ]
    )
    return env


def _mb_command(config: dict, command: list[str]) -> None:
    subprocess.run(command, cwd=config["mobilitybench_root"], env=_mb_env(config), check=True)


def _mb_runtime(config: dict) -> None:
    import yaml

    path = Path(config["official_config_path"])
    model = config["agent"]
    url = model["base_url"] or model["chat_url"].removesuffix("/chat/completions")
    value = {
        "models": {
            config["official_model_alias"]: {
                "provider": "openai",
                "base_url": url,
                "api_key": "${LLM_API_KEY}",
                "temperature": model["temperature"],
                "max_tokens": model["max_tokens"],
                "roles": {r: model["model"] for r in ("planner", "worker", "reporter")},
            }
        },
        "agent": {
            "default_framework": "plan_and_execute",
            "sandbox_data_dir": str(Path(config["data_dir"]) / "sandbox"),
            "plan_and_execute": {"max_plan_iterations": 10, "max_worker_retries": 3},
        },
        "evaluation": {"metrics": ["tool", "instruction", "planning", "answer", "efficiency"]},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and yaml.safe_load(path.read_text()) != value:
        raise ValueError("Official runtime config differs")
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    prompts = Path(config["official_prompt_bundle_path"])
    if not prompts.exists():
        _mb_command(
            config,
            [
                config["mobilitybench_python"],
                "-m",
                "cce_mobility.workflows.mobilitybench.prepare_mobilitybench_continuation_protocol",
                "prompt-bundle",
                "--output",
                str(prompts),
            ],
        )


def _mb_factual(config: dict, split: str, size: int) -> Path:
    from cce_mobility.workflows.mobilitybench.convert_mobilitybench_official_to_cce import (
        _load_cases,
        _task_sample,
        _trajectory_from_official,
    )

    root = Path(config["output_root"]) / "factual" / split
    records = root / "records.jsonl"
    if records.exists():
        _verify_mb_factual(records, size)
        locked = read_json(root / "manifest.json")
        if locked["records"]["sha256"] != sha256(records):
            raise ValueError("Official factual records mutated")
        return records
    root.mkdir(parents=True, exist_ok=True)
    source = Path(config["data_dir"]) / f"{split}.csv"
    with source.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = list(reader)[:size]
    if len(rows) != size:
        raise ValueError("MB split does not contain requested tasks")
    dataset = root / "dataset.csv"
    with dataset.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    executable = str(Path(config["mobilitybench_python"]).parent / "mbench")
    raw_path = root / "official" / config["official_model_alias"] / "raw_results.json"
    if not raw_path.exists():
        _mb_command(
            config,
            [
                executable,
                "run",
                "--model",
                config["official_model_alias"],
                "--dataset",
                str(dataset),
                "--framework",
                "plan_and_execute",
                "--config",
                config["official_config_path"],
                "--output-dir",
                str(root / "official"),
                "--sandbox",
            ],
        )
    raw = read_json(raw_path)
    if len(raw) != size or len({str(row["case_id"]) for row in raw}) != size:
        raise RuntimeError("Official factual trace coverage incomplete or duplicated")
    cases = _load_cases(dataset)
    converted = []
    for row in raw:
        case_id = str(row["case_id"])
        if case_id not in cases or row.get("error"):
            raise RuntimeError(f"Invalid official factual trace: {case_id}")
        sample = _task_sample(cases[case_id], dataset)
        converted.append(
            EvaluationRecord(
                sample=sample,
                factual=_trajectory_from_official(
                    row, sample.sample_id, model=config["agent"]["model"]
                ),
                metadata={
                    "source": "official_mobilitybench_raw_results",
                    "official_case_id": case_id,
                },
            )
        )
    temporary = root / "records.pending.jsonl"
    temporary.write_text(
        "".join(row.model_dump_json() + "\n" for row in converted), encoding="utf-8"
    )
    _verify_mb_factual(temporary, size)
    temporary.replace(records)
    write_locked(
        root / "manifest.json",
        {
            "records": artifact(records),
            "raw_results": artifact(raw_path),
            "dataset": artifact(dataset),
            "planner_output_present": sum(
                bool(row.get("planner_output") or row.get("planner_training_data")) for row in raw
            ),
            "worker_output_avg_len": sum(len(row.get("worker_output") or []) for row in raw) / size,
            "reporter_output_present": sum(bool(row.factual.final_response) for row in converted),
            "error_count": 0,
        },
    )
    return records


def _verify_mb_factual(path: Path, size: int) -> None:
    rows = [
        EvaluationRecord.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if len(rows) != size:
        raise RuntimeError("MB factual trace sample count differs")
    for row in rows:
        if (
            not row.factual
            or not row.factual.final_response
            or not any(
                e.tool_name and e.metadata.get("official_role") == "worker"
                for e in row.factual.events
            )
        ):
            raise RuntimeError(
                f"MB factual trace has no reporter or worker tool event: {row.sample.sample_id}"
            )


def _mb_challenges(config: dict, split: str, factual: Path) -> Path:
    from cce_mobility.workflows.mobilitybench.generate_mobilitybench_official_frozen_intervention_set import (
        generate_frozen_interventions,
    )
    from cce_mobility.workflows.mobilitybench.prepare_mobilitybench_continuation_protocol import (
        audit_cases,
        prepare_cases,
    )

    path = Path(config[split]["frozen_intervention_path"])
    cases = path.with_name(f"{split}_cases.jsonl")
    if not path.exists():
        generation = ExperimentConfig.model_validate(
            {
                "experiment_name": f"{config['experiment_name']}_{split}_challenges",
                "seed": config["seed"],
                "method": "no_evolution",
                "benchmark": {
                    "name": "mobilitybench",
                    "data_path": str(factual),
                    "limit": config[split]["sample_count"],
                },
                **{r: config[r] for r in ("agent", "world_model", "judge")},
                "intervention": config["intervention"],
            }
        )
        candidates = generate_frozen_interventions(generation)
        if len(candidates) != config[split]["sample_count"]:
            raise RuntimeError("MB held-out challenge generation did not cover every task")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            "".join(row.model_dump_json() + "\n" for row in candidates), encoding="utf-8"
        )
        write_locked(
            path.with_suffix(".manifest.json"),
            {
                "challenges": artifact(path),
                "factual": artifact(factual),
                "historical_paper_artifact": False,
            },
        )
    locked = read_json(path.with_suffix(".manifest.json"))
    if locked["challenges"]["sha256"] != sha256(path) or locked["factual"]["sha256"] != sha256(
        factual
    ):
        raise ValueError("MB held-out challenges or factual source changed")
    if not cases.exists():
        prepare_cases(
            factual_records_path=factual,
            frozen_interventions_path=path,
            output_path=cases,
            expected_count=config[split]["sample_count"],
        )
    case_manifest = read_json(cases.with_suffix(".audit.json"))
    if (
        case_manifest["cases_sha256"] != sha256(cases)
        or case_manifest["factual_records_sha256"] != sha256(factual)
        or case_manifest["frozen_interventions_sha256"] != sha256(path)
    ):
        raise ValueError("Frozen MB continuation cases or their inputs changed")
    audit_cases(cases, expected_count=config[split]["sample_count"])
    return cases


def _mb_holdout(config: dict, split: str, cases: Path, memory: Path, label: str) -> dict:
    from cce_mobility.workflows.mobilitybench import (
        run_mobilitybench_formal_coevolution_train as runner,
    )

    root = Path(config["output_root"]) / split / label
    evaluation = {
        "schema_version": "mobilitybench-continuation-eval-v1",
        "method_version": config["method_version"],
        "experiment_name": f"{config['experiment_name']}_{split}_{label}",
        "output_dir": str(root),
        "continuation_cases_path": str(cases),
        "official_config_path": config["official_config_path"],
        "official_prompt_bundle_path": config["official_prompt_bundle_path"],
        "official_model_alias": config["official_model_alias"],
        "expected_count": config[split]["sample_count"],
        "max_case_attempts": config["max_case_attempts"],
        "agent": config["agent"],
        "judge": config["judge"],
        "agent_memory_path": str(memory),
        "retrieval": config["retrieval"],
        "seed": config["seed"],
    }
    report = runner._run_evaluation(config, evaluation, root / "evaluation_config.json")
    if not report["metrics"].get("clean"):
        raise RuntimeError("Hold-out execution has unresolved errors; do not select/report it")
    return report
