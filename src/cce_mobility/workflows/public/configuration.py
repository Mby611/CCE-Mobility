from __future__ import annotations

import copy
from pathlib import Path
from urllib.parse import urlsplit

from cce_mobility.core.config import ModelConfig
from .artifacts import artifact, read_json, write_locked
from .data import pin_checkout

ROOT = Path(__file__).resolve().parents[4]
VARIANTS = {
    "full": "full",
    "agent-only": "agent_only",
    "without-memory-curation": "without_memory_curation",
}


def _reject_embedded_credentials(value) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in {"api_key", "authorization", "x-api-key"}:
                raise ValueError("Use local_secrets.py, not literal credentials in model config")
            _reject_embedded_credentials(child)
    elif isinstance(value, list):
        for child in value:
            _reject_embedded_credentials(child)


def validate_models(models: dict) -> dict:
    if set(models) != {"agent", "world_model", "judge"}:
        raise ValueError("Model file must define agent, world_model, judge")
    result = {}
    for role, value in models.items():
        _reject_embedded_credentials(value)
        model = ModelConfig.model_validate(value)
        if model.max_tokens < 1 or not 0 <= model.temperature <= 2:
            raise ValueError(f"Invalid token limit or temperature for {role}")
        if (
            float(model.extra.get("api_timeout_seconds", 180)) <= 0
            or int(model.extra.get("api_max_retries", 2)) < 0
        ):
            raise ValueError(f"Invalid timeout or retry count for {role}")
        if not model.model or model.model.startswith(("your-", "replace-")):
            raise ValueError(
                f"Set an explicit available {role} model name; no automatic substitutions"
            )
        if model.model.startswith("placeholder") and model.provider != "placeholder":
            raise ValueError("Placeholder models require the explicit offline placeholder protocol")
        if model.provider not in {
            "openai",
            "openai_compatible",
            "responses",
            "anthropic",
            "placeholder",
        }:
            raise ValueError(f"Unsupported public protocol: {model.provider}")
        if model.provider != "placeholder" and not (model.chat_url or model.base_url).startswith(
            ("https://", "http://")
        ):
            raise ValueError(f"Set an explicit HTTP(S) endpoint for {role}")
        for endpoint in (model.chat_url, model.base_url):
            if endpoint and (urlsplit(endpoint).username or urlsplit(endpoint).password):
                raise ValueError("Endpoint URLs must not contain credentials")
        result[role] = model.model_dump(mode="json")
    identities = [
        (result[r]["model"], result[r]["provider"], result[r]["base_url"], result[r]["chat_url"])
        for r in ("agent", "world_model")
    ]
    if identities[0] != identities[1]:
        raise ValueError("Agent and WM must share their foundation model and endpoint")
    return result


def prepare_config(args) -> dict:
    data = args.data_dir.resolve()
    manifest = read_json(data / "split_manifest.json")
    if manifest["benchmark"] != args.benchmark:
        raise ValueError("Prepared data benchmark does not match requested benchmark")
    models = validate_models(read_json(args.models))
    for name, value in (
        ("train tasks", args.train_tasks),
        ("cycles", args.cycles),
        ("phase updates", args.max_phase_updates),
        ("validation tasks", args.validation_tasks),
        ("test tasks", args.test_tasks),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive")
    if args.cycles * 2 > args.max_phase_updates:
        raise ValueError(
            "Each paired cycle has one Agent and one World update; increase --max-phase-updates or reduce --cycles"
        )
    for split, size in (
        ("train", args.train_tasks),
        ("validation", args.validation_tasks),
        ("test", args.test_tasks),
    ):
        needed = (
            size * args.cycles if args.benchmark == "mobilitybench" and split == "train" else size
        )
        if manifest["splits"][split]["count"] < needed:
            raise ValueError(f"Prepared {split} split is too small: need {needed}")
    upstream = pin_checkout(args.benchmark, args.repo_path.resolve(), args.code_revision)
    root = args.output_root.resolve()
    if args.benchmark == "travelplanner":
        config = copy.deepcopy(
            read_json(
                ROOT / "configs/travelplanner/travelplanner_memory_curation_ablation.template.json"
            )
        )
        config.update(models)
        config.update(
            {
                "public_profile": True,
                "experiment_name": args.name,
                "output_root": str(root),
                "cycles": args.cycles,
                "train_task_count": args.train_tasks,
                "seed": args.seed,
            }
        )
        config["benchmark"] = {
            "repo_path": str(args.repo_path.resolve()),
            **{
                f"{split}_data_path": str(data / f"{split}_ref_info.jsonl")
                for split in ("train", "validation", "test")
            },
        }
        for split, count in (("validation", args.validation_tasks), ("test", args.test_tasks)):
            config[split].update(
                {
                    "sample_count": count,
                    "frozen_intervention_path": str(root / "challenges" / f"{split}.jsonl"),
                    "expected_counterfactual_count": 0,
                }
            )
    else:
        if models["agent"]["provider"] not in {"openai", "openai_compatible"}:
            raise ValueError(
                "Official MB continuation requires an OpenAI-compatible acting endpoint; Judge can use other supported protocols"
            )
        if not args.mb_python:
            raise ValueError(
                "MB requires --mb-python pointing to its separately installed upstream runtime"
            )
        if not args.mb_python.is_file():
            raise FileNotFoundError(f"Configured upstream MB Python missing: {args.mb_python}")
        config = copy.deepcopy(read_json(ROOT / "configs/mobilitybench/coevolution.template.json"))
        config.update(models)
        config.update(
            {
                "public_profile": True,
                "experiment_name": args.name,
                "output_root": str(root),
                "cycles": args.cycles,
                "train_task_count": args.train_tasks,
                "seed": args.seed,
                "mobilitybench_root": str(args.repo_path.resolve()),
                "mobilitybench_python": str(args.mb_python.resolve()),
                "data_dir": str(data),
                "train_factual_records_path": str(root / "factual" / "train" / "records.jsonl"),
                "official_config_path": str(root / "runtime" / "official.yaml"),
                "official_prompt_bundle_path": str(root / "runtime" / "prompt_bundle.json"),
            }
        )
        for split, count in (("validation", args.validation_tasks), ("test", args.test_tasks)):
            config[split] = {
                "sample_count": count,
                "selected_budget": 1,
                "frozen_intervention_path": str(root / "challenges" / f"{split}.jsonl"),
            }
    config["variant"] = VARIANTS[args.variant]
    base_guidance = getattr(args, "base_agent_guidance", None)
    if base_guidance:
        if not base_guidance.is_file():
            raise FileNotFoundError(f"Frozen base guidance missing: {base_guidance}")
        config["base_agent_guidance"] = artifact(base_guidance.resolve())
    for split in ("validation", "test"):
        frozen = getattr(args, f"frozen_{split}", None)
        if frozen:
            if not frozen.is_file() or not frozen.with_suffix(".manifest.json").is_file():
                raise FileNotFoundError(
                    "Imported frozen challenges need their .manifest.json sidecar"
                )
            config[split]["frozen_intervention_path"] = str(frozen.resolve())
    config["budget"] = {
        "cycle_cap": args.cycles,
        "phase_update_cap": args.max_phase_updates,
        "agent_evaluations_per_cycle": 3,
        "definition": "A paired cycle contains one Agent-memory update and one World-memory update, with pre/candidate/final Agent evaluations.",
    }
    config["input_provenance"] = {
        "upstream_code": upstream,
        "data_manifest": artifact(data / "split_manifest.json"),
        "model_config": artifact(args.models.resolve()),
    }
    write_locked(args.output, config)
    return config
