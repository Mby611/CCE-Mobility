from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cce_mobility.methods.factory import create_method
from cce_mobility.workflows.public.configuration import prepare_config, validate_models
from cce_mobility.workflows.public.data import prepare_data
from cce_mobility.workflows.public.pipeline import run_stage
from cce_mobility.workflows.public.travelplanner_official import patch_query_loader


def models():
    acting = {"model": "offline-agent", "provider": "placeholder"}
    return {
        "agent": acting,
        "world_model": acting,
        "judge": {"model": "offline-judge", "provider": "placeholder"},
    }


@pytest.mark.parametrize("method_name", ["reflexion", "voyager"])
def test_public_baseline_names_resolve_and_memory_can_be_frozen(method_name):
    method = create_method(method_name)
    assert method.name == method_name
    assert create_method(method_name.upper()).name == method_name
    method.before_iteration([])
    method.freeze()
    assert method.frozen is True


def dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    example = Path("examples/travelplanner_synthetic/validation_ref_info.jsonl").read_text()
    for split in ("train", "validation", "test"):
        (source / f"{split}_ref_info.jsonl").write_text(example)
        with (source / f"{split}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=["query", "org", "dest", "days", "date", "level"]
            )
            writer.writeheader()
            writer.writerow(
                {
                    "query": f"Plan a one-day {split} trip from Alpha to Beta with a budget of $1000.",
                    "org": "Alpha",
                    "dest": "Beta",
                    "days": 1,
                    "date": "2025-01-01",
                    "level": "easy",
                }
            )
    return source


def data_args(tmp_path, source):
    return SimpleNamespace(
        benchmark="travelplanner",
        output=tmp_path / "data",
        source=source,
        sandbox_source=None,
        revision="local",
        seed=42,
        train_size=1,
        validation_size=1,
        test_size=1,
    )


def test_data_cache_is_verified_and_overlap_fails(tmp_path):
    source = dataset(tmp_path)
    args = data_args(tmp_path, source)
    first = prepare_data(args)
    assert first == prepare_data(args)
    assert first["split_overlap"] is False
    (args.output / "test_ref_info.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="mutated"):
        prepare_data(args)
    (source / "test.csv").write_text((source / "train.csv").read_text())
    args.output = tmp_path / "overlap"
    with pytest.raises(ValueError, match="both"):
        prepare_data(args)


def test_model_names_protocols_and_foundation_identity_are_explicit():
    assert validate_models(models())["agent"]["model"] == "offline-agent"
    for field, value, match in (
        ("model", "your-model", "explicit"),
        ("provider", "unsupported", "protocol"),
        ("api_key", "do-not-store-secrets", "literal"),
    ):
        config = models()
        config["agent"] = {**config["agent"], field: value}
        with pytest.raises(ValueError, match=match):
            validate_models(config)
    config = models()
    config["world_model"] = {**config["world_model"], "model": "different"}
    with pytest.raises(ValueError, match="share"):
        validate_models(config)
    config = models()
    config["agent"] = {**config["agent"], "extra": {"headers": {"Authorization": "secret"}}}
    with pytest.raises(ValueError, match="literal"):
        validate_models(config)
    config = models()
    config["agent"] = {**config["agent"], "base_url": "https://secret@example.org"}
    with pytest.raises(ValueError, match="credentials"):
        validate_models(config)
    config = models()
    config["agent"] = {
        "model": "placeholder-agent",
        "provider": "openai",
        "base_url": "https://model.example.org/v1",
    }
    with pytest.raises(ValueError, match="explicit offline"):
        validate_models(config)


def test_public_training_validation_test_are_same_config(tmp_path):
    source = dataset(tmp_path)
    prepare_data(data_args(tmp_path, source))
    repo = tmp_path / "upstream"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "Synthetic upstream",
        ],
        check=True,
    )
    model_path = tmp_path / "models.json"
    model_path.write_text(json.dumps(models()))
    args = SimpleNamespace(
        benchmark="travelplanner",
        data_dir=tmp_path / "data",
        models=model_path,
        repo_path=repo,
        code_revision="",
        output=tmp_path / "experiment.json",
        output_root=tmp_path / "outputs",
        name="offline_cce",
        variant="full",
        train_tasks=1,
        cycles=1,
        max_phase_updates=2,
        seed=42,
        validation_tasks=1,
        test_tasks=1,
    )
    prepare_config(args)
    with pytest.raises(FileNotFoundError, match="validate"):
        run_stage(args.output, "test")
    assert run_stage(args.output, "train")["completed_cycles"] == 1
    selected = run_stage(args.output, "validate")
    assert selected["selection_split"] == "validation"
    run_stage(args.output, "test")
    manifest = json.loads((args.output_root / "test_manifest.json").read_text())
    assert manifest["method_updates"] == 0
    assert manifest["memory_sha256_before"] == manifest["memory_sha256_after"]
    assert selected == run_stage(args.output, "validate")
    assert set(selected["world_artifacts"]) == {"world_memory.jsonl", "world_ontology.jsonl"}
    from cce_mobility.workflows.public.baselines import prepare

    files = prepare(args.output, tmp_path / "baseline_configs")
    assert [path.name for path in files] == [
        "no_evolution.json",
        "reflexion.json",
        "voyager.json",
    ]
    for path in files:
        baseline = json.loads(path.read_text())
        assert baseline["method"] == path.stem
        assert create_method(baseline["method"]).name == path.stem
        online = path.stem in {"reflexion", "voyager"}
        assert baseline["extra"]["freeze_method"] is not online
        assert baseline["extra"]["save_skill_memory"] is online


def test_official_compatibility_patch_only_changes_loader_and_denominators():
    source = """def eval_score(set_type, file_path):
    if set_type == 'train':
        query_data_list = load_dataset('upstream')['train']
    elif set_type == 'validation':
        query_data_list = load_dataset('upstream')['validation']
    checked = unchanged_constraint(query_data_list)
    result = {}
    result['Delivery Rate'] = checked / 1000
    return result
"""
    calls = []
    namespace = {
        "__cce_query_rows": [{"id": 1}],
        "__cce_task_count": 1,
        "unchanged_constraint": lambda rows: calls.append(rows) or len(rows),
    }
    exec(compile(patch_query_loader(source), "test", "exec"), namespace)
    assert namespace["eval_score"]("test", "unused")["Delivery Rate"] == 1
    assert calls == [[{"id": 1}]]
    with pytest.raises(ValueError, match="loader"):
        patch_query_loader("def eval_score(a,b): return {}")


def test_public_mb_continuation_provenance_paths_exist():
    from cce_mobility.integrations.mobilitybench_continuation import (
        cce_continuation_runtime_provenance,
        formal_experiment_code_provenance,
    )

    root = Path(__file__).resolve().parents[2]
    assert cce_continuation_runtime_provenance(root)["runtime_file_count"] > 0
    assert formal_experiment_code_provenance(root)["formal_file_count"] > 0
