from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.config import ExperimentConfig  # noqa: E402
from cce_mobility.core.env import load_dotenv  # noqa: E402
from cce_mobility.core.llm import ChatClient  # noqa: E402
from cce_mobility.core.schema import EvaluationRecord, InterventionCandidate  # noqa: E402
from cce_mobility.integrations.mobilitybench_continuation import (  # noqa: E402
    file_sha256,
    stable_hash,
)
from cce_mobility.interventions.analyzer import HeuristicTrajectoryAnalyzer  # noqa: E402
from cce_mobility.interventions.frontier import FrontierSelector  # noqa: E402
from cce_mobility.interventions.generator import (  # noqa: E402
    GridInterventionGenerator,
    LLMInterventionGenerator,
    OntologyAwareLLMInterventionGenerator,
)
from cce_mobility.interventions.spec_builder import RuleSpecBuilder  # noqa: E402
from cce_mobility.interventions.validator import HeuristicInterventionValidator  # noqa: E402
from cce_mobility.methods.factory import create_method  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a fixed intervention JSONL from official MobilityBench "
            "factual traces for strict paired post-hoc evaluation."
        )
    )
    parser.add_argument("config", help="Experiment JSON/YAML config.")
    parser.add_argument("output", help="Output frozen intervention JSONL path.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    config = ExperimentConfig.from_yaml(Path(args.config))
    if config.intervention.policy == "frozen":
        raise SystemExit("Generation config cannot use intervention.policy=frozen.")

    output_path = Path(args.output)
    if output_path.exists() and bool(config.extra.get("refuse_overwrite", False)):
        raise FileExistsError(f"Refusing to overwrite locked intervention set: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates, world_model_transport = _generate_frozen_interventions(config)
    expected_samples = _expected_sample_count(config)
    expected_slots = expected_samples * config.intervention.selected_budget
    actual_samples = len({candidate.spec.sample_id for candidate in candidates})
    if bool(config.extra.get("require_full_coverage", False)) and (
        len(candidates) != expected_slots or actual_samples != expected_samples
    ):
        raise RuntimeError(
            "Refusing to write incomplete frozen intervention set: "
            f"expected_samples={expected_samples}, actual_samples={actual_samples}, "
            f"expected_slots={expected_slots}, actual_slots={len(candidates)}"
        )
    with output_path.open("w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False) + "\n")

    summary = _summary(
        config,
        output_path,
        candidates,
        world_model_transport=world_model_transport,
    )
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def generate_frozen_interventions(config: ExperimentConfig) -> list[InterventionCandidate]:
    candidates, _ = _generate_frozen_interventions(config)
    return candidates


def _generate_frozen_interventions(
    config: ExperimentConfig,
) -> tuple[list[InterventionCandidate], dict[str, Any]]:
    factual_records = _load_factual_records(Path(config.benchmark.data_path))
    if config.benchmark.limit:
        factual_records = factual_records[: config.benchmark.limit]

    method = create_method(config.method)
    _configure_method(method, config.extra)
    _load_method_memory(method, config.extra, config.experiment_name)
    _maybe_freeze_method(method, config.extra, config.experiment_name)

    world_model_client = ChatClient(config.world_model)
    analyzer = HeuristicTrajectoryAnalyzer()
    grid_generator = GridInterventionGenerator(
        dimensions=config.intervention.dimensions or None,
        severity_grid=config.intervention.severity_grid,
    )
    if config.intervention.policy in {
        "ontology",
        "ontology_discovery",
        "llm_ontology",
        "ontology_frontier",
    }:
        generator = OntologyAwareLLMInterventionGenerator(
            world_model_client,
            dimensions=config.intervention.dimensions or None,
            severity_grid=config.intervention.severity_grid,
            fallback=grid_generator,
        )
    elif config.intervention.policy in {"llm", "llm_frontier", "world_model"}:
        generator = LLMInterventionGenerator(
            world_model_client,
            dimensions=config.intervention.dimensions or None,
            severity_grid=config.intervention.severity_grid,
            fallback=grid_generator,
        )
    else:
        generator = grid_generator

    seed_ontology_fn = getattr(method, "seed_intervention_ontology", None)
    if callable(seed_ontology_fn):
        seed_ontology_fn(config.intervention.dimensions or None)
    _apply_method_guidance(method, generator)

    spec_builder = RuleSpecBuilder()
    validator = HeuristicInterventionValidator(config.intervention.validity_threshold)
    selector = FrontierSelector(
        low=config.intervention.frontier_low,
        high=config.intervention.frontier_high,
    )

    frozen: list[InterventionCandidate] = []
    for sample_idx, base_record in enumerate(factual_records, start=1):
        sample = base_record.sample
        factual = base_record.factual
        if factual is None:
            print(
                f"[{config.experiment_name}] sample {sample_idx}/{len(factual_records)} "
                f"{sample.sample_id}: skip missing factual",
                file=sys.stderr,
                flush=True,
            )
            continue
        print(
            f"[{config.experiment_name}] sample {sample_idx}/{len(factual_records)} "
            f"{sample.sample_id}: generate frozen intervention(s)",
            file=sys.stderr,
            flush=True,
        )
        points = analyzer.find_points(sample, factual)
        raw_candidates = generator.generate(sample, factual, points)
        prepared: list[InterventionCandidate] = []
        for candidate in raw_candidates[: config.intervention.candidate_budget]:
            candidate.recovery_spec = candidate.recovery_spec or spec_builder.build(
                candidate, sample
            )
            candidate = validator.validate(candidate, sample)
            if validator.is_valid(candidate):
                prepared.append(candidate)
        selected = selector.select(prepared, budget=config.intervention.selected_budget)
        for rank, candidate in enumerate(selected, start=1):
            candidate = candidate.model_copy(deep=True)
            candidate.spec.metadata["frozen_generation_experiment"] = config.experiment_name
            candidate.spec.metadata["frozen_generation_rank"] = rank
            candidate.spec.metadata["frozen_generation_policy"] = config.intervention.policy
            frozen.append(candidate)
        print(f"  selected {len(selected)} frozen counterfactual(s)", file=sys.stderr, flush=True)
    return frozen, world_model_client.transport_metrics()


def _load_factual_records(path: Path) -> list[EvaluationRecord]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(EvaluationRecord.model_validate(json.loads(line)))
    return rows


def _configure_method(method: Any, extra: dict[str, Any]) -> None:
    configure_fn = getattr(method, "configure", None)
    if callable(configure_fn):
        configure_fn(extra)


def _load_method_memory(method: Any, extra: dict[str, Any], experiment_name: str) -> None:
    path = extra.get("load_skill_memory_path") or extra.get("load_registry_path")
    if path:
        load_fn = getattr(method, "load_registry", None)
        if callable(load_fn):
            load_fn(path)
            print(f"[{experiment_name}] loaded skill/spec memory from {path}", file=sys.stderr)


def _maybe_freeze_method(method: Any, extra: dict[str, Any], experiment_name: str) -> None:
    if not bool(extra.get("freeze_method", False)):
        return
    freeze_fn = getattr(method, "freeze", None)
    if callable(freeze_fn):
        freeze_fn()
        print(f"[{experiment_name}] method memory frozen", file=sys.stderr)


def _apply_method_guidance(method: Any, generator: Any) -> None:
    wm_guidance_fn = getattr(method, "world_model_guidance", None)
    if callable(wm_guidance_fn) and hasattr(generator, "set_guidance"):
        generator.set_guidance(wm_guidance_fn())
    ontology_registry_fn = getattr(method, "intervention_ontology_registry", None)
    if callable(ontology_registry_fn) and hasattr(generator, "set_ontology_registry"):
        generator.set_ontology_registry(ontology_registry_fn())


def _summary(
    config: ExperimentConfig,
    output_path: Path,
    candidates: list[InterventionCandidate],
    *,
    world_model_transport: dict[str, Any] | None = None,
) -> dict[str, Any]:
    sample_ids = {candidate.spec.sample_id for candidate in candidates}
    expected = _expected_sample_count(config) * config.intervention.selected_budget
    factual_path = Path(config.benchmark.data_path).expanduser().resolve()
    world_memory = _optional_artifact(config.extra.get("load_skill_memory_path"))
    world_ontology = _optional_artifact(config.extra.get("load_intervention_ontology_path"))
    return {
        "config": config.experiment_name,
        "config_sha256": stable_hash(config.model_dump(mode="json")),
        "official_factual_records_path": str(factual_path),
        "official_factual_records_sha256": file_sha256(factual_path),
        "world_memory": world_memory,
        "world_ontology": world_ontology,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "policy": config.intervention.policy,
        "candidate_count": len(candidates),
        "sample_count": len(sample_ids),
        "expected_counterfactual_slots": expected,
        "coverage_rate": round(len(candidates) / expected, 4) if expected else 0.0,
        "dimensions": _dimension_counts(candidates),
        "domain_operators": _domain_operator_counts(candidates),
        "capability_targets": _capability_counts(candidates),
        "selected_ontology_id_count": sum(
            1
            for candidate in candidates
            if candidate.spec.metadata.get("llm_wm_selected_ontology_id")
        ),
        "selected_spec_id_count": sum(
            1 for candidate in candidates if candidate.spec.metadata.get("llm_wm_selected_spec_id")
        ),
        "world_model_transport": world_model_transport or {},
    }


def _optional_artifact(value: Any) -> dict[str, Any] | None:
    if not value:
        return None
    path = Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured frozen-generation artifact is missing: {path}")
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _expected_sample_count(config: ExperimentConfig) -> int:
    if config.benchmark.limit:
        return int(config.benchmark.limit)
    with Path(config.benchmark.data_path).open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _dimension_counts(candidates: list[InterventionCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        counts[candidate.spec.dimension] = counts.get(candidate.spec.dimension, 0) + 1
    return counts


def _domain_operator_counts(candidates: list[InterventionCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        key = candidate.spec.domain_operator or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _capability_counts(candidates: list[InterventionCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for candidate in candidates:
        for capability in candidate.spec.capability_targets or ["unknown"]:
            counts[capability] = counts.get(capability, 0) + 1
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
