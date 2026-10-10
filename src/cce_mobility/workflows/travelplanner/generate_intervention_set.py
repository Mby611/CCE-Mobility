from __future__ import annotations

# ruff: noqa: E402

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.benchmarks import create_benchmark
from cce_mobility.core.config import ExperimentConfig
from cce_mobility.core.env import load_dotenv
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.schema import InterventionCandidate
from cce_mobility.interventions.analyzer import HeuristicTrajectoryAnalyzer
from cce_mobility.interventions.frontier import FrontierSelector
from cce_mobility.interventions.generator import GridInterventionGenerator, LLMInterventionGenerator
from cce_mobility.interventions.spec_builder import RuleSpecBuilder
from cce_mobility.interventions.travelplanner_spec import TravelPlannerSpecBuilder
from cce_mobility.interventions.validator import HeuristicInterventionValidator
from cce_mobility.methods.factory import create_method


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    if len(sys.argv) < 3:
        raise SystemExit(
            "Usage: generate_intervention_set.py <config.json> <output-interventions.jsonl>"
        )

    cfg = ExperimentConfig.from_yaml(Path(sys.argv[1]))
    output_path = Path(sys.argv[2])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    candidates = generate_intervention_set(cfg)
    with output_path.open("w", encoding="utf-8") as f:
        for candidate in candidates:
            f.write(json.dumps(candidate.model_dump(mode="json"), ensure_ascii=False) + "\n")

    summary = {
        "config": sys.argv[1],
        "output_path": str(output_path),
        "candidate_count": len(candidates),
        "sample_count": len({cand.spec.sample_id for cand in candidates}),
        "dimensions": _dimension_counts(candidates),
        "domain_operators": _domain_operator_counts(candidates),
        "capability_targets": _capability_counts(candidates),
        "wm_mapping_valid": _mapping_valid_counts(candidates),
        "selected_spec_id_count": _selected_spec_id_count(candidates),
    }
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def generate_intervention_set(cfg: ExperimentConfig) -> list[InterventionCandidate]:
    benchmark = create_benchmark(cfg.benchmark)
    agent_client = ChatClient(cfg.agent)
    wm_client = ChatClient(cfg.world_model)
    if hasattr(benchmark, "set_chat_client"):
        benchmark.set_chat_client(agent_client)

    analyzer = HeuristicTrajectoryAnalyzer()
    grid_generator = GridInterventionGenerator(
        dimensions=cfg.intervention.dimensions or None,
        severity_grid=cfg.intervention.severity_grid,
    )
    generator = (
        LLMInterventionGenerator(
            wm_client,
            dimensions=cfg.intervention.dimensions or None,
            severity_grid=cfg.intervention.severity_grid,
            fallback=grid_generator,
        )
        if cfg.intervention.policy in {"llm", "llm_frontier", "world_model"}
        else grid_generator
    )
    _apply_world_model_guidance(cfg, generator)
    spec_builder = (
        TravelPlannerSpecBuilder()
        if cfg.benchmark.name.lower() in {"travelplanner", "tp"}
        else RuleSpecBuilder()
    )
    validator = HeuristicInterventionValidator(cfg.intervention.validity_threshold)
    selector = FrontierSelector(
        low=cfg.intervention.frontier_low,
        high=cfg.intervention.frontier_high,
    )

    samples = benchmark.load_samples(limit=cfg.benchmark.limit)
    frozen: list[InterventionCandidate] = []
    for sample_idx, sample in enumerate(samples, start=1):
        print(
            f"[generate_intervention_set] sample {sample_idx}/{len(samples)} {sample.sample_id}",
            file=sys.stderr,
            flush=True,
        )
        factual = benchmark.run_agent(sample, agent_id="frozen_set_seed_agent", intervention=None)
        points = analyzer.find_points(sample, factual)
        raw_candidates = generator.generate(sample, factual, points)
        prepared = []
        for candidate in raw_candidates[: cfg.intervention.candidate_budget]:
            candidate.recovery_spec = spec_builder.build(candidate, sample)
            candidate = validator.validate(candidate, sample)
            if validator.is_valid(candidate):
                prepared.append(candidate)
        selected = selector.select(prepared, budget=cfg.intervention.selected_budget)
        frozen.extend(selected)
    return frozen


def _apply_world_model_guidance(cfg: ExperimentConfig, generator) -> None:
    path = cfg.extra.get("load_skill_memory_path") or cfg.extra.get("load_registry_path")
    if not path or not hasattr(generator, "set_guidance"):
        return
    method = create_method(cfg.method)
    load_fn = getattr(method, "load_registry", None)
    wm_guidance_fn = getattr(method, "world_model_guidance", None)
    if not callable(load_fn) or not callable(wm_guidance_fn):
        return
    load_fn(path)
    generator.set_guidance(wm_guidance_fn())


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


def _mapping_valid_counts(candidates: list[InterventionCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {"true": 0, "false": 0, "missing": 0}
    for candidate in candidates:
        value = candidate.spec.metadata.get("llm_wm_mapping_valid")
        if value is True:
            counts["true"] += 1
        elif value is False:
            counts["false"] += 1
        else:
            counts["missing"] += 1
    return counts


def _selected_spec_id_count(candidates: list[InterventionCandidate]) -> int:
    return sum(
        1 for candidate in candidates if candidate.spec.metadata.get("llm_wm_selected_spec_id")
    )


if __name__ == "__main__":
    raise SystemExit(main())
