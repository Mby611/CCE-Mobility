from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

from cce_mobility.benchmarks import create_benchmark
from cce_mobility.core.capabilities import infer_failure_types
from cce_mobility.core.config import ExperimentConfig
from cce_mobility.core.llm import ChatClient
from cce_mobility.core.metrics import add_experiment_coverage_metrics, aggregate_records
from cce_mobility.core.recovery_context import (
    append_factual_context_artifact,
    build_factual_context_artifact,
    build_factual_recovery_context,
    load_factual_context_artifacts,
    validate_factual_context_artifact,
)
from cce_mobility.core.schema import (
    AgentTrajectory,
    EvaluationRecord,
    ExperimentReport,
    FactualContextArtifact,
    FactualRecoveryContext,
    InterventionSpec,
    TaskSample,
)
from cce_mobility.interventions.analyzer import HeuristicTrajectoryAnalyzer
from cce_mobility.interventions.frontier import FrontierSelector
from cce_mobility.interventions.generator import (
    FrozenInterventionGenerator,
    GridInterventionGenerator,
    LLMInterventionGenerator,
    OntologyAwareLLMInterventionGenerator,
)
from cce_mobility.interventions.predictor import HeuristicWMPredictiveCritic
from cce_mobility.interventions.spec_builder import RuleSpecBuilder
from cce_mobility.interventions.travelplanner_spec import TravelPlannerSpecBuilder
from cce_mobility.interventions.validator import HeuristicInterventionValidator
from cce_mobility.judges.recovery import HeuristicRecoveryJudge
from cce_mobility.judges.spec_llm import SpecGroundedLLMJudge
from cce_mobility.methods.factory import create_method
from cce_mobility.skills.retrieval import sanitize_agent_visible_text


class EvaluationRunner:
    """Run CCE evaluation or light co-evolution loops."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.output_dir = Path(config.output_dir) / config.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.factual_context_artifact_path = self.output_dir / "factual_contexts.jsonl"
        frozen_context_path = str(config.extra.get("frozen_factual_context_path") or "").strip()
        self.frozen_factual_context_path = (
            Path(frozen_context_path).expanduser().resolve() if frozen_context_path else None
        )
        if (
            self.frozen_factual_context_path is not None
            and self.frozen_factual_context_path == self.factual_context_artifact_path.resolve()
        ):
            raise ValueError(
                "frozen_factual_context_path must differ from the current run's "
                "factual_contexts.jsonl output"
            )
        self._frozen_factual_artifacts = (
            load_factual_context_artifacts(self.frozen_factual_context_path)
            if self.frozen_factual_context_path is not None
            else {}
        )
        self._output_factual_artifacts: dict[tuple[int, str], FactualContextArtifact] = {}
        self._resume_retryable_record_count = 0
        provenance = config.extra.get("record_transport_provenance") or {}
        self.record_transport_provenance = dict(provenance) if isinstance(provenance, dict) else {}

        self.benchmark = create_benchmark(config.benchmark)
        runtime_options_fn = getattr(self.benchmark, "set_runtime_options", None)
        if callable(runtime_options_fn):
            runtime_options_fn(config.extra)
        self.agent_client = ChatClient(config.agent)
        self.world_model_client = ChatClient(config.world_model)
        self.judge_client = ChatClient(config.judge)
        if hasattr(self.benchmark, "set_chat_client"):
            self.benchmark.set_chat_client(self.agent_client)
        self.method = create_method(config.method)
        self.base_agent_guidance = self._load_base_agent_guidance()
        if self.base_agent_guidance and callable(runtime_options_fn):
            runtime_options_fn({"base_agent_guidance": self.base_agent_guidance})
        self._configure_method()
        self._load_method_memory()
        self._maybe_freeze_method()
        self.analyzer = HeuristicTrajectoryAnalyzer()
        grid_generator = GridInterventionGenerator(
            dimensions=config.intervention.dimensions or None,
            severity_grid=config.intervention.severity_grid,
        )
        if config.intervention.policy == "frozen":
            frozen_path = config.extra.get("frozen_intervention_path", "")
            self.generator = FrozenInterventionGenerator(frozen_path)
        elif config.intervention.policy in {
            "ontology",
            "ontology_discovery",
            "llm_ontology",
            "ontology_frontier",
        }:
            self.generator = OntologyAwareLLMInterventionGenerator(
                self.world_model_client,
                dimensions=config.intervention.dimensions or None,
                severity_grid=config.intervention.severity_grid,
                fallback=grid_generator,
            )
        elif config.intervention.policy in {"llm", "llm_frontier", "world_model"}:
            self.generator = LLMInterventionGenerator(
                self.world_model_client,
                dimensions=config.intervention.dimensions or None,
                severity_grid=config.intervention.severity_grid,
                fallback=grid_generator,
            )
        else:
            self.generator = grid_generator
        seed_ontology_fn = getattr(self.method, "seed_intervention_ontology", None)
        if callable(seed_ontology_fn):
            seed_ontology_fn(config.intervention.dimensions or None)
        self.spec_builder = (
            TravelPlannerSpecBuilder()
            if config.benchmark.name.lower() in {"travelplanner", "tp"}
            else RuleSpecBuilder()
        )
        self.validator = HeuristicInterventionValidator(config.intervention.validity_threshold)
        self.selector = FrontierSelector(
            low=config.intervention.frontier_low,
            high=config.intervention.frontier_high,
        )
        self.wm_predictor = HeuristicWMPredictiveCritic()
        self.judge = (
            HeuristicRecoveryJudge()
            if config.judge.provider in {"local", "heuristic", "placeholder"}
            else SpecGroundedLLMJudge(self.judge_client)
        )

    def run(self) -> ExperimentReport:
        samples = self.benchmark.load_samples(limit=self.config.benchmark.limit)
        records_path = self.output_dir / "records.jsonl"
        resume = bool(self.config.extra.get("resume", False))
        all_records = self._read_jsonl(records_path) if resume and records_path.exists() else []
        if resume and all_records:
            retryable = [record for record in all_records if self._record_has_llm_error(record)]
            if retryable:
                self._resume_retryable_record_count = len(retryable)
                archive_path = records_path.with_name("records_retryable_llm_errors.jsonl")
                with archive_path.open("a", encoding="utf-8") as handle:
                    for record in retryable:
                        handle.write(
                            json.dumps(record.model_dump(), ensure_ascii=False, default=str) + "\n"
                        )
                all_records = [
                    record for record in all_records if not self._record_has_llm_error(record)
                ]
                records_path.write_text(
                    "".join(
                        json.dumps(record.model_dump(), ensure_ascii=False, default=str) + "\n"
                        for record in all_records
                    ),
                    encoding="utf-8",
                )
        completed = self._completed_keys(all_records) if resume else set()

        self._preflight_frozen_factual_contexts(samples)
        if resume and self.factual_context_artifact_path.exists():
            self._output_factual_artifacts = load_factual_context_artifacts(
                self.factual_context_artifact_path
            )
        else:
            self.factual_context_artifact_path.write_text("", encoding="utf-8")
            self._output_factual_artifacts = {}

        if not resume:
            records_path.write_text("", encoding="utf-8")

        online_update_after_sample = bool(
            self.config.extra.get("online_update_after_sample", False)
        )
        skip_factual_judge = bool(self.config.extra.get("skip_factual_judge", False))

        for _iteration in range(1, self.config.iterations + 1):
            iteration_path = self.output_dir / f"iteration_{_iteration}.jsonl"
            if not resume:
                iteration_path.write_text("", encoding="utf-8")

            self._log(
                f"[{self.config.experiment_name}] iteration "
                f"{_iteration}/{self.config.iterations}: {len(samples)} samples"
            )
            self.method.before_iteration(all_records)
            self._apply_method_guidance()
            iter_records: list[EvaluationRecord] = []

            for sample_idx, sample in enumerate(samples, start=1):
                sample_records: list[EvaluationRecord] = []
                self._log(
                    f"[{self.config.experiment_name}] sample "
                    f"{sample_idx}/{len(samples)} {sample.sample_id}: factual"
                )
                factual, recovery_context, factual_context_metadata = self._resolve_factual_context(
                    iteration=_iteration,
                    sample=sample,
                    agent_id=self.method.agent_id(),
                    resume=resume,
                )
                factual_metrics = self._evaluate_trajectory(sample, factual, None)
                factual_record = EvaluationRecord(
                    sample=sample,
                    factual=factual,
                    metadata={
                        "factual_hard_metrics": factual_metrics,
                        **factual_context_metadata,
                        **self._record_transport_metadata(),
                    },
                )
                factual_score = None if skip_factual_judge else self.judge.score(factual_record)
                points = self.analyzer.find_points(sample, factual)
                candidates = self.generator.generate(sample, factual, points)

                prepared = []
                for cand in candidates[: self.config.intervention.candidate_budget]:
                    cand.recovery_spec = cand.recovery_spec or self.spec_builder.build(cand, sample)
                    cand = self.validator.validate(cand, sample)
                    if self.validator.is_valid(cand):
                        prepared.append(cand)

                selected = self.selector.select(
                    prepared,
                    budget=self.config.intervention.selected_budget,
                )
                self._log(
                    f"[{self.config.experiment_name}] sample "
                    f"{sample_idx}/{len(samples)} {sample.sample_id}: "
                    f"{len(selected)} counterfactual(s)"
                )
                if not selected:
                    record_key = self._record_key(_iteration, sample.sample_id, None)
                    if record_key in completed:
                        self._log(
                            f"[{self.config.experiment_name}] sample "
                            f"{sample_idx}/{len(samples)} {sample.sample_id}: skip completed factual"
                        )
                        continue
                    factual_record.factual_score = factual_score
                    factual_record.metadata["iteration"] = _iteration
                    iter_records.append(factual_record)
                    sample_records.append(factual_record)
                    all_records.append(factual_record)
                    self._append_jsonl(records_path, factual_record)
                    self._append_jsonl(iteration_path, factual_record)
                    if online_update_after_sample:
                        self.method.observe(sample_records)
                        self._apply_method_guidance()
                    continue

                for cf_idx, cand in enumerate(selected, start=1):
                    record_key = self._record_key(
                        _iteration,
                        sample.sample_id,
                        cand.spec.intervention_id,
                    )
                    if record_key in completed:
                        self._log(
                            f"[{self.config.experiment_name}] sample "
                            f"{sample_idx}/{len(samples)} {sample.sample_id}: "
                            f"skip completed {cand.spec.dimension}"
                        )
                        continue
                    self._log(
                        f"[{self.config.experiment_name}] sample "
                        f"{sample_idx}/{len(samples)} {sample.sample_id}: "
                        f"counterfactual {cf_idx}/{len(selected)} {cand.spec.dimension}"
                    )
                    wm_prediction = self.wm_predictor.predict(
                        sample=sample,
                        factual=factual,
                        candidate=cand,
                        world_model_guidance=self._world_model_guidance(),
                    )
                    cf = self._run_counterfactual_agent(
                        sample,
                        agent_id=self.method.agent_id(),
                        intervention=cand.spec,
                        recovery_context=recovery_context,
                    )
                    # The benchmark may know the exact records affected by the
                    # intervention only after execution. Feed them back into
                    # the spec for judge/report grounding.
                    if cf.raw.get("intervention_records"):
                        cand.spec.metadata["intervention_records"] = cf.raw["intervention_records"]
                    if cf.raw.get("unavailable_ids"):
                        cand.spec.metadata["unavailable_ids"] = cf.raw["unavailable_ids"]
                        cand.spec.target_entity = {
                            **cand.spec.target_entity,
                            "unavailable_ids": cf.raw["unavailable_ids"],
                        }
                    if cf.raw.get("intervention_records") or cf.raw.get("unavailable_ids"):
                        cand.recovery_spec = self.spec_builder.build(cand, sample)
                    counterfactual_metrics = self._evaluate_trajectory(sample, cf, cand.spec)
                    record = EvaluationRecord(
                        sample=sample,
                        factual=factual,
                        counterfactual=cf,
                        intervention=cand.spec,
                        recovery_spec=cand.recovery_spec,
                        wm_prediction=wm_prediction,
                        metadata={
                            "intervention_validity": cand.validity_score,
                            "intervention_utility": cand.utility,
                            "validation_notes": cand.validation_notes,
                            "factual_hard_metrics": factual_metrics,
                            "counterfactual_hard_metrics": counterfactual_metrics,
                            **factual_context_metadata,
                            **self._record_transport_metadata(),
                            "iteration": _iteration,
                        },
                    )
                    record.factual_score = factual_score
                    record.counterfactual_score = self.judge.score(record)
                    record.metadata["wm_prediction_metrics"] = _prediction_metrics(record)
                    iter_records.append(record)
                    sample_records.append(record)
                    all_records.append(record)
                    self._append_jsonl(records_path, record)
                    self._append_jsonl(iteration_path, record)

                if online_update_after_sample and sample_records:
                    self.method.observe(sample_records)
                    self._apply_method_guidance()

            if not online_update_after_sample:
                self.method.observe(iter_records)
            self._save_method_memory_checkpoint(_iteration)
            should_stop_fn = getattr(self.method, "should_stop", None)
            if callable(should_stop_fn) and should_stop_fn():
                self._log(
                    f"[{self.config.experiment_name}] method requested early stop "
                    f"after iteration {_iteration}"
                )
                break

        metrics = aggregate_records(all_records)
        add_experiment_coverage_metrics(
            metrics,
            sample_count=len(samples),
            iterations=self.config.iterations,
            selected_budget=self.config.intervention.selected_budget,
        )
        metrics.update(
            {
                "agent_model": self.config.agent.model,
                "agent_provider": self.config.agent.provider,
                "world_model": self.config.world_model.model,
                "world_model_provider": self.config.world_model.provider,
                "judge_model": self.config.judge.model,
                "judge_provider": self.config.judge.provider,
                "factual_judge_skipped": int(skip_factual_judge),
                "resume_retryable_llm_error_record_count": self._resume_retryable_record_count,
                **self._record_transport_metrics(all_records),
            }
        )
        for role, client in (
            ("agent", self.agent_client),
            ("world_model", self.world_model_client),
            ("judge", self.judge_client),
        ):
            metrics.update(
                {
                    f"{role}_transport_{key}": value
                    for key, value in client.transport_metrics().items()
                }
            )
        method_metrics_fn = getattr(self.method, "report_metrics", None)
        if callable(method_metrics_fn):
            metrics.update(method_metrics_fn())
        if self.base_agent_guidance:
            metrics.update(
                {
                    "base_agent_guidance_loaded": 1,
                    "base_agent_guidance_chars": len(self.base_agent_guidance),
                }
            )
        self._save_method_memory_final()
        report = ExperimentReport(
            experiment_name=self.config.experiment_name,
            benchmark=self.config.benchmark.name,
            method=self.config.method,
            total_samples=len(samples),
            metrics=metrics,
            records_path=str(records_path),
        )
        (self.output_dir / "report.json").write_text(
            report.model_dump_json(indent=2),
            encoding="utf-8",
        )
        return report

    @staticmethod
    def _write_jsonl(path: Path, records: list[EvaluationRecord]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record.model_dump(), ensure_ascii=False, default=str) + "\n")

    @staticmethod
    def _append_jsonl(path: Path, record: EvaluationRecord) -> None:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record.model_dump(), ensure_ascii=False, default=str) + "\n")

    def _record_transport_metadata(self) -> dict:
        if not self.record_transport_provenance:
            return {}
        return {"transport_provenance": dict(self.record_transport_provenance)}

    @staticmethod
    def _record_transport_metrics(records: list[EvaluationRecord]) -> dict:
        counts: dict[str, int] = {}
        unlabeled = 0
        for record in records:
            provenance = record.metadata.get("transport_provenance")
            if not isinstance(provenance, dict):
                unlabeled += 1
                continue
            label = str(provenance.get("label") or "unlabeled").strip() or "unlabeled"
            counts[label] = counts.get(label, 0) + 1
        return {
            "record_transport_provenance_counts": counts,
            "record_transport_unlabeled_count": unlabeled,
        }

    @staticmethod
    def _read_jsonl(path: Path) -> list[EvaluationRecord]:
        records: list[EvaluationRecord] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(EvaluationRecord.model_validate(json.loads(line)))
        return records

    @classmethod
    def _completed_keys(cls, records: list[EvaluationRecord]) -> set[tuple[int, str, str]]:
        return {
            cls._record_key(
                int(record.metadata.get("iteration") or 1),
                record.sample.sample_id,
                record.intervention.intervention_id if record.intervention else None,
            )
            for record in records
            if not cls._record_has_llm_error(record)
        }

    @staticmethod
    def _record_has_llm_error(record: EvaluationRecord) -> bool:
        for trajectory in (record.factual, record.counterfactual):
            if trajectory is None:
                continue
            raw = trajectory.raw or {}
            if bool(raw.get("llm_error")):
                return True
            if str(trajectory.final_response or "").startswith("[LLM_ERROR]"):
                return True
        return False

    @staticmethod
    def _record_key(
        iteration: int,
        sample_id: str,
        intervention_id: str | None,
    ) -> tuple[int, str, str]:
        return (iteration, sample_id, intervention_id or "no_intervention")

    def _evaluate_trajectory(
        self,
        sample,
        trajectory,
        intervention,
    ) -> dict:
        if not hasattr(self.benchmark, "evaluate_trajectory"):
            return {}
        return self.benchmark.evaluate_trajectory(sample, trajectory, intervention)

    def _run_counterfactual_agent(
        self,
        sample: TaskSample,
        *,
        agent_id: str,
        intervention: InterventionSpec,
        recovery_context: FactualRecoveryContext,
    ) -> AgentTrajectory:
        """Pass the factual prefix when supported without breaking legacy adapters."""

        run_agent = self.benchmark.run_agent
        try:
            parameters = inspect.signature(run_agent).parameters.values()
            accepts_context = any(
                parameter.name == "recovery_context"
                or parameter.kind == inspect.Parameter.VAR_KEYWORD
                for parameter in parameters
            )
        except (TypeError, ValueError):
            accepts_context = False
        if accepts_context:
            return run_agent(
                sample,
                agent_id=agent_id,
                intervention=intervention,
                recovery_context=recovery_context,
            )
        return run_agent(sample, agent_id=agent_id, intervention=intervention)

    def _preflight_frozen_factual_contexts(self, samples: list[TaskSample]) -> None:
        if self.frozen_factual_context_path is None:
            return
        for iteration in range(1, self.config.iterations + 1):
            for sample in samples:
                key = (iteration, sample.sample_id)
                artifact = self._frozen_factual_artifacts.get(key)
                if artifact is None:
                    raise ValueError(
                        "Missing frozen factual context for "
                        f"iteration={iteration}, sample_id={sample.sample_id!r} in "
                        f"{self.frozen_factual_context_path}"
                    )
                validate_factual_context_artifact(
                    artifact,
                    sample=sample,
                    iteration=iteration,
                )

    def _resolve_factual_context(
        self,
        *,
        iteration: int,
        sample: TaskSample,
        agent_id: str,
        resume: bool,
    ) -> tuple[AgentTrajectory, FactualRecoveryContext, dict]:
        key = (iteration, sample.sample_id)
        if self.frozen_factual_context_path is not None:
            artifact = self._frozen_factual_artifacts[key]
            source = "frozen_artifact"
            source_path = self.frozen_factual_context_path
        elif resume and key in self._output_factual_artifacts:
            artifact = self._output_factual_artifacts[key]
            validate_factual_context_artifact(
                artifact,
                sample=sample,
                iteration=iteration,
            )
            source = "resume_artifact"
            source_path = self.factual_context_artifact_path.resolve()
        else:
            factual = self.benchmark.run_agent(
                sample,
                agent_id=agent_id,
                intervention=None,
            )
            recovery_context = build_factual_recovery_context(sample, factual)
            artifact = build_factual_context_artifact(
                iteration=iteration,
                sample=sample,
                factual=factual,
                recovery_context=recovery_context,
            )
            source = "generated"
            source_path = self.factual_context_artifact_path.resolve()

        self._persist_factual_context_artifact(key, artifact)
        context = artifact.recovery_context
        metadata = {
            "factual_recovery_context": context.model_dump(mode="json"),
            "factual_prefix_hash": context.prefix_hash,
            "factual_context_artifact_hash": artifact.artifact_hash,
            "factual_context_source": source,
            "factual_context_source_path": str(source_path),
            "factual_context_artifact_path": str(self.factual_context_artifact_path.resolve()),
        }
        return artifact.factual.model_copy(deep=True), context.model_copy(deep=True), metadata

    def _persist_factual_context_artifact(
        self,
        key: tuple[int, str],
        artifact: FactualContextArtifact,
    ) -> None:
        existing = self._output_factual_artifacts.get(key)
        if existing is not None:
            if existing.artifact_hash != artifact.artifact_hash:
                raise ValueError(
                    "Current run already contains a different factual context for "
                    f"iteration={key[0]}, sample_id={key[1]!r}"
                )
            return
        append_factual_context_artifact(self.factual_context_artifact_path, artifact)
        self._output_factual_artifacts[key] = artifact.model_copy(deep=True)

    @staticmethod
    def _log(message: str) -> None:
        print(message, file=sys.stderr, flush=True)

    def _apply_method_guidance(self) -> None:
        agent_guidance_fn = getattr(self.method, "agent_guidance", None)
        if callable(agent_guidance_fn) and hasattr(self.benchmark, "set_agent_guidance"):
            self.benchmark.set_agent_guidance(agent_guidance_fn())

        agent_registry_fn = getattr(self.method, "agent_skill_registry", None)
        if callable(agent_registry_fn) and hasattr(self.benchmark, "set_agent_skill_registry"):
            self.benchmark.set_agent_skill_registry(agent_registry_fn())

        wm_guidance_fn = getattr(self.method, "world_model_guidance", None)
        if callable(wm_guidance_fn) and hasattr(self.generator, "set_guidance"):
            self.generator.set_guidance(wm_guidance_fn())

        ontology_registry_fn = getattr(self.method, "intervention_ontology_registry", None)
        if callable(ontology_registry_fn) and hasattr(self.generator, "set_ontology_registry"):
            self.generator.set_ontology_registry(ontology_registry_fn())

    def _world_model_guidance(self) -> str:
        wm_guidance_fn = getattr(self.method, "world_model_guidance", None)
        return wm_guidance_fn() if callable(wm_guidance_fn) else ""

    def _load_method_memory(self) -> None:
        path = self.config.extra.get("load_skill_memory_path") or self.config.extra.get(
            "load_registry_path",
        )
        if not path:
            return
        memory_path = Path(path)
        if not memory_path.exists():
            raise FileNotFoundError(f"Configured skill/spec memory path not found: {memory_path}")
        load_fn = getattr(self.method, "load_registry", None)
        if not callable(load_fn):
            self._log(f"[{self.config.experiment_name}] method does not support loading memory")
            return
        load_fn(memory_path)
        self._log(f"[{self.config.experiment_name}] loaded skill/spec memory from {memory_path}")

    def _load_base_agent_guidance(self) -> str:
        texts: list[str] = []
        inline = str(self.config.extra.get("base_agent_guidance") or "").strip()
        if inline:
            texts.append(inline)
        path = str(self.config.extra.get("base_agent_guidance_path") or "").strip()
        if path:
            guidance_path = Path(path)
            if not guidance_path.exists():
                raise FileNotFoundError(
                    f"Configured base agent guidance path not found: {guidance_path}"
                )
            texts.append(guidance_path.read_text(encoding="utf-8"))
        if not texts:
            return ""
        return sanitize_agent_visible_text("\n\n".join(texts)).strip()

    def _configure_method(self) -> None:
        configure_fn = getattr(self.method, "configure", None)
        if callable(configure_fn):
            configure_fn(self.config.extra)

    def _maybe_freeze_method(self) -> None:
        if not bool(self.config.extra.get("freeze_method", False)):
            return
        freeze_fn = getattr(self.method, "freeze", None)
        if callable(freeze_fn):
            freeze_fn()
            self._log(f"[{self.config.experiment_name}] method memory frozen")

    def _save_method_memory_checkpoint(self, iteration: int) -> None:
        if not bool(self.config.extra.get("save_skill_memory", True)):
            return
        save_fn = getattr(self.method, "save_registry", None)
        if not callable(save_fn):
            save_fn = None
        if callable(save_fn):
            save_fn(self.output_dir / f"skill_memory_iteration_{iteration}.jsonl")
        extra_save_fn = getattr(self.method, "save_extra_artifacts", None)
        if callable(extra_save_fn):
            extra_save_fn(self.output_dir, iteration=iteration)

    def _save_method_memory_final(self) -> None:
        if not bool(self.config.extra.get("save_skill_memory", True)):
            return
        save_fn = getattr(self.method, "save_registry", None)
        if not callable(save_fn):
            save_fn = None
        if callable(save_fn):
            path = self.output_dir / "skill_memory_final.jsonl"
            save_fn(path)
        extra_save_fn = getattr(self.method, "save_extra_artifacts", None)
        if callable(extra_save_fn):
            extra_save_fn(self.output_dir, iteration=None)


def _prediction_metrics(record: EvaluationRecord) -> dict:
    if not record.wm_prediction or not record.counterfactual_score or not record.intervention:
        return {}
    actual_bin = _score_bin(record.counterfactual_score.total)
    hard = record.metadata.get("counterfactual_hard_metrics", {})
    actual_failures = infer_failure_types(
        dimension=record.intervention.dimension,
        judge_strategy=record.counterfactual_score.strategy.value,
        hard_metrics=hard,
        score_total=record.counterfactual_score.total,
    )
    predicted = set(record.wm_prediction.predicted_failure_modes)
    actual = set(actual_failures)
    overlap = predicted & actual
    return {
        "predicted_score_bin": record.wm_prediction.predicted_recovery_score_bin,
        "actual_score_bin": actual_bin,
        "score_bin_hit": record.wm_prediction.predicted_recovery_score_bin == actual_bin,
        "predicted_failure_modes": sorted(predicted),
        "actual_failure_modes": sorted(actual),
        "failure_mode_hit": bool(overlap),
        "failure_mode_precision": round(len(overlap) / len(predicted), 4) if predicted else 0.0,
        "failure_mode_recall": round(len(overlap) / len(actual), 4) if actual else 0.0,
        "confidence": record.wm_prediction.confidence,
    }


def _score_bin(score: float) -> str:
    if score < 2.0:
        return "low"
    if score < 3.0:
        return "medium"
    return "high"
