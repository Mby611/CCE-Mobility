from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    model: str = ""
    provider: str = "openai"
    base_url: str = ""
    chat_url: str = ""
    api_key_env: str = "LLM_API_KEY"
    temperature: float = 0.1
    max_tokens: int = 8192
    extra: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    name: str
    data_path: str = ""
    split: str = "validation"
    limit: int | None = None
    repo_path: str = ""
    extra: dict[str, Any] = Field(default_factory=dict)


class InterventionConfig(BaseModel):
    policy: str = "fixed"
    candidate_budget: int = 8
    selected_budget: int = 1
    dimensions: list[str] = Field(default_factory=list)
    severity_grid: list[float] = Field(default_factory=lambda: [0.5, 1.0, 1.5])
    validity_threshold: float = 0.7
    frontier_low: float = 0.4
    frontier_high: float = 0.6


class ExperimentConfig(BaseModel):
    experiment_name: str = "cce_smoke"
    output_dir: str = "runs"
    benchmark: BenchmarkConfig
    agent: ModelConfig = Field(default_factory=ModelConfig)
    world_model: ModelConfig = Field(default_factory=ModelConfig)
    judge: ModelConfig = Field(default_factory=ModelConfig)
    method: str = "no_evolution"
    iterations: int = 1
    intervention: InterventionConfig = Field(default_factory=InterventionConfig)
    seed: int = 42
    extra: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: Path) -> "ExperimentConfig":
        if path.suffix.lower() == ".json":
            return cls.from_json(path)
        try:
            import yaml  # type: ignore
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "PyYAML is required for YAML configs. Use a .json config or install pyyaml."
            ) from exc
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        with path.open("r", encoding="utf-8") as f:
            return cls.model_validate(json.load(f))
