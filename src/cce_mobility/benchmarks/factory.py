from __future__ import annotations

from cce_mobility.benchmarks.travelplanner import TravelPlannerAdapter
from cce_mobility.core.config import BenchmarkConfig
from cce_mobility.core.protocols import BenchmarkAdapter


def create_benchmark(config: BenchmarkConfig) -> BenchmarkAdapter:
    name = config.name.lower()
    if name in {"travelplanner", "tp"}:
        return TravelPlannerAdapter(config)
    if name in {"travelplanner_react", "tp_react"}:
        from cce_mobility.benchmarks.travelplanner_react import TravelPlannerReactAdapter

        return TravelPlannerReactAdapter(config)
    if name in {"travelbench", "tb"}:
        from cce_mobility.benchmarks.observable import TravelBenchAdapter

        return TravelBenchAdapter(config)
    if name in {"mobilitybench", "mb"}:
        from cce_mobility.benchmarks.observable import MobilityBenchAdapter

        return MobilityBenchAdapter(config)
    raise ValueError(f"Unknown benchmark adapter: {config.name}")
