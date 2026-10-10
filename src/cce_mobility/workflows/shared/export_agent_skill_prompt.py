from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.skills.prompting import (  # noqa: E402
    AGENT_RETRIEVAL_BOUNDARY_NONE,
    OBSERVABLE_MOBILITY_BOUNDARY,
    agent_self_retrieval_prompt,
)
from cce_mobility.skills.registry import SkillRegistry  # noqa: E402
from cce_mobility.skills.retrieval import sanitize_agent_visible_text  # noqa: E402
from cce_mobility.skills.schema import SkillType  # noqa: E402


FORBIDDEN_AGENT_VISIBLE_TERMS = (
    "counterfactual",
    "intervention",
    "evaluator",
    "judge",
    "benchmark",
    "posthoc",
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export agent-visible recovery skill guidance for official MobilityBench."
    )
    parser.add_argument("--memory-path", required=True, help="Skill memory JSONL path.")
    parser.add_argument("--output", required=True, help="Output markdown prompt path.")
    parser.add_argument("--limit", type=int, default=8, help="Maximum active agent skills.")
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow exporting an empty prompt for wiring tests.",
    )
    parser.add_argument(
        "--strict-agent-visible",
        action="store_true",
        help="Fail if evaluator-only terms remain after sanitization.",
    )
    parser.add_argument(
        "--boundary-mode",
        choices=[
            AGENT_RETRIEVAL_BOUNDARY_NONE,
            OBSERVABLE_MOBILITY_BOUNDARY,
        ],
        default=AGENT_RETRIEVAL_BOUNDARY_NONE,
        help=(
            "Optional agent-owned applicability boundary. "
            "Use observable_mobility for the official MobilityBench boundary ablation."
        ),
    )
    args = parser.parse_args()

    memory_path = Path(args.memory_path)
    output_path = Path(args.output)
    if not memory_path.exists():
        raise FileNotFoundError(f"skill memory not found: {memory_path}")
    registry = SkillRegistry.load_jsonl_filtered(
        memory_path,
        skill_types={SkillType.AGENT_RECOVERY},
    )
    active_count = len(registry.active(SkillType.AGENT_RECOVERY))
    if active_count == 0 and not args.allow_empty:
        raise ValueError(f"no active agent recovery skills found in: {memory_path}")
    prompt = sanitize_agent_visible_text(
        agent_self_retrieval_prompt(
            registry,
            limit=args.limit,
            boundary_mode=args.boundary_mode,
        )
    )
    if args.strict_agent_visible:
        lower = prompt.lower()
        leaked = [term for term in FORBIDDEN_AGENT_VISIBLE_TERMS if term in lower]
        if leaked:
            raise ValueError(f"agent-visible prompt leaked evaluator-only terms: {leaked}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(prompt.rstrip() + "\n", encoding="utf-8")

    summary = {
        "memory_path": str(memory_path),
        "output": str(output_path),
        "active_agent_skill_count": active_count,
        "exported_chars": len(prompt),
        "strict_agent_visible": bool(args.strict_agent_visible),
        "boundary_mode": args.boundary_mode,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
