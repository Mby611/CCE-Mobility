from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


LOWER_IS_BETTER = {
    ("efficiency", "IT"),
    ("efficiency", "OT"),
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare two official MobilityBench evaluation_results.json files "
            "at overall and by-intent-family levels."
        )
    )
    parser.add_argument("--baseline", required=True, help="NAME=PATH for baseline eval JSON.")
    parser.add_argument("--target", required=True, help="NAME=PATH for target eval JSON.")
    parser.add_argument("--output-json", default="", help="Optional output JSON path.")
    parser.add_argument("--output-md", default="", help="Optional output Markdown path.")
    parser.add_argument("--output-csv", default="", help="Optional flat CSV output path.")
    args = parser.parse_args()

    baseline_name, baseline_path = _split_named_path(args.baseline)
    target_name, target_path = _split_named_path(args.target)
    baseline = _load_eval(Path(baseline_path))
    target = _load_eval(Path(target_path))

    comparison = {
        "baseline": {"name": baseline_name, "path": baseline_path},
        "target": {"name": target_name, "path": target_path},
        "overall": _compare_scope(
            baseline.get("overall", {}),
            target.get("overall", {}),
            scope="overall",
        ),
        "by_intent_family": {},
        "metric_note": (
            "Positive delta means target - baseline. For efficiency IT/OT, "
            "negative delta is treated as improvement."
        ),
    }
    families = sorted(
        set(baseline.get("by_intent_family", {})) | set(target.get("by_intent_family", {}))
    )
    for family in families:
        comparison["by_intent_family"][family] = _compare_scope(
            baseline.get("by_intent_family", {}).get(family, {}),
            target.get("by_intent_family", {}).get(family, {}),
            scope=family,
        )

    text = json.dumps(comparison, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        output = Path(args.output_json)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    if args.output_md:
        output = Path(args.output_md)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(_render_markdown(comparison), encoding="utf-8")
    if args.output_csv:
        _write_csv(Path(args.output_csv), comparison)
    return 0


def _split_named_path(item: str) -> tuple[str, str]:
    if "=" not in item:
        path = Path(item)
        return path.parent.name or path.stem, item
    name, path = item.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"invalid NAME=PATH argument: {item}")
    return name, path


def _load_eval(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "overall": _extract_overall(data),
        "by_intent_family": _extract_by_intent_family(data),
    }


def _extract_overall(data: dict[str, Any]) -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    for metric, models in data.items():
        if not isinstance(models, dict):
            continue
        for payload in models.values():
            summary = payload.get("summary") if isinstance(payload, dict) else None
            sub_scores = (summary or {}).get("sub_scores") or {}
            for sub_dim, value in sub_scores.items():
                numeric = _as_float(value)
                if numeric is not None:
                    rows.setdefault(metric, {})[sub_dim] = numeric
    return rows


def _extract_by_intent_family(
    data: dict[str, Any],
) -> dict[str, dict[str, dict[str, float]]]:
    rows: dict[str, dict[str, dict[str, float]]] = {}
    for metric, models in data.items():
        if not isinstance(models, dict):
            continue
        for payload in models.values():
            by_family = payload.get("by_intent_family") if isinstance(payload, dict) else None
            if not isinstance(by_family, dict):
                continue
            for family, family_payload in by_family.items():
                sub_scores = (family_payload or {}).get("sub_scores") or {}
                for sub_dim, value in sub_scores.items():
                    numeric = _as_float(value)
                    if numeric is not None:
                        rows.setdefault(family, {}).setdefault(metric, {})[sub_dim] = numeric
    return rows


def _compare_scope(
    baseline: dict[str, dict[str, float]],
    target: dict[str, dict[str, float]],
    *,
    scope: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metric in sorted(set(baseline) | set(target)):
        sub_dims = sorted(set(baseline.get(metric, {})) | set(target.get(metric, {})))
        for sub_dim in sub_dims:
            base_value = baseline.get(metric, {}).get(sub_dim)
            target_value = target.get(metric, {}).get(sub_dim)
            delta = (
                round(target_value - base_value, 6)
                if base_value is not None and target_value is not None
                else None
            )
            rows.append(
                {
                    "scope": scope,
                    "metric": metric,
                    "sub_dimension": sub_dim,
                    "baseline": _round(base_value),
                    "target": _round(target_value),
                    "delta": delta,
                    "preferred_direction": _preferred_direction(metric, sub_dim),
                    "improved": _is_improved(metric, sub_dim, delta),
                }
            )
    return rows


def _preferred_direction(metric: str, sub_dim: str) -> str:
    return "lower" if (metric, sub_dim) in LOWER_IS_BETTER else "higher"


def _is_improved(metric: str, sub_dim: str, delta: float | None) -> bool | None:
    if delta is None:
        return None
    if (metric, sub_dim) in LOWER_IS_BETTER:
        return delta < 0
    return delta > 0


def _render_markdown(comparison: dict[str, Any]) -> str:
    baseline = comparison["baseline"]["name"]
    target = comparison["target"]["name"]
    lines = [
        "# MobilityBench Official Metric Comparison",
        "",
        f"- Baseline: `{baseline}`",
        f"- Target: `{target}`",
        "",
        comparison["metric_note"],
        "",
        "## Overall",
        "",
    ]
    lines.extend(_render_table(comparison["overall"]))
    by_family = comparison.get("by_intent_family") or {}
    if by_family:
        lines.extend(["", "## By Intent Family", ""])
        for family, rows in by_family.items():
            lines.extend([f"### {family}", ""])
            lines.extend(_render_table(rows))
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| metric | sub_dimension | baseline | target | delta | preferred | improved |",
        "| --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {metric} | {sub_dimension} | {baseline} | {target} | {delta} | "
            "{preferred_direction} | {improved} |".format(**row)
        )
    return lines


def _write_csv(path: Path, comparison: dict[str, Any]) -> None:
    rows = list(comparison["overall"])
    for family_rows in (comparison.get("by_intent_family") or {}).values():
        rows.extend(family_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "metric",
                "sub_dimension",
                "baseline",
                "target",
                "delta",
                "preferred_direction",
                "improved",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _round(value: float | None) -> float | None:
    return round(value, 6) if isinstance(value, float) else value


if __name__ == "__main__":
    raise SystemExit(main())
