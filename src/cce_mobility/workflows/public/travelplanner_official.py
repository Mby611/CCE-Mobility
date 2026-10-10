from __future__ import annotations

import argparse
import ast
import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from .artifacts import artifact, write_locked


def normalize_query(row: dict) -> dict:
    row = dict(row)
    query = row["query"]
    lower = query.lower()
    row["days"] = int(row["days"])
    words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    people = re.findall(r"\b(\d+)\s+(?:people|persons|travelers|travellers|individuals)\b", lower)
    inferred_people = (
        int(people[-1])
        if people
        else next(
            (
                n
                for word, n in words.items()
                if re.search(
                    rf"\b{word}\s+(people|persons|travelers|travellers|individuals)\b", lower
                )
            ),
            1,
        )
    )
    money = re.findall(r"\$([0-9,]+)", query)
    cities = re.findall(r"\b(?:visit|visiting|traversing|covering)\s+(\d+)\s+cities\b", lower)
    inferred_cities = (
        int(cities[-1])
        if cities
        else next(
            (
                n
                for word, n in words.items()
                if re.search(
                    rf"\b(?:visit|visiting|traversing|covering)\s+{word}\s+cities\b", lower
                )
            ),
            {3: 1, 5: 2, 7: 3}.get(row["days"], 1),
        )
    )
    for key, default in (
        ("people_number", inferred_people),
        ("budget", int(money[-1].replace(",", "")) if money else 10**9),
        ("visiting_city_number", inferred_cities),
    ):
        row[key] = int(row.get(key) or default)
    constraint = row.get("local_constraint")
    if constraint:
        row["local_constraint"] = (
            ast.literal_eval(constraint) if isinstance(constraint, str) else constraint
        )
        return row
    constraint = {"house rule": None, "cuisine": None, "room type": None, "transportation": None}
    if any(text in lower for text in ("no flight", "no flights", "avoid flights")):
        constraint["transportation"] = "no flight"
    elif any(text in lower for text in ("no self-driving", "avoid self-driving")):
        constraint["transportation"] = "no self-driving"
    for phrase, value in (
        ("not shared rooms", "not shared room"),
        ("not shared room", "not shared room"),
        ("shared rooms", "shared room"),
        ("shared room", "shared room"),
        ("private rooms", "private room"),
        ("private room", "private room"),
        ("entire rooms", "entire room"),
        ("entire room", "entire room"),
    ):
        if phrase in lower:
            constraint["room type"] = value
            break
    for phrase, value in (
        ("allow pets", "pets"),
        ("pets are allowed", "pets"),
        ("allow visitors", "visitors"),
        ("visitors are allowed", "visitors"),
        ("allow parties", "parties"),
        ("parties are allowed", "parties"),
        ("allow smoking", "smoking"),
        ("smoking is allowed", "smoking"),
        ("allow children under 10", "children under 10"),
        ("children under 10 are allowed", "children under 10"),
    ):
        if phrase in lower:
            constraint["house rule"] = value
            break
    cuisines = [
        name
        for name in (
            "Chinese",
            "Indian",
            "Italian",
            "Mexican",
            "French",
            "Mediterranean",
            "American",
        )
        if name.lower() in lower
    ]
    if cuisines:
        constraint["cuisine"] = cuisines
    if "no specific constraints" in lower or "no specific requirements" in lower:
        constraint = dict.fromkeys(constraint)
    row["local_constraint"] = constraint
    return row


def patch_query_loader(source: str) -> ast.Module:
    tree = ast.parse(source)
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and node.module == "datasets")
    ]
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "eval_score"
    ]
    if len(functions) != 1:
        raise ValueError("Unsupported upstream evaluator: missing unique eval_score")
    function = functions[0]
    body = []
    removed = 0
    for node in function.body:
        is_loader_if = isinstance(node, ast.If) and any(
            isinstance(child, ast.Name) and child.id == "load_dataset" for child in ast.walk(node)
        )
        is_loader_assign = isinstance(node, ast.Assign) and any(
            isinstance(child, ast.Name) and child.id == "_load_query_data_list"
            for child in ast.walk(node.value)
        )
        if is_loader_if or is_loader_assign:
            removed += 1
            continue
        body.append(node)
    if removed != 1:
        raise ValueError(
            "Unsupported upstream evaluator loader layout; refusing an ambiguous patch"
        )
    function.body = ast.parse("query_data_list = __cce_query_rows").body + body

    # Upstream test denominators are fixed at 1000 tasks. Use the same checks
    # with explicit subset denominators for reader-configured smoke runs.
    class Denominators(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Subscript):
                target = node.targets[0]
                if (
                    isinstance(target.value, ast.Name)
                    and target.value.id == "result"
                    and isinstance(node.value, ast.BinOp)
                    and isinstance(node.value.op, ast.Div)
                ):
                    name = target.slice.value if isinstance(target.slice, ast.Constant) else ""
                    node.value.right = ast.parse(
                        "__cce_cs_count"
                        if "Commonsense Constraint Micro" in name
                        else "__cce_hard_count"
                        if "Hard Constraint Micro" in name
                        else "__cce_task_count",
                        mode="eval",
                    ).body
            return self.generic_visit(node)

    tree = Denominators().visit(tree)
    return ast.fix_missing_locations(tree)


def evaluate(config: dict, records: Path, output: Path) -> dict:
    from cce_mobility.workflows.travelplanner.export_travelplanner_official_eval import (
        _first_factual_plan_by_index,
        _load_jsonl,
    )

    repo = Path(config["benchmark"]["repo_path"])
    data = Path(config["benchmark"]["test_data_path"]).parent
    if not (repo / "database/flights/clean_Flights_2022.csv").exists():
        raise FileNotFoundError(
            "Download the official TravelPlanner database before evaluate-official; see docs/reproducibility/travelplanner.md"
        )
    if not (data / "test.csv").exists():
        raise FileNotFoundError("Official test.csv is required for factual constraint evaluation")
    output.mkdir(parents=True, exist_ok=True)
    plans = _first_factual_plan_by_index(_load_jsonl(records))
    count = config["test"]["sample_count"]
    if set(plans) != set(range(count)):
        raise ValueError("Official factual records must cover every configured test index")
    exported = output / "factual_plans.jsonl"
    exported.write_text(
        "".join(json.dumps(plans[i]) + "\n" for i in range(count)), encoding="utf-8"
    )
    with (output / "evaluator.log").open("w") as handle:
        subprocess.run(
            [
                sys.executable,
                "-m",
                __name__,
                "--repo",
                str(repo),
                "--data",
                str(data),
                "--plans",
                str(exported),
                "--limit",
                str(count),
                "--output",
                str(output / "metrics.json"),
            ],
            env=os.environ.copy(),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=True,
        )
    manifest = {
        "scope": "offline compatibility evaluation using unchanged upstream constraint functions; not leaderboard submission",
        "query_labels": "official columns when available, otherwise deterministic query-derived compatibility fields",
        "denominators": "configured tasks, eight commonsense checks per task, budget plus applicable local constraints",
        "records": artifact(records),
        "plans": artifact(exported),
        "upstream_evaluator": artifact(repo / "evaluation/eval.py"),
        "compatibility_adapter": artifact(Path(__file__)),
        "configured_tasks": count,
    }
    write_locked(output / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--plans", type=Path, required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.plans = args.plans.resolve()
    args.output = args.output.resolve()
    with (args.data / "test.csv").open(encoding="utf-8", newline="") as handle:
        rows = [normalize_query(row) for row in csv.DictReader(handle)][: args.limit]
    if len(rows) != args.limit:
        raise ValueError("Official query coverage differs from requested evaluation size")
    evaluation_dir = args.repo.resolve() / "evaluation"
    script = evaluation_dir / "eval.py"
    tree = patch_query_loader(script.read_text())
    sys.path.insert(0, str(evaluation_dir))
    sys.path.insert(0, str(args.repo.resolve()))
    os.chdir(evaluation_dir)
    hard_count = sum(
        1 + sum(value is not None for value in row["local_constraint"].values()) for row in rows
    )
    namespace = {
        "__name__": "cce_upstream_evaluator",
        "__file__": str(script),
        "__cce_query_rows": rows,
        "__cce_task_count": len(rows),
        "__cce_cs_count": 8 * len(rows),
        "__cce_hard_count": hard_count,
    }
    exec(compile(tree, str(script), "exec"), namespace)
    scores, details = namespace["eval_score"]("test", str(args.plans.resolve()))
    args.output.resolve().write_text(
        json.dumps({"scores": scores, "details": details}, default=str, indent=2) + "\n"
    )
    print(json.dumps(scores, default=str, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
