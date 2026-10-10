import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("src/cce_mobility/workflows/mobilitybench/compare_mobilitybench_official_metrics.py")


def _write_eval(path: Path, *, tool_precision: float, tokens: float) -> None:
    payload = {
        "tool": {
            "aihub-qwen35b": {
                "summary": {
                    "average_score": 0.0,
                    "total_cases": 2,
                    "failed_cases": 0,
                    "sub_scores": {
                        "TS_P": tool_precision,
                        "TS_R": 0.5,
                    },
                },
                "by_intent_family": {
                    "Basic Route Planning": {
                        "sub_scores": {"TS_P": tool_precision - 0.1},
                        "total_cases": 1,
                    }
                },
            }
        },
        "efficiency": {
            "aihub-qwen35b": {
                "summary": {
                    "average_score": 0.0,
                    "total_cases": 2,
                    "failed_cases": 0,
                    "sub_scores": {"IT": tokens},
                },
                "by_intent_family": {
                    "Basic Route Planning": {
                        "sub_scores": {"IT": tokens + 1},
                        "total_cases": 1,
                    }
                },
            }
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_compare_official_metrics_outputs_overall_and_by_intent(tmp_path):
    baseline = tmp_path / "baseline.json"
    target = tmp_path / "target.json"
    output_json = tmp_path / "comparison.json"
    output_md = tmp_path / "comparison.md"
    output_csv = tmp_path / "comparison.csv"
    _write_eval(baseline, tool_precision=0.8, tokens=100.0)
    _write_eval(target, tool_precision=0.9, tokens=90.0)

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--baseline",
            f"base={baseline}",
            "--target",
            f"target={target}",
            "--output-json",
            str(output_json),
            "--output-md",
            str(output_md),
            "--output-csv",
            str(output_csv),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "Basic Route Planning" in result.stdout
    data = json.loads(output_json.read_text(encoding="utf-8"))
    overall = {(row["metric"], row["sub_dimension"]): row for row in data["overall"]}
    assert overall[("tool", "TS_P")]["delta"] == 0.1
    assert overall[("tool", "TS_P")]["improved"] is True
    assert overall[("efficiency", "IT")]["delta"] == -10.0
    assert overall[("efficiency", "IT")]["improved"] is True
    assert "MobilityBench Official Metric Comparison" in output_md.read_text(encoding="utf-8")
    assert "Basic Route Planning" in output_csv.read_text(encoding="utf-8")
