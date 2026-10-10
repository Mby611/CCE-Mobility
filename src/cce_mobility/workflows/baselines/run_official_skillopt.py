from __future__ import annotations

import os
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cce_mobility.core.env import load_dotenv  # noqa: E402
from cce_mobility.core.llm import _get_secret  # noqa: E402

load_dotenv(PROJECT_ROOT / ".env")


def _bridge_model_keys() -> None:
    if os.environ.get("QWEN_CHAT_API_KEY") and os.environ.get("JUDGE_API_KEY"):
        return
    model_key = _get_secret("CCE_MODEL_API_KEY")
    judge_key = _get_secret("CCE_JUDGE_API_KEY")
    if model_key:
        os.environ.setdefault("QWEN_CHAT_API_KEY", model_key)
        os.environ.setdefault("OPTIMIZER_QWEN_CHAT_API_KEY", model_key)
        os.environ.setdefault("TARGET_QWEN_CHAT_API_KEY", model_key)
    if judge_key:
        os.environ.setdefault("JUDGE_API_KEY", judge_key)
        os.environ.setdefault("CCE_JUDGE_API_KEY", judge_key)


def _selected_environment(argv: list[str]) -> str:
    config_path = None
    for index, arg in enumerate(argv):
        if arg == "--config" and index + 1 < len(argv):
            config_path = argv[index + 1]
        elif arg.startswith("--config="):
            config_path = arg.partition("=")[2]
    if not config_path:
        raise ValueError("SkillOpt requires --config with env.name")

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    env_name = str(config.get("env", {}).get("name", ""))
    for arg in argv:
        if arg.startswith("env.name="):
            env_name = arg.partition("=")[2]
    if env_name not in {"cce_mobilitybench", "cce_travelplanner"}:
        raise ValueError(f"Unsupported SkillOpt environment: {env_name!r}")
    return env_name


def main() -> int:
    args = sys.argv[1:]
    skillopt_root = os.environ.get("SKILLOPT_ROOT", "")
    for index, arg in enumerate(args):
        if arg == "--skillopt-root" and index + 1 < len(args):
            skillopt_root = args[index + 1]
            args = args[:index] + args[index + 2 :]
            break
    if not skillopt_root:
        raise ValueError("Set SKILLOPT_ROOT or pass --skillopt-root PATH")
    root = Path(skillopt_root).expanduser().resolve()
    if not (root / "scripts" / "train.py").is_file():
        raise FileNotFoundError(f"SkillOpt trainer not found under: {root}")
    sys.path.insert(0, str(root))
    sys.argv = [sys.argv[0], *args]
    env_name = _selected_environment(args)

    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)
    _bridge_model_keys()
    from cce_mobility.integrations.skillopt_qwen_compat import (
        install_qwen38_explicit_no_thinking,
    )
    from skillopt.model import qwen_backend
    from scripts import train as official_train

    install_qwen38_explicit_no_thinking(qwen_backend)
    print("SkillOpt transport compatibility: explicit no-thinking for qwen3.8-max only")
    if env_name == "cce_mobilitybench":
        from cce_mobility.integrations.skillopt_mobilitybench import (
            CCEMobilityBenchSkillOptAdapter,
        )

        official_train._ENV_REGISTRY["cce_mobilitybench"] = CCEMobilityBenchSkillOptAdapter
    else:
        from cce_mobility.integrations.skillopt_travelplanner import (
            CCETravelPlannerSkillOptAdapter,
        )

        official_train._ENV_REGISTRY["cce_travelplanner"] = CCETravelPlannerSkillOptAdapter
    official_train.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
