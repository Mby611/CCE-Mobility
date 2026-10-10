from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import re
import sys


_MOBILITYBENCH_RATE_LIMITER = None
_MOBILITYBENCH_RATE_LIMITER_CONFIG = None
_MOBILITYBENCH_LOOP_SAFE_CLIENT_LOGGED = False


def _patch_mobilitybench_llm_manager() -> None:
    try:
        from langchain_openai import ChatOpenAI
        from mobility_bench.agent.roles import llm_manager
    except Exception:
        return

    def _create_llm(self, model_name: str):  # type: ignore[no-untyped-def]
        if not self.model_config:
            raise ValueError("Model config not set")

        base_url = self.model_config.base_url
        if base_url and not base_url.startswith(("http://", "https://")):
            base_url = f"https://{base_url}"

        extra_body = json.loads(os.getenv("CCE_MODEL_EXTRA_BODY", "{}"))
        if not isinstance(extra_body, dict):
            raise ValueError("CCE_MODEL_EXTRA_BODY must be a JSON object")
        force_no_thinking = os.getenv("CCE_MOBILITYBENCH_DISABLE_THINKING", "0") != "0"
        if force_no_thinking:
            extra_body["enable_thinking"] = False

        kwargs = {
            "base_url": base_url,
            "api_key": self.model_config.api_key,
            "model": model_name,
            "temperature": self.model_config.temperature,
            "max_tokens": self.model_config.max_tokens,
            "timeout": self.model_config.timeout,
            "extra_body": extra_body if extra_body else None,
        }
        rate_limiter = _get_mobilitybench_rate_limiter()
        if rate_limiter is not None:
            kwargs["rate_limiter"] = rate_limiter

        async_http_client = _new_loop_safe_async_http_client()
        if async_http_client is not None:
            kwargs["http_async_client"] = async_http_client

        max_retries = os.getenv("CCE_MOBILITYBENCH_API_MAX_RETRIES", "").strip()
        if max_retries:
            kwargs["max_retries"] = max(0, int(max_retries))

        return ChatOpenAI(
            **kwargs,
        )

    llm_manager.LLMManager._create_llm = _create_llm


def _new_loop_safe_async_http_client():  # type: ignore[no-untyped-def]
    """Avoid reusing LangChain's cached async client across per-case event loops."""
    global _MOBILITYBENCH_LOOP_SAFE_CLIENT_LOGGED

    enabled = (
        os.getenv(
            "CCE_MOBILITYBENCH_LOOP_SAFE_ASYNC_CLIENT",
            "0",
        )
        .strip()
        .lower()
    )
    if enabled in {"", "0", "false", "no", "off"}:
        return None

    import httpx

    client = httpx.AsyncClient(
        headers={"Connection": "close"},
        limits=httpx.Limits(max_keepalive_connections=0),
    )
    if not _MOBILITYBENCH_LOOP_SAFE_CLIENT_LOGGED:
        print(
            "[CCE LLM transport] loop-safe async client enabled; cross-case keep-alive disabled",
            file=sys.stderr,
            flush=True,
        )
        _MOBILITYBENCH_LOOP_SAFE_CLIENT_LOGGED = True
    return client


def _get_mobilitybench_rate_limiter():  # type: ignore[no-untyped-def]
    global _MOBILITYBENCH_RATE_LIMITER, _MOBILITYBENCH_RATE_LIMITER_CONFIG

    config = _rate_limit_config_from_env()
    if config is None:
        return None
    if _MOBILITYBENCH_RATE_LIMITER is not None:
        if config != _MOBILITYBENCH_RATE_LIMITER_CONFIG:
            raise RuntimeError("MobilityBench LLM rate-limit settings changed after initialization")
        return _MOBILITYBENCH_RATE_LIMITER

    from langchain_core.rate_limiters import InMemoryRateLimiter

    requests_per_second = _effective_requests_per_second(**config)
    check_every_n_seconds = min(
        0.25,
        max(0.05, (1.0 / requests_per_second) / 20.0),
    )
    _MOBILITYBENCH_RATE_LIMITER = InMemoryRateLimiter(
        requests_per_second=requests_per_second,
        check_every_n_seconds=check_every_n_seconds,
        max_bucket_size=1.0,
    )
    _MOBILITYBENCH_RATE_LIMITER_CONFIG = config
    print(
        "[CCE LLM throttle] enabled: "
        f"target_tpm={config['tokens_per_minute'] * config['utilization']:.0f}, "
        f"target_rpm={config['requests_per_minute'] * config['utilization']:.1f}, "
        f"reserved_tokens_per_request={config['tokens_per_request']:.0f}, "
        f"effective_rpm={requests_per_second * 60.0:.1f}",
        file=sys.stderr,
        flush=True,
    )
    return _MOBILITYBENCH_RATE_LIMITER


def _rate_limit_config_from_env():  # type: ignore[no-untyped-def]
    enabled = os.getenv("CCE_MOBILITYBENCH_RATE_LIMIT_ENABLED", "0").strip().lower()
    if enabled in {"", "0", "false", "no", "off"}:
        return None

    return {
        "tokens_per_minute": _positive_float_env(
            "CCE_MOBILITYBENCH_RATE_LIMIT_TPM",
            2_000_000.0,
        ),
        "requests_per_minute": _positive_float_env(
            "CCE_MOBILITYBENCH_RATE_LIMIT_RPM",
            500.0,
        ),
        "tokens_per_request": _positive_float_env(
            "CCE_MOBILITYBENCH_RATE_LIMIT_TOKENS_PER_REQUEST",
            50_000.0,
        ),
        "utilization": _bounded_float_env(
            "CCE_MOBILITYBENCH_RATE_LIMIT_UTILIZATION",
            0.6,
            lower=0.01,
            upper=1.0,
        ),
        "minimum_interval_seconds": _non_negative_float_env(
            "CCE_MOBILITYBENCH_RATE_LIMIT_MIN_INTERVAL_SECONDS",
            0.0,
        ),
    }


def _effective_requests_per_second(
    *,
    tokens_per_minute: float,
    requests_per_minute: float,
    tokens_per_request: float,
    utilization: float,
    minimum_interval_seconds: float,
) -> float:
    token_limited = tokens_per_minute * utilization / tokens_per_request / 60.0
    request_limited = requests_per_minute * utilization / 60.0
    limits = [token_limited, request_limited]
    if minimum_interval_seconds > 0:
        limits.append(1.0 / minimum_interval_seconds)
    return min(limits)


def _positive_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _non_negative_float_env(name: str, default: float) -> float:
    value = float(os.getenv(name, str(default)))
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _bounded_float_env(
    name: str,
    default: float,
    *,
    lower: float,
    upper: float,
) -> float:
    value = float(os.getenv(name, str(default)))
    if not lower <= value <= upper:
        raise ValueError(f"{name} must be between {lower} and {upper}")
    return value


def _patch_chat_openai_bind_tools() -> None:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        return

    original_bind_tools = ChatOpenAI.bind_tools
    if getattr(original_bind_tools, "_cce_mobility_patched", False):
        return

    def _bind_tools(self, tools, *args, **kwargs):  # type: ignore[no-untyped-def]
        return original_bind_tools(
            self,
            tools,
            *args,
            **_normalize_bind_tools_kwargs(tools, kwargs),
        )

    _bind_tools._cce_mobility_patched = True  # type: ignore[attr-defined]
    ChatOpenAI.bind_tools = _bind_tools


_patch_mobilitybench_llm_manager()
_patch_chat_openai_bind_tools()


def _patch_mobilitybench_skill_prompt() -> None:
    """Optionally append CCE agent-visible skill guidance to official MB prompts."""

    try:
        from mobility_bench.agent import prompts as prompt_module
    except Exception:
        return

    original_get_system_prompt = prompt_module.get_system_prompt
    if getattr(original_get_system_prompt, "_cce_mobility_skill_patched", False):
        return

    def _get_system_prompt(framework, role, template_vars=None):  # type: ignore[no-untyped-def]
        prompt = original_get_system_prompt(framework, role, template_vars)
        suffixes = [
            item
            for item in [
                _official_structured_fallback_suffix(framework=framework, role=role),
                _official_tool_argument_compatibility_suffix(
                    framework=framework,
                    role=role,
                ),
                _official_skill_prompt_suffix(framework=framework, role=role),
            ]
            if item
        ]
        if suffixes:
            return f"{prompt.rstrip()}\n\n" + "\n\n".join(suffixes)
        return prompt

    _get_system_prompt._cce_mobility_skill_patched = True  # type: ignore[attr-defined]
    prompt_module.get_system_prompt = _get_system_prompt


def _official_skill_prompt_suffix(*, framework: str, role: str) -> str:
    path = os.getenv("CCE_MOBILITYBENCH_SKILL_PROMPT_PATH", "").strip()
    if not path:
        return ""

    roles_raw = os.getenv("CCE_MOBILITYBENCH_SKILL_PROMPT_ROLES", "worker,reporter")
    allowed_roles = {item.strip() for item in roles_raw.split(",") if item.strip()}
    if role not in allowed_roles:
        return ""

    frameworks_raw = os.getenv("CCE_MOBILITYBENCH_SKILL_PROMPT_FRAMEWORKS", "plan_and_execute")
    allowed_frameworks = {item.strip() for item in frameworks_raw.split(",") if item.strip()}
    if framework not in allowed_frameworks:
        return ""

    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    if not text:
        return ""

    return "\n".join(
        [
            "Reusable recovery guidance from prior mobility-assistant experience:",
            "Use it only when the current user request, tool results, or observable task state supports it.",
            "Ignore guidance that is not supported by current observations.",
            _sanitize_agent_visible_prompt(text),
        ]
    )


def _sanitize_agent_visible_prompt(text: str) -> str:
    replacements = {
        "counterfactual": "environment update",
        "Counterfactual": "Environment update",
        "intervention": "environment update",
        "Intervention": "Environment update",
        "evaluator": "system",
        "Evaluator": "System",
        "judge": "review",
        "Judge": "Review",
    }
    sanitized = text
    for old, new in replacements.items():
        sanitized = sanitized.replace(old, new)
    return sanitized


_patch_mobilitybench_skill_prompt()


def _patch_plan_json_content_fallback() -> None:
    try:
        from mobility_bench.agent.frameworks.plan_and_execute import nodes
    except Exception:
        return

    original_remove_json_codeblock = nodes.remove_json_codeblock
    if getattr(original_remove_json_codeblock, "_cce_mobility_patched", False):
        return

    def _remove_json_codeblock(text: str) -> str:
        return _normalize_planner_json_content(
            text,
            original_remove_json_codeblock=original_remove_json_codeblock,
        )

    _remove_json_codeblock._cce_mobility_patched = True  # type: ignore[attr-defined]
    nodes.remove_json_codeblock = _remove_json_codeblock


def _normalize_bind_tools_kwargs(tools, kwargs):  # type: ignore[no-untyped-def]
    normalized = dict(kwargs)
    tool_names = _bound_tool_names(tools)
    scoped_env = (
        "CCE_MOBILITYBENCH_PLANNER_TOOL_CHOICE"
        if "RawPlan" in tool_names
        else "CCE_MOBILITYBENCH_WORKER_TOOL_CHOICE"
    )
    tool_choice = os.getenv(scoped_env, "").strip()
    if not tool_choice:
        tool_choice = os.getenv("CCE_MOBILITYBENCH_TOOL_CHOICE", "").strip()
    if tool_choice:
        if tool_choice.lower() in {"0", "none", "off", "unset"}:
            normalized.pop("tool_choice", None)
        else:
            normalized["tool_choice"] = tool_choice
        return normalized

    force_required = os.getenv("CCE_MOBILITYBENCH_FORCE_TOOL_CHOICE_REQUIRED", "0") != "0"
    if force_required and tools and not normalized.get("tool_choice"):
        normalized["tool_choice"] = "required"
    return normalized


def _bound_tool_names(tools) -> set[str]:  # type: ignore[no-untyped-def]
    names: set[str] = set()
    for tool in tools or []:
        if isinstance(tool, dict):
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name"):
                names.add(str(function["name"]))
                continue
            if tool.get("name"):
                names.add(str(tool["name"]))
                continue

        name = getattr(tool, "name", None) or getattr(tool, "__name__", None)
        if name:
            names.add(str(name))
    return names


def _tool_argument_normalization_enabled() -> bool:
    return os.getenv("CCE_MOBILITYBENCH_NORMALIZE_TOOL_ARGS", "1") != "0"


def _patch_mobilitybench_tool_argument_normalization() -> None:
    if not _tool_argument_normalization_enabled():
        return

    _patch_coordinate_helpers()

    try:
        from mobility_bench.tools.registry import ToolRegistry
    except Exception:
        return

    original_register = ToolRegistry.register
    if getattr(original_register, "_cce_mobility_args_patched", False):
        return

    def _register(cls, name, tool):  # type: ignore[no-untyped-def]
        return original_register(name, _wrap_tool_for_argument_normalization(name, tool))

    _register._cce_mobility_args_patched = True  # type: ignore[attr-defined]
    ToolRegistry.register = classmethod(_register)

    try:
        for name, tool in list(ToolRegistry._tools.items()):
            ToolRegistry._tools[name] = _wrap_tool_for_argument_normalization(name, tool)
    except Exception:
        return


def _patch_coordinate_helpers() -> None:
    try:
        from mobility_bench.tools.common import validators
    except Exception:
        validators = None

    if validators is not None:
        original_validate = validators.validate_coordinate
        if not getattr(original_validate, "_cce_mobility_args_patched", False):

            def _validate_coordinate(coord_str, coord_name="coordinate"):  # type: ignore[no-untyped-def]
                return original_validate(
                    _normalize_coordinate_string(coord_str),
                    coord_name,
                )

            _validate_coordinate._cce_mobility_args_patched = True  # type: ignore[attr-defined]
            validators.validate_coordinate = _validate_coordinate

            common_module = sys.modules.get("mobility_bench.tools.common")
            if common_module is not None:
                common_module.validate_coordinate = _validate_coordinate

    try:
        from mobility_bench.tools.sandbox import utils
    except Exception:
        return

    original_format = utils.format_decimal_places
    if getattr(original_format, "_cce_mobility_args_patched", False):
        return

    def _format_decimal_places(coord_str, places=6):  # type: ignore[no-untyped-def]
        return original_format(_normalize_coordinate_string(coord_str), places)

    _format_decimal_places._cce_mobility_args_patched = True  # type: ignore[attr-defined]
    utils.format_decimal_places = _format_decimal_places


def _wrap_tool_for_argument_normalization(name, tool):  # type: ignore[no-untyped-def]
    if not _tool_argument_normalization_enabled():
        return tool
    if getattr(tool, "_cce_mobility_args_patched", False):
        return tool

    if hasattr(tool, "func") and callable(getattr(tool, "func")):
        original_func = tool.func

        def _func(*args, **kwargs):  # type: ignore[no-untyped-def]
            return original_func(*args, **_normalize_tool_call_args(name, kwargs))

        _func._cce_mobility_args_patched = True  # type: ignore[attr-defined]
        try:
            tool.func = _func
        except Exception:
            object.__setattr__(tool, "func", _func)
        tool._cce_mobility_args_patched = True
        return tool

    if callable(tool):

        def _callable(*args, **kwargs):  # type: ignore[no-untyped-def]
            return tool(*args, **_normalize_tool_call_args(name, kwargs))

        _callable._cce_mobility_args_patched = True  # type: ignore[attr-defined]
        return _callable

    return tool


def _normalize_tool_call_args(name: str, kwargs):  # type: ignore[no-untyped-def]
    if not isinstance(kwargs, dict):
        return kwargs

    normalized = dict(kwargs)
    for key in ("origin", "destination", "location"):
        if key in normalized:
            normalized[key] = _normalize_coordinate_string(normalized[key])

    if "waypoints" in normalized:
        normalized["waypoints"] = _normalize_coordinate_list(normalized["waypoints"])

    if name == "driving_route" and "strategy" in normalized:
        normalized["strategy"] = _normalize_strategy_list(normalized["strategy"])
    elif name == "bus_route" and "strategy" in normalized:
        normalized["strategy"] = _normalize_bus_strategy(normalized["strategy"])

    if name == "reverse_geocoding":
        _normalize_reverse_geocoding_args(normalized)

    if "radius" in normalized:
        normalized["radius"] = _normalize_int(normalized["radius"])

    return normalized


def _normalize_coordinate_string(value):  # type: ignore[no-untyped-def]
    if not isinstance(value, str):
        return value

    text = value.strip()
    match = re.fullmatch(
        r"[\(\[（【]?\s*([+-]?\d+(?:\.\d+)?)\s*[,，]\s*([+-]?\d+(?:\.\d+)?)\s*[\)\]）】]?",
        text,
    )
    if not match:
        return text

    return f"{match.group(1)},{match.group(2)}"


def _normalize_coordinate_list(value):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        parsed = _parse_maybe_json_or_literal(value)
        if isinstance(parsed, list):
            return [_normalize_coordinate_string(item) for item in parsed]
        normalized = _normalize_coordinate_string(value)
        return [] if not normalized else [normalized]

    if isinstance(value, list):
        return [_normalize_coordinate_string(item) for item in value]

    return value


def _normalize_strategy_list(value):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        parsed = _parse_maybe_json_or_literal(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        text = value.strip()
        return [] if not text else [text]

    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]

    return value


def _normalize_bus_strategy(value):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        parsed = _parse_maybe_json_or_literal(value)
        if isinstance(parsed, list):
            return _normalize_bus_strategy(parsed)
        return value.strip()

    if isinstance(value, list):
        for item in value:
            text = str(item).strip()
            if text:
                return text
        return "0"

    if value is None:
        return "0"
    return str(value).strip()


def _normalize_reverse_geocoding_args(kwargs: dict) -> None:
    location = kwargs.pop("location", None)
    if location and ("longitude" not in kwargs or "latitude" not in kwargs):
        normalized = _normalize_coordinate_string(location)
        if isinstance(normalized, str) and "," in normalized:
            lon, lat = normalized.split(",", 1)
            kwargs.setdefault("longitude", lon)
            kwargs.setdefault("latitude", lat)

    for key in ("longitude", "latitude"):
        if isinstance(kwargs.get(key), str):
            kwargs[key] = kwargs[key].strip().strip("()[]（）【】")


def _normalize_int(value):  # type: ignore[no-untyped-def]
    if isinstance(value, str):
        text = value.strip()
        if re.fullmatch(r"\d+", text):
            return int(text)
    return value


def _parse_maybe_json_or_literal(value: str):  # type: ignore[no-untyped-def]
    text = value.strip()
    if not text:
        return None
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            continue
    return None


def _normalize_planner_json_content(
    text: str,
    *,
    original_remove_json_codeblock=None,  # type: ignore[no-untyped-def]
) -> str:
    cleaned = (
        original_remove_json_codeblock(text)
        if original_remove_json_codeblock
        else _strip_json_codeblock(text)
    )
    parsed = _extract_json_object(cleaned)
    if parsed:
        return parsed

    if not _synthetic_plan_fallback_enabled():
        return cleaned

    return json.dumps(
        {
            "thinking": "工具调用兼容兜底：根据用户需求生成最小可执行计划。",
            "intent": "出行信息查询或路线规划",
            "steps": [
                [
                    (
                        "根据用户需求和上下文调用合适的地图工具查询地点、路线或周边信息，"
                        "并整理当前可用结果。"
                    )
                ]
            ],
        },
        ensure_ascii=False,
    )


def _strip_json_codeblock(text: str) -> str:
    stripped = (text or "").strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def _extract_json_object(text: str) -> str:
    decoder = json.JSONDecoder()
    candidate = (text or "").strip()
    if not candidate:
        return ""

    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            parsed, end = decoder.raw_decode(candidate[index:])
        except Exception:
            continue
        if isinstance(parsed, dict):
            return candidate[index : index + end]
    return ""


def _synthetic_plan_fallback_enabled() -> bool:
    value = os.getenv(
        "CCE_MOBILITYBENCH_SYNTHETIC_PLAN_FALLBACK",
        os.getenv("CCE_MOBILITYBENCH_PLAN_JSON_FALLBACK", "0"),
    )
    return value != "0"


def _official_structured_fallback_suffix(*, framework: str, role: str) -> str:
    if os.getenv("CCE_MOBILITYBENCH_PLAN_JSON_FALLBACK", "0") == "0":
        return ""
    if framework != "plan_and_execute" or role != "planner":
        return ""
    return "\n".join(
        [
            "Structured planning compatibility instruction:",
            (
                "If tool/function calling is unavailable, output only one JSON object "
                "with keys thinking, intent, and steps."
            ),
            (
                "The steps value must be a non-empty list of step lists, for example "
                '{"thinking":"...","intent":"...","steps":[["查询地点或路线信息"]]}'
            ),
            "Do not output Markdown, explanations, or an empty steps list.",
        ]
    )


def _official_tool_argument_compatibility_suffix(*, framework: str, role: str) -> str:
    if not _tool_argument_normalization_enabled():
        return ""
    if framework != "plan_and_execute":
        return ""

    if role == "worker":
        return "\n".join(
            [
                "Tool argument compatibility instruction:",
                (
                    "Use exact tool names only, such as driving_route, walking_route, "
                    "bicycling_route, bus_route, search_around_poi, reverse_geocoding, "
                    "weather_query, traffic_status, and query_poi."
                ),
                (
                    "Coordinates must be plain longitude,latitude strings without "
                    "parentheses, labels, or extra text."
                ),
                (
                    "If a route strategy is needed, put it in the strategy argument "
                    "instead of inventing a new tool name."
                ),
            ]
        )

    if role == "reporter":
        return "\n".join(
            [
                "Reporting compatibility instruction:",
                "Do not call tools or functions while writing the final report.",
            ]
        )

    return ""


def _patch_raw_plan_normalization() -> None:
    try:
        from mobility_bench.agent.graph import state as graph_state
    except Exception:
        return

    raw_plan_cls = graph_state.RawPlan
    original_init = raw_plan_cls.__init__
    if getattr(original_init, "_cce_mobility_patched", False):
        return

    def _init(self, **data):  # type: ignore[no-untyped-def]
        return original_init(self, **_normalize_raw_plan_data(data))

    _init._cce_mobility_patched = True  # type: ignore[attr-defined]
    raw_plan_cls.__init__ = _init


def _normalize_raw_plan_data(data):  # type: ignore[no-untyped-def]
    if not isinstance(data, dict):
        return data

    normalized = dict(data)
    steps = normalized.get("steps")
    if isinstance(steps, str):
        parsed_steps = _parse_plan_steps(steps)
        if parsed_steps:
            normalized["steps"] = parsed_steps
    elif isinstance(steps, list):
        normalized_steps = _normalize_plan_steps(steps)
        if normalized_steps:
            normalized["steps"] = normalized_steps

    if not normalized.get("thinking"):
        normalized["thinking"] = (
            normalized.get("reasoning")
            or normalized.get("analysis")
            or normalized.get("thought")
            or "根据用户需求生成执行计划。"
        )

    if not normalized.get("intent"):
        normalized["intent"] = (
            normalized.get("user_intent") or normalized.get("task_intent") or "出行信息查询"
        )

    return normalized


def _parse_plan_steps(value: str):  # type: ignore[no-untyped-def]
    text = value.strip()
    if not text:
        return None

    candidates = [text]
    match = re.search(r"\[\s*\[.*\]\s*\]", text, re.DOTALL)
    if match:
        candidates.insert(0, match.group(0))

    for candidate in candidates:
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(candidate)
            except Exception:
                continue
            if isinstance(parsed, list):
                normalized = _normalize_plan_steps(parsed)
                if normalized:
                    return normalized
    return None


def _normalize_plan_steps(steps):  # type: ignore[no-untyped-def]
    normalized = []
    for step in steps:
        if isinstance(step, str):
            task = step.strip()
            if task:
                normalized.append([task])
            continue
        if isinstance(step, list):
            tasks = [str(task).strip() for task in step if str(task).strip()]
            if tasks:
                normalized.append(tasks)
            continue
        return None
    return normalized or None


_patch_plan_json_content_fallback()
_patch_raw_plan_normalization()
_patch_mobilitybench_tool_argument_normalization()
