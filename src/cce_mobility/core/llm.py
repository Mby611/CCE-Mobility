from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

from cce_mobility.core.config import ModelConfig


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    error: str = ""
    latency_ms: int = 0
    raw: dict | None = None


class QoderQuotaExhaustedError(RuntimeError):
    """The Qoder CLI reported an exhausted allocation/credit quota."""


class APIQuotaExhaustedError(RuntimeError):
    """An HTTP model endpoint reported exhausted balance or allocation."""


class ChatClient:
    """Small OpenAI-compatible chat client.

    The client intentionally uses the Python standard library so this framework
    stays portable. For smoke tests, set provider/model to `placeholder`.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.api_key = _get_secret(config.api_key_env)
        self.call_delay_seconds = max(
            0.0,
            float((config.extra or {}).get("api_call_delay_seconds", 0.0)),
        )
        self.max_retries = max(0, int((config.extra or {}).get("api_max_retries", 0)))
        self.retry_backoff_seconds = max(
            0.0,
            float((config.extra or {}).get("api_retry_backoff_seconds", 2.0)),
        )
        self._transport_stats: dict[str, Any] = {
            "call_count": 0,
            "request_count": 0,
            "retry_count": 0,
            "error_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "credits": 0.0,
            "duration_ms": 0,
            "duration_api_ms": 0,
            "model_usage_keys": set(),
        }
        self._qoder_cli_version_cache = ""

    @property
    def is_placeholder(self) -> bool:
        return self.config.provider == "placeholder" or self.config.model.startswith("placeholder")

    @property
    def is_qoder_cli(self) -> bool:
        provider = self.config.provider.strip().lower().replace("-", "_")
        return provider in {"qoder", "qoder_cli"}

    @property
    def is_anthropic_messages(self) -> bool:
        provider = self.config.provider.strip().lower().replace("-", "_")
        return provider in {
            "anthropic",
            "anthropic_messages",
            "quickrouter_anthropic",
        }

    @property
    def is_openai_responses(self) -> bool:
        provider = self.config.provider.strip().lower().replace("-", "_")
        return provider in {
            "openai_responses",
            "responses",
            "responses_api",
        }

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        if self.is_placeholder:
            return LLMResponse(ok=True, text=self._placeholder_response(user_prompt))

        if self.is_qoder_cli:
            return self._chat_qoder_cli(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if self.is_anthropic_messages:
            return self._chat_anthropic_messages(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        if self.is_openai_responses:
            return self._chat_openai_responses(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )

        url = self._chat_url()
        if not url:
            error = "Model chat_url/base_url is empty."
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if not self.api_key:
            error = f"Missing API key: {self.config.api_key_env} (local_secrets.py or environment)"
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if self.call_delay_seconds > 0:
            time.sleep(self.call_delay_seconds)

        extra = self.config.extra or {}
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        if not extra.get("omit_temperature", False):
            payload["temperature"] = self.config.temperature if temperature is None else temperature
        payload.update(
            {key: value for key, value in extra.items() if key not in _LOCAL_TRANSPORT_OPTIONS}
        )
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.time()
        raw = None
        request_count = 0
        timeout_seconds = max(
            1.0,
            float((self.config.extra or {}).get("api_timeout_seconds", 180.0)),
        )
        for attempt in range(self.max_retries + 1):
            request_count += 1
            try:
                with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                    raw = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error = f"HTTP {exc.code}: {body[:500]}"
                if self._fail_fast_on_quota and _is_api_quota_exhausted(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    raise APIQuotaExhaustedError(error)
                if attempt >= self.max_retries or not _is_retriable_http_error(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=error, latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))
            except Exception as exc:
                if attempt >= self.max_retries:
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=str(exc), latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))
        if raw is None:
            latency_ms = int((time.time() - start) * 1000)
            self._record_openai_result(
                None,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error="API request failed without a response.",
                latency_ms=latency_ms,
            )
        latency_ms = int((time.time() - start) * 1000)

        try:
            text = _message_content_to_text(raw["choices"][0]["message"].get("content"))
        except Exception:
            self._record_openai_result(
                raw,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error=f"Unexpected response: {str(raw)[:500]}",
                latency_ms=latency_ms,
                raw=raw,
            )
        self._record_openai_result(
            raw,
            ok=True,
            request_count=request_count,
            duration_ms=latency_ms,
        )
        return LLMResponse(ok=True, text=text, latency_ms=latency_ms, raw=raw)

    def _chat_openai_responses(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None,
    ) -> LLMResponse:
        url = self._responses_url()
        if not url:
            error = "Model chat_url/base_url is empty."
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if not self.api_key:
            error = f"Missing API key: {self.config.api_key_env} (local_secrets.py or environment)"
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if self.call_delay_seconds > 0:
            time.sleep(self.call_delay_seconds)

        extra = self.config.extra or {}
        payload: dict[str, Any] = {
            "model": self.config.model,
            "instructions": system_prompt,
            "input": user_prompt,
            "max_output_tokens": (self.config.max_tokens if max_tokens is None else max_tokens),
            "store": bool(extra.get("responses_store", False)),
        }
        reasoning_effort = str(extra.get("responses_reasoning_effort") or "").strip()
        if reasoning_effort:
            payload["reasoning"] = {"effort": reasoning_effort}
        payload.update(
            {key: value for key, value in extra.items() if key not in _LOCAL_TRANSPORT_OPTIONS}
        )
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        start = time.time()
        raw = None
        request_count = 0
        timeout_seconds = max(1.0, float(extra.get("api_timeout_seconds", 180.0)))
        for attempt in range(self.max_retries + 1):
            request_count += 1
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error = f"HTTP {exc.code}: {body[:500]}"
                if self._fail_fast_on_quota and _is_api_quota_exhausted(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    raise APIQuotaExhaustedError(error)
                if attempt >= self.max_retries or not _is_retriable_http_error(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=error, latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))
            except Exception as exc:
                if attempt >= self.max_retries:
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=str(exc), latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))

        latency_ms = int((time.time() - start) * 1000)
        if raw is None:
            self._record_openai_result(
                None,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error="Responses API request failed without a response.",
                latency_ms=latency_ms,
            )
        text = _responses_output_text(raw)
        if not text:
            self._record_openai_result(
                raw,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error=f"Unexpected Responses API response: {str(raw)[:500]}",
                latency_ms=latency_ms,
                raw=raw,
            )
        self._record_openai_result(
            raw,
            ok=True,
            request_count=request_count,
            duration_ms=latency_ms,
        )
        return LLMResponse(ok=True, text=text, latency_ms=latency_ms, raw=raw)

    def _chat_anthropic_messages(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        url = self._anthropic_messages_url()
        if not url:
            error = "Anthropic messages chat_url/base_url is empty."
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if not self.api_key:
            error = f"Missing API key: {self.config.api_key_env} (local_secrets.py or environment)"
            self._record_openai_result(None, ok=False, request_count=0, duration_ms=0)
            return LLMResponse(ok=False, error=error)
        if self.call_delay_seconds > 0:
            time.sleep(self.call_delay_seconds)

        payload = {
            "model": self.config.model,
            "system": system_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "stream": False,
            "temperature": self.config.temperature if temperature is None else temperature,
            "max_tokens": self.config.max_tokens if max_tokens is None else max_tokens,
        }
        payload.update(
            {
                key: value
                for key, value in (self.config.extra or {}).items()
                if key not in _LOCAL_TRANSPORT_OPTIONS
            }
        )
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        auth_style = (
            str((self.config.extra or {}).get("anthropic_auth_style", "bearer")).strip().lower()
        )
        if auth_style == "x-api-key":
            headers["x-api-key"] = self.api_key
            headers["anthropic-version"] = str(
                (self.config.extra or {}).get("anthropic_version", "2023-06-01")
            )
        else:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        start = time.time()
        raw = None
        request_count = 0
        timeout_seconds = max(
            1.0,
            float((self.config.extra or {}).get("api_timeout_seconds", 180.0)),
        )
        for attempt in range(self.max_retries + 1):
            request_count += 1
            try:
                with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                    raw = json.loads(response.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                error = f"HTTP {exc.code}: {body[:500]}"
                if self._fail_fast_on_quota and _is_api_quota_exhausted(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    raise APIQuotaExhaustedError(error)
                if attempt >= self.max_retries or not _is_retriable_http_error(exc.code, body):
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=error, latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))
            except Exception as exc:
                if attempt >= self.max_retries:
                    latency_ms = int((time.time() - start) * 1000)
                    self._record_openai_result(
                        None,
                        ok=False,
                        request_count=request_count,
                        duration_ms=latency_ms,
                    )
                    return LLMResponse(ok=False, error=str(exc), latency_ms=latency_ms)
                self._transport_stats["retry_count"] += 1
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))

        latency_ms = int((time.time() - start) * 1000)
        if raw is None:
            self._record_openai_result(
                None,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error="Anthropic messages request failed without a response.",
                latency_ms=latency_ms,
            )
        try:
            text = _message_content_to_text(raw["content"])
        except Exception:
            self._record_openai_result(
                raw,
                ok=False,
                request_count=request_count,
                duration_ms=latency_ms,
            )
            return LLMResponse(
                ok=False,
                error=f"Unexpected Anthropic response: {str(raw)[:500]}",
                latency_ms=latency_ms,
                raw=raw,
            )
        self._record_openai_result(
            raw,
            ok=True,
            request_count=request_count,
            duration_ms=latency_ms,
        )
        return LLMResponse(ok=True, text=text, latency_ms=latency_ms, raw=raw)

    def transport_metrics(self) -> dict[str, Any]:
        """Return transport provenance for report aggregation.

        Qoder CLI needs explicit provenance because a tier is not a disclosed
        base-model identifier and its calls consume account Credits. Standard
        OpenAI-compatible calls expose the aggregate usage reported by the API.
        """
        if self.is_placeholder:
            return {}
        stats = self._transport_stats
        if not self.is_qoder_cli:
            return {
                "backend": (
                    "anthropic_messages"
                    if self.is_anthropic_messages
                    else ("openai_responses" if self.is_openai_responses else "openai_compatible")
                ),
                "provider": self.config.provider,
                "configured_model": self.config.model,
                "call_count": int(stats["call_count"]),
                "request_count": int(stats["request_count"]),
                "retry_count": int(stats["retry_count"]),
                "error_count": int(stats["error_count"]),
                "input_tokens": int(stats["input_tokens"]),
                "output_tokens": int(stats["output_tokens"]),
                "total_tokens": int(stats["input_tokens"] + stats["output_tokens"]),
                "duration_ms": int(stats["duration_ms"]),
                "reported_model_usage_keys": sorted(stats["model_usage_keys"]),
                "underlying_model_disclosed": True,
                "temperature_controlled": (
                    not self.is_openai_responses
                    and not bool((self.config.extra or {}).get("omit_temperature"))
                ),
                **(
                    {
                        "reasoning_effort_requested": str(
                            self.config.extra["responses_reasoning_effort"]
                        ).strip()
                    }
                    if self.is_openai_responses
                    and (self.config.extra or {}).get("responses_reasoning_effort")
                    else {}
                ),
            }
        return {
            "backend": "qoder_cli",
            "configured_model": self.config.model,
            "cli_version": self._qoder_cli_version(),
            **(
                {
                    "reasoning_effort_requested": str(
                        self.config.extra["qoder_reasoning_effort"]
                    ).strip()
                }
                if (self.config.extra or {}).get("qoder_reasoning_effort")
                else {}
            ),
            "call_count": int(stats["call_count"]),
            "error_count": int(stats["error_count"]),
            "input_tokens": int(stats["input_tokens"]),
            "output_tokens": int(stats["output_tokens"]),
            "credits": round(float(stats["credits"]), 6),
            "duration_ms": int(stats["duration_ms"]),
            "duration_api_ms": int(stats["duration_api_ms"]),
            "reported_model_usage_keys": sorted(stats["model_usage_keys"]),
            "underlying_model_disclosed": False,
            "temperature_controlled": False,
        }

    def _chat_qoder_cli(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None,
        max_tokens: int | None,
    ) -> LLMResponse:
        executable = self._qoder_cli_path()
        if not executable:
            error = "Qoder CLI executable not found. Set extra.qoder_cli_path or install QoderWork."
            self._record_qoder_result(None, ok=False)
            return LLMResponse(ok=False, error=error)

        if self.call_delay_seconds > 0:
            time.sleep(self.call_delay_seconds)

        effective_max_tokens = self.config.max_tokens if max_tokens is None else max_tokens
        permission_mode = str((self.config.extra or {}).get("qoder_permission_mode", "dont_ask"))
        command = [
            executable,
            "--model",
            self.config.model,
            "--tools",
            "",
            "--permission-mode",
            permission_mode,
            "--no-session-persistence",
            "--output-format",
            "json",
            "--max-output-tokens",
            str(effective_max_tokens),
            "--system-prompt",
            system_prompt,
            "--print",
        ]
        reasoning_effort = str(
            (self.config.extra or {}).get("qoder_reasoning_effort") or ""
        ).strip()
        if reasoning_effort:
            command[3:3] = ["--reasoning-effort", reasoning_effort]
        timeout_seconds = max(
            1.0,
            float((self.config.extra or {}).get("qoder_timeout_seconds", 300.0)),
        )
        configured_cwd = (self.config.extra or {}).get("qoder_cwd")
        if configured_cwd:
            cwd = str(configured_cwd)
        else:
            try:
                cwd = os.getcwd()
            except FileNotFoundError:
                # Restore relative artifact paths as well as the CLI working directory.
                cwd = str(Path(__file__).resolve().parents[3])
                os.chdir(cwd)
        requested_temperature = self.config.temperature if temperature is None else temperature
        start = time.time()
        last_error = ""
        last_raw: dict[str, Any] | None = None
        for attempt in range(self.max_retries + 1):
            try:
                completed = subprocess.run(
                    command,
                    cwd=cwd,
                    input=user_prompt,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=timeout_seconds,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                last_error = f"Qoder CLI execution failed: {exc}"
            else:
                raw = _parse_qoder_json_output(completed.stdout)
                last_raw = raw
                if completed.returncode == 0 and _qoder_result_ok(raw):
                    assert raw is not None
                    raw["_cce_transport"] = {
                        "provider": "qoder_cli",
                        "configured_model": self.config.model,
                        "cli_version": self._qoder_cli_version(),
                        "temperature_requested": requested_temperature,
                        "temperature_applied": False,
                        **(
                            {"reasoning_effort_requested": reasoning_effort}
                            if reasoning_effort
                            else {}
                        ),
                    }
                    self._record_qoder_result(raw, ok=True)
                    text = raw.get("result", "")
                    if not isinstance(text, str):
                        text = json.dumps(text, ensure_ascii=False)
                    latency_ms = int(raw.get("duration_ms") or (time.time() - start) * 1000)
                    return LLMResponse(
                        ok=True,
                        text=text,
                        latency_ms=latency_ms,
                        raw=raw,
                    )
                last_error = _qoder_error_message(completed, raw)
                if _is_qoder_quota_exhausted(last_error):
                    self._record_qoder_result(raw, ok=False)
                    raise QoderQuotaExhaustedError(last_error)

            if attempt < self.max_retries:
                time.sleep(min(60.0, self.retry_backoff_seconds * (2**attempt)))

        self._record_qoder_result(last_raw, ok=False)
        return LLMResponse(
            ok=False,
            error=last_error or "Qoder CLI request failed without a response.",
            latency_ms=int((time.time() - start) * 1000),
        )

    def _qoder_cli_path(self) -> str:
        configured = str((self.config.extra or {}).get("qoder_cli_path") or "").strip()
        candidates = [
            configured,
            shutil.which("qodercli") or "",
            shutil.which("qoder") or "",
            "/Applications/QoderWork.app/Contents/Resources/bin/qodercli",
            str(Path.home() / ".qoder" / "bin" / "qodercli"),
            str(Path.home() / ".qoder" / "bin" / "qoder"),
            str(Path.home() / ".local" / "bin" / "qoder"),
        ]
        for candidate in candidates:
            if candidate and Path(candidate).is_file() and os.access(candidate, os.X_OK):
                return candidate
        return ""

    def _qoder_cli_version(self) -> str:
        if self._qoder_cli_version_cache:
            return self._qoder_cli_version_cache
        executable = self._qoder_cli_path()
        if not executable:
            return ""
        try:
            completed = subprocess.run(
                [executable, "--version"],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return ""
        if completed.returncode == 0:
            self._qoder_cli_version_cache = completed.stdout.strip().splitlines()[-1]
        return self._qoder_cli_version_cache

    def _record_qoder_result(self, raw: dict[str, Any] | None, *, ok: bool) -> None:
        stats = self._transport_stats
        stats["call_count"] += 1
        if not ok:
            stats["error_count"] += 1
        if not raw:
            return
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        stats["input_tokens"] += int(usage.get("input_tokens") or 0)
        stats["output_tokens"] += int(usage.get("output_tokens") or 0)
        stats["credits"] += float(raw.get("total_credits") or 0.0)
        stats["duration_ms"] += int(raw.get("duration_ms") or 0)
        stats["duration_api_ms"] += int(raw.get("duration_api_ms") or 0)
        model_usage = raw.get("modelUsage")
        if isinstance(model_usage, dict):
            stats["model_usage_keys"].update(str(key) for key in model_usage)

    def _record_openai_result(
        self,
        raw: dict[str, Any] | None,
        *,
        ok: bool,
        request_count: int,
        duration_ms: int,
    ) -> None:
        stats = self._transport_stats
        stats["call_count"] += 1
        stats["request_count"] += request_count
        stats["duration_ms"] += duration_ms
        if not ok:
            stats["error_count"] += 1
        if not raw:
            return
        usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
        stats["input_tokens"] += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        stats["output_tokens"] += int(
            usage.get("completion_tokens") or usage.get("output_tokens") or 0
        )
        reported_model = raw.get("model")
        if reported_model:
            stats["model_usage_keys"].add(str(reported_model))

    def _chat_url(self) -> str:
        if self.config.chat_url:
            return self.config.chat_url
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/chat/completions"

    def _responses_url(self) -> str:
        if self.config.chat_url:
            return self.config.chat_url
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/responses"):
            return base_url
        return base_url + "/responses"

    def _anthropic_messages_url(self) -> str:
        if self.config.chat_url:
            return self.config.chat_url
        base_url = self.config.base_url.rstrip("/")
        if not base_url:
            return ""
        if base_url.endswith("/messages"):
            return base_url
        return base_url + "/v1/messages"

    @property
    def _fail_fast_on_quota(self) -> bool:
        return bool((self.config.extra or {}).get("fail_fast_on_quota", False))

    @staticmethod
    def _placeholder_response(user_prompt: str) -> str:
        return (
            "Travel Plan:\n"
            "Day 1:\n"
            "Current City: placeholder\n"
            "Transportation: placeholder option\n"
            "Breakfast: placeholder\n"
            "Attraction: placeholder\n"
            "Lunch: placeholder\n"
            "Dinner: placeholder\n"
            "Accommodation: placeholder\n\n"
            "Note: placeholder LLM response for smoke testing."
        )


def _get_secret(name: str) -> str:
    """Read the configured key name from the environment or a local-only file."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        local_secrets = import_module("cce_mobility.local_secrets")
    except ModuleNotFoundError:
        return ""
    return str(getattr(local_secrets, name, "") or "")


_LOCAL_TRANSPORT_OPTIONS = {
    "api_call_delay_seconds",
    "api_max_retries",
    "api_retry_backoff_seconds",
    "api_timeout_seconds",
    "fail_fast_on_quota",
    "omit_temperature",
    "anthropic_auth_style",
    "anthropic_version",
    "qoder_cli_path",
    "qoder_cwd",
    "qoder_permission_mode",
    "qoder_timeout_seconds",
    "responses_reasoning_effort",
    "responses_store",
}


def _is_api_quota_exhausted(status: int, body: str) -> bool:
    if status not in {400, 402, 403, 429, 500}:
        return False
    lowered = body.lower().replace("_", " ").replace("-", " ")
    return any(
        marker in lowered
        for marker in (
            "allocated quota exceeded",
            "allocation quota",
            "insufficient quota",
            "quota exceeded",
            "quota exhausted",
            "credit balance is too low",
            "insufficient balance",
            "balance insufficient",
            "credits exhausted",
            "no credits available",
        )
    )


def _is_retriable_http_error(status: int, body: str) -> bool:
    if status in {429, 502, 503, 504}:
        return True
    lowered = body.lower()
    return status >= 500 and any(
        marker in lowered
        for marker in (
            "throttling",
            "quota",
            "rate limit",
            "too many requests",
            "mpe-429",
            "temporarily unavailable",
            "限流",
        )
    )


def _message_content_to_text(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                value = item.get("text")
                if isinstance(value, str):
                    parts.append(value)
                elif isinstance(item.get("content"), str):
                    parts.append(item["content"])
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if isinstance(content, dict):
        if isinstance(content.get("text"), str):
            return content["text"]
        if isinstance(content.get("content"), str):
            return content["content"]
        return json.dumps(content, ensure_ascii=False)
    return str(content)


def _responses_output_text(raw: dict[str, Any]) -> str:
    direct = raw.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct
    parts: list[str] = []
    output = raw.get("output")
    if not isinstance(output, list):
        return ""
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if block.get("type") == "output_text" and isinstance(text, str):
                parts.append(text)
    return "\n".join(part for part in parts if part)


def _parse_qoder_json_output(output: str) -> dict[str, Any] | None:
    text = output.strip()
    if not text:
        return None
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        for line in reversed(text.splitlines()):
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
        return None
    return value if isinstance(value, dict) else None


def _qoder_result_ok(raw: dict[str, Any] | None) -> bool:
    if not raw:
        return False
    return (
        raw.get("type") == "result"
        and raw.get("subtype") == "success"
        and not bool(raw.get("is_error"))
        and "result" in raw
    )


def _qoder_error_message(
    completed: subprocess.CompletedProcess[str],
    raw: dict[str, Any] | None,
) -> str:
    if raw:
        result = raw.get("result") or raw.get("error") or raw
        detail = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
        return f"Qoder CLI returned an error: {detail[:500]}"
    stderr = completed.stderr.strip()
    stdout = completed.stdout.strip()
    detail = stderr or stdout or f"exit code {completed.returncode}"
    return f"Qoder CLI returned an invalid response: {detail[:500]}"


def _is_qoder_quota_exhausted(message: str) -> bool:
    lowered = message.lower().replace("_", " ").replace("-", " ")
    return any(
        marker in lowered
        for marker in (
            "allocated quota exceeded",
            "allocationquota",
            "insufficient quota",
            "quota exceeded",
            "quota exhausted",
            "you exceeded your current quota",
            "credits exhausted",
            "no credits available",
        )
    )
