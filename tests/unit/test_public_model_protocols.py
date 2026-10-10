import ast
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from cce_mobility.core.config import ModelConfig
from cce_mobility.core.llm import ChatClient, _get_secret
from cce_mobility.workflows.public.configuration import validate_models


@pytest.mark.parametrize(
    "provider,response",
    [
        ("openai_compatible", {"choices": [{"message": {"content": "OK"}}]}),
        (
            "responses",
            {"output": [{"type": "message", "content": [{"type": "output_text", "text": "OK"}]}]},
        ),
        ("anthropic", {"content": [{"type": "text", "text": "OK"}]}),
    ],
)
def test_explicit_protocol_request_shapes(monkeypatch, provider, response):
    monkeypatch.setenv("PUBLIC_TEST_KEY", "example-key")
    requests = []

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(response).encode()

    def receive(request, timeout):
        requests.append(request)
        assert timeout > 0
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", receive)
    client = ChatClient(
        ModelConfig(
            model="explicit-model",
            provider=provider,
            base_url="https://example.org/v1",
            api_key_env="PUBLIC_TEST_KEY",
            extra={"anthropic_auth_style": "x-api-key", "api_max_retries": 0},
        )
    )
    result = client.chat(system_prompt="system", user_prompt="task", max_tokens=64)
    assert result.ok and result.text == "OK"
    assert len(requests) == 1
    body = json.loads(requests[0].data)
    assert body["model"] == "explicit-model"
    if provider == "responses":
        assert requests[0].full_url.endswith("/responses")
        assert body["max_output_tokens"] == 64 and "messages" not in body
    elif provider == "anthropic":
        assert requests[0].full_url.endswith("/messages")
        assert requests[0].get_header("X-api-key") == "example-key"
        assert body["system"] == "system"
    else:
        assert requests[0].full_url.endswith("/chat/completions")
        assert body["messages"][0]["role"] == "system"
    assert "example-key" not in json.dumps(client.transport_metrics())


@pytest.mark.parametrize(
    "field,value",
    [
        ("max_tokens", 0),
        ("temperature", -1),
        ("extra", {"api_timeout_seconds": 0}),
        ("extra", {"api_max_retries": -1}),
    ],
)
def test_invalid_inference_settings_fail_before_network(field, value):
    model = {"model": "offline", "provider": "placeholder", field: value}
    with pytest.raises(ValueError):
        validate_models({"agent": model, "world_model": model, "judge": model})


@pytest.mark.parametrize("name", ["CCE_MODEL_API_KEY", "CCE_JUDGE_API_KEY"])
def test_role_key_can_be_read_from_local_file_without_export(monkeypatch, name):
    monkeypatch.delenv(name, raising=False)
    local = ModuleType("cce_mobility.local_secrets")
    setattr(local, name, "example-local-token")
    monkeypatch.setitem(sys.modules, local.__name__, local)
    assert _get_secret(name) == "example-local-token"


@pytest.mark.parametrize("name", ["CCE_MODEL_API_KEY", "CCE_JUDGE_API_KEY"])
def test_role_environment_override_takes_precedence(monkeypatch, name):
    local = ModuleType("cce_mobility.local_secrets")
    setattr(local, name, "example-local-token")
    monkeypatch.setitem(sys.modules, local.__name__, local)
    monkeypatch.setenv(name, "example-env-token")
    assert _get_secret(name) == "example-env-token"


def test_shared_key_and_custom_alias_are_explicit_local_choices(monkeypatch):
    names = ("CCE_MODEL_API_KEY", "CCE_JUDGE_API_KEY", "CUSTOM_PROVIDER_API_KEY")
    local = ModuleType("cce_mobility.local_secrets")
    for name in names:
        monkeypatch.delenv(name, raising=False)
        setattr(local, name, "example-shared-token")
    monkeypatch.setitem(sys.modules, local.__name__, local)
    assert {_get_secret(name) for name in names} == {"example-shared-token"}


def test_default_key_template_and_model_roles_use_only_two_keys():
    root = Path(__file__).resolve().parents[2]
    template = ast.parse((root / "src/cce_mobility/local_secrets.example.py").read_text())
    names = {
        target.id
        for node in template.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert names == {"CCE_MODEL_API_KEY", "CCE_JUDGE_API_KEY"}
    models = json.loads((root / "configs/models/openai_compatible.example.json").read_text())
    assert models["agent"]["api_key_env"] == "CCE_MODEL_API_KEY"
    assert models["world_model"]["api_key_env"] == "CCE_MODEL_API_KEY"
    assert models["judge"]["api_key_env"] == "CCE_JUDGE_API_KEY"
