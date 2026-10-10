import json
import subprocess
from pathlib import Path

import pytest

from cce_mobility.workflows.release.security_audit import (
    audit_repository,
    content_findings,
    path_findings,
    read_known_secrets,
)


@pytest.mark.parametrize(
    "content",
    [
        "AIHUB_API_KEY = '" + "not_a_publishable_model_credential'",
        '{"accessKeySecret": "' + "not_a_publishable_cloud_credential" + '"}',
        "access_key_" + "id: " + "LTA" + "I" + "x" * 20,
        "AWS_SECRET_ACCESS_KEY = '" + "not_a_publishable_aws_credential'",
        'securityToken: "' + "not_a_publishable_cloud_session" + '"',
        'Authorization: "Bearer ' + "x" * 30 + '"',
        'url = "https://example.com?OSSAccess' + "KeyId=" + "x" * 20 + '"',
        'url = "https://account:' + "x" * 20 + '@example.com"',
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
        'key = "' + "sk-" + "x" * 30 + '"',
        'endpoint = "https://nebula2.' + "alibaba-inc" + '.com"',
        'bucket = "' + "oss://" + 'private-bucket/data"',
        'API_KEY = (\n "' + "not_a_publishable_model_credential" + '"\n)',
    ],
)
def test_credentials_and_private_locations_are_redacted(content: str):
    findings = content_findings("example.py", content.encode())
    assert findings
    assert content not in json.dumps(findings)
    assert all(set(item) <= {"kind", "path", "line"} for item in findings)


def test_runtime_key_lookup_and_placeholders_are_allowed():
    content = "\n".join(
        [
            'CCE_MODEL_API_KEY = "replace-with-your-model-token"',
            'client.api_key = get_secret("CCE_MODEL_API_KEY")',
            'headers = {"Authorization": f"Bearer {self.api_key}"}',
            'settings = {"api_key_env": "CCE_MODEL_API_KEY"}',
            'kwargs = {"api_key": self.model_config.api_key}',
            "apiKey: ${MODEL_TOKEN}",
        ]
    )
    assert not content_findings("example.py", content.encode())


def test_reviewed_invalid_test_header_does_not_allow_arbitrary_credentials():
    name = "tests/integration/test_public_workflow.py"
    header = "Author" + "ization"
    assert not content_findings(name, json.dumps({header: "secret"}).encode())
    assert content_findings(name, json.dumps({header: "not_a_publishable_credential"}).encode())
    assert content_findings("example.py", json.dumps({header: "secret"}).encode())


@pytest.mark.parametrize(
    "name",
    [
        "src/cce_mobility/local_secrets.py",
        ".aws/credentials",
        ".ossutilconfig",
        "config/.env",
        "cert/key.pem",
        "data/rows.csv",
        "tests/fixtures/real_rows.jsonl",
        "configs/local/model.json",
        "logs/record.jsonl",
    ],
)
def test_private_paths_and_unreviewed_data_are_blocked(name: str):
    assert path_findings(name)


def test_named_synthetic_data_files_are_allowed():
    assert not path_findings("tests/fixtures/mobilitybench_public_mock.csv")
    assert not path_findings("examples/travelplanner_synthetic/validation_ref_info.jsonl")


def test_unreviewed_binary_is_blocked():
    assert content_findings("example.bin", b"\0payload")


def test_known_key_is_found_without_a_recognizable_name_or_prefix(tmp_path: Path):
    opaque_value = "synthetic_opaque_credential"
    local = tmp_path / "local_secrets.py"
    local.write_text(
        'CCE_MODEL_API_KEY = "' + opaque_value + '"\nraise RuntimeError("must not import")\n'
    )
    values = read_known_secrets([local])
    assert values == {opaque_value.encode()}
    findings = content_findings(
        "example.txt", f"unmarked {opaque_value}".encode(), known_secrets=values
    )
    assert findings and opaque_value not in json.dumps(findings)


def test_invalid_local_key_file_never_echoes_syntax_error_contents(tmp_path: Path):
    local = tmp_path / "local_secrets.py"
    local.write_text('API_KEY = "' + "not_a_publishable_credential")
    with pytest.raises(ValueError, match="raw contents are withheld") as error:
        read_known_secrets([local])
    assert "not_a_publishable_credential" not in str(error.value)


def test_png_metadata_and_appended_payload_are_blocked():
    import struct

    def chunk(kind, payload=b""):
        return struct.pack(">I", len(payload)) + kind + payload + b"\0" * 4

    header = b"\x89PNG\r\n\x1a\n"
    assert content_findings("assets/method.png", header + chunk(b"tEXt", b"author"))
    assert content_findings("assets/method.png", header + chunk(b"IEND") + b"extra")


def _git(root: Path, *args: str):
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _repository(tmp_path: Path, *, bootstrap: dict[str, str] | None = None) -> Path:
    root = tmp_path / "repository"
    root.mkdir()
    _git(root, "init")
    _git(root, "config", "user.name", "Audit Test")
    _git(root, "config", "user.email", "audit@example.com")
    if bootstrap:
        for name, content in bootstrap.items():
            (root / name).write_text(content)
        _git(root, "add", ".")
        _git(root, "commit", "-m", "Initial documentation fixture")
    (root / "tools").mkdir()
    (root / "tools/public_files.txt").write_text(
        "tools/public_files.txt\nexample.py\n" + "".join(f"{name}\n" for name in bootstrap or {})
    )
    (root / "example.py").write_text("VALUE = 1\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "Initial synthetic fixture")
    return root


def test_documentation_bootstrap_history_is_scanned_without_a_manifest(tmp_path: Path):
    root = _repository(
        tmp_path,
        bootstrap={
            "README.md": "# Synthetic project\n",
            "LICENSE": "Synthetic license fixture\n",
            ".gitignore": "__pycache__/\n",
        },
    )
    report = audit_repository(root)
    assert report["status"] == "passed"
    assert report["reachable_commit_count"] == 2
    assert report["documentation_bootstrap_commit_count"] == 1


def test_documentation_bootstrap_does_not_exempt_deleted_credentials(tmp_path: Path):
    root = _repository(
        tmp_path,
        bootstrap={"README.md": "API_KEY = '" + "not_a_publishable_credential'\n"},
    )
    (root / "README.md").write_text("# Synthetic project\n")
    _git(root, "add", "README.md")
    _git(root, "commit", "-m", "Remove synthetic credential fixture")
    report = audit_repository(root)
    assert report["status"] == "failed"
    assert report["documentation_bootstrap_commit_count"] == 1
    assert any(item["scope"] == "history" for item in report["findings"])
    assert all(item["scope"] == "history" for item in report["findings"])


def test_source_history_without_a_manifest_is_not_documentation_bootstrap(tmp_path: Path):
    root = _repository(tmp_path, bootstrap={"example.py": "VALUE = 0\n"})
    report = audit_repository(root)
    assert report["status"] == "failed"
    assert report["documentation_bootstrap_commit_count"] == 0
    assert any(item["kind"] == "missing publication allowlist" for item in report["findings"])


def test_deleted_secret_is_still_detected_in_history(tmp_path: Path):
    root = _repository(tmp_path)
    (root / "example.py").write_text("API_KEY = '" + "not_a_publishable_credential'\n")
    _git(root, "add", "example.py")
    _git(root, "commit", "-m", "Synthetic credential fixture")
    (root / "example.py").write_text("VALUE = 2\n")
    _git(root, "add", "example.py")
    _git(root, "commit", "-m", "Remove synthetic credential fixture")
    report = audit_repository(root)
    assert report["status"] == "failed"
    assert report["reachable_commit_count"] == 3
    assert any(item["scope"] == "history" for item in report["findings"])
    assert all(item["scope"] == "history" for item in report["findings"])
    assert "not_a_publishable_credential" not in json.dumps(report)


def test_staged_secret_is_detected_even_if_worktree_is_clean(tmp_path: Path):
    root = _repository(tmp_path)
    (root / "example.py").write_text("API_KEY = '" + "not_a_publishable_credential'\n")
    _git(root, "add", "example.py")
    (root / "example.py").write_text("VALUE = 2\n")
    report = audit_repository(root)
    assert report["status"] == "failed"
    assert any(item["scope"] == "index" for item in report["findings"])
    assert not any(item["scope"] == "worktree" for item in report["findings"])


def test_ignored_secret_is_not_a_publication_payload_but_force_add_is_blocked(tmp_path: Path):
    root = _repository(tmp_path)
    (root / ".gitignore").write_text("local_secrets.py\n")
    (root / "local_secrets.py").write_text("API_KEY = '" + "not_a_publishable_credential'\n")
    clean = audit_repository(root)
    assert clean["status"] == "passed"
    assert clean["ignored_file_count"] == 1
    _git(root, "add", "--force", "local_secrets.py")
    assert audit_repository(root)["status"] == "failed"


def test_unlisted_untracked_file_blocks_publication(tmp_path: Path):
    root = _repository(tmp_path)
    (root / "extra.py").write_text("VALUE = 2\n")
    report = audit_repository(root)
    assert report["status"] == "failed"
    assert report["untracked_file_count"] == 1
