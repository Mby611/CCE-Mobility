from pathlib import Path

import pytest

from tools.build_public_candidate import (
    build_candidate,
    preflight,
    verify_candidate,
)


def _fixture(tmp_path: Path, name: str, content: str) -> tuple[Path, Path]:
    source = tmp_path / "private"
    path = source / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    manifest = tmp_path / "public_files.txt"
    manifest.write_text(name + "\n", encoding="utf-8")
    return source, manifest


def test_candidate_hashes_explicit_files_without_private_history(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path, "src/example.py", "VALUE = 1\n")
    target = tmp_path / "candidate"
    dry_run = build_candidate(source, manifest, target, dry_run=True)
    assert dry_run["scope"] == "cce-mobility-unified"
    assert not target.exists()

    report = build_candidate(source, manifest, target)
    assert report["file_count"] == 2
    assert (target / "src/example.py").read_text() == "VALUE = 1\n"
    assert not (target / ".git").exists()
    assert (target / ".gitignore").is_file()
    assert verify_candidate(target)["file_count"] == 2
    with pytest.raises(FileExistsError):
        build_candidate(source, manifest, target)
    (target / "src/example.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="differs from manifest"):
        verify_candidate(target)


@pytest.mark.parametrize(
    "path",
    ["data/private.jsonl", "runs/records.jsonl", ".env", "src/cce_mobility/local_secrets.py"],
)
def test_forbidden_paths_fail_closed(tmp_path: Path, path: str) -> None:
    source, manifest = _fixture(tmp_path, path, "safe\n")
    with pytest.raises(ValueError, match="Forbidden publication path"):
        preflight(source, manifest)


@pytest.mark.parametrize(
    "content",
    [
        "url = 'https://ai-llm-" + "gateway.amap.com/open_api/v1'\n",
        "path = '/Users/" + "mby/private/file'\n",
        "API_KEY = '" + "very_long_actual_credential_123'\n",
        "-----BEGIN " + "PRIVATE KEY-----\n",
    ],
)
def test_sensitive_content_fails_without_echoing_value(tmp_path: Path, content: str) -> None:
    source, manifest = _fixture(tmp_path, "src/example.py", content)
    with pytest.raises(ValueError) as error:
        preflight(source, manifest)
    assert "src/example.py:1" in str(error.value)
    assert content.strip() not in str(error.value)


def test_placeholder_secret_is_allowed(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path, "src/example.py", "API_KEY = 'replace-with-token'\n")
    assert len(preflight(source, manifest)) == 1


def test_candidate_cannot_be_nested_in_private_repo(tmp_path: Path) -> None:
    source, manifest = _fixture(tmp_path, "src/example.py", "VALUE = 1\n")
    with pytest.raises(ValueError, match="outside"):
        build_candidate(source, manifest, source / "candidate")


def test_multiple_manifests_form_one_deduplicated_candidate(tmp_path: Path) -> None:
    source, tp_manifest = _fixture(tmp_path, "src/shared.py", "VALUE = 1\n")
    (source / "src/mb.py").write_text("MB = True\n", encoding="utf-8")
    mb_manifest = tmp_path / "mb_files.txt"
    mb_manifest.write_text("src/shared.py\nsrc/mb.py\n", encoding="utf-8")
    target = tmp_path / "unified"
    report = build_candidate(source, [tp_manifest, mb_manifest], target)
    assert report["scope"] == "cce-mobility-unified"
    assert report["file_count"] == 3
    assert (target / "src/shared.py").is_file()
    assert (target / "src/mb.py").is_file()
    assert verify_candidate(target)["file_count"] == 3


@pytest.mark.parametrize(
    "content",
    [
        "CI" + "E_MODEL_API_KEY = 'replace-with-token'\n",
        "class CI" + "ETravelPlannerAdapter: pass\n",
        "runtime = {'ci" + "e_runtime': 'example'}\n",
        "# CI" + "E counterfactual recovery\n",
    ],
)
def test_historical_namespace_is_rejected_in_public_content(tmp_path: Path, content: str):
    source, manifest = _fixture(tmp_path, "src/example.py", content)
    with pytest.raises(ValueError, match="historical project namespace"):
        preflight(source, manifest)


def test_historical_namespace_is_rejected_in_public_filenames(tmp_path: Path):
    source, manifest = _fixture(tmp_path, "src/ci" + "e_adapter.py", "VALUE = 1\n")
    with pytest.raises(ValueError, match="Historical project namespace"):
        preflight(source, manifest)


def test_namespace_scan_keeps_ordinary_english_words_and_current_name(tmp_path: Path):
    source, manifest = _fixture(
        tmp_path,
        "src/cce_adapter.py",
        "# CCE scientific dependencies and efficiency\nclass CCETravelPlannerAdapter: pass\n",
    )
    assert len(preflight(source, manifest)) == 1
