"""Offline publication checks. Findings never include matched secret values."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import struct
import subprocess
from pathlib import Path, PurePosixPath


FORBIDDEN_DIRECTORIES = {
    ".aws",
    ".azure",
    ".git",
    ".kube",
    ".oss",
    ".ssh",
    ".venv",
    "amapdata",
    "benchmarks",
    "data",
    "logs",
    "outputs",
    "runs",
    "third_party",
}
REVIEWED_DATA_FILES = {
    "examples/travelplanner_synthetic/validation.csv",
    "examples/travelplanner_synthetic/validation_ref_info.jsonl",
    "tests/fixtures/mobilitybench_public_mock.csv",
}
REVIEWED_IMAGES = {"assets/method.png", "assets/motivation.png"}
DOCUMENTATION_BOOTSTRAP_FILES = {".gitignore", "LICENSE", "README.md"}
# Audited EXIF contains only image dimensions, color space and resolution.
REVIEWED_EXIF = {
    "b4bb53b22c0838141a46d47ee19d0005e26f0744dd86cef4a0576e974df3c20d",
    "fdc9b03ba4ec1a2644593fac9bc635fc0f2188ee7b2ee6dc90c31b33d7a935f9",
}
SECRET_NAMES = {
    "apikey",
    "accesskey",
    "accesskeyid",
    "accesskeysecret",
    "secretaccesskey",
    "securitytoken",
    "sessiontoken",
    "authtoken",
    "authorization",
    "bearertoken",
    "password",
    "passwd",
    "secret",
    "token",
    "clientsecret",
    "privatekey",
}
CONTENT_PATTERNS = {
    "cloud access key": re.compile(r"\b(?:LTAI[A-Za-z0-9]{16,}|(?:AKIA|ASIA)[A-Z0-9]{16})\b"),
    "provider or Git token": re.compile(
        r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}"
        r"|github_pat_[A-Za-z0-9_]{20,}|AIza[A-Za-z0-9_-]{30,})\b"
    ),
    "private key block": re.compile(r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----"),
    "internal service address": re.compile(
        r"\b(?:[a-z0-9-]+\.)*(?:alibaba-inc|aliyun-inc|antfin)\.com\b"
        r"|ai-llm-gateway\.amap\.com",
        re.I,
    ),
    "personal filesystem path": re.compile(r"/(?:Users|home)/[A-Za-z0-9_.-]+/"),
    "credential-bearing URL": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s/:'\"]+:[^\s/@]+@", re.I),
    "signed cloud URL": re.compile(
        r"[?&](?:OSSAccessKeyId|AWSAccessKeyId|X-Amz-Credential|X-Amz-Signature"
        r"|X-Amz-Security-Token|x-oss-signature|security-token|Signature)=[^&\s'\"]+",
        re.I,
    ),
    "cloud bucket location": re.compile(r"\b(?:oss|s3)://[A-Za-z0-9][^\s'\"<>]+", re.I),
    "cloud bucket endpoint": re.compile(
        r"\b[a-z0-9-]+\.(?:oss[-.][a-z0-9.-]+\.aliyuncs\.com|s3[.-][a-z0-9.-]+\.amazonaws\.com)",
        re.I,
    ),
    "literal bearer credential": re.compile(r"\bBearer\s+[A-Za-z0-9._+/=-]{16,}"),
}
ASSIGNMENT = re.compile(
    r"(?<![\w])['\"]?(?P<name>[A-Za-z_][A-Za-z0-9_.-]*)['\"]?\s*[:=]\s*"
    r"(?:(?P<quote>['\"])(?P<quoted>[^'\"\r\n]+)(?P=quote)"
    r"|(?P<bare>[A-Za-z0-9_+/.-]{16,}))"
)
MAX_FILE_BYTES = 10 * 1024 * 1024


def _secret_name(name: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", name.lower())
    return normalized in {"ak", "sk", "ossak", "osssk", "aihubak"} or any(
        normalized.endswith(suffix) for suffix in SECRET_NAMES
    )


def _placeholder(value: str) -> bool:
    value = value.strip().lower()
    return (
        not value
        or value.startswith(
            ("replace-", "your-", "example", "placeholder", "${", "{{", "<", "{self.")
        )
        or value in {"none", "null", "true", "false"}
    )


def _reviewed_test_value(name: str, key: str, value: str) -> bool:
    # This deliberately invalid header tests configuration rejection, not authentication.
    return (
        name == "tests/integration/test_public_workflow.py"
        and key.lower() == "authorization"
        and value == "secret"
    )


def _finding(kind: str, path: str, line: int | None = None) -> dict:
    finding = {"kind": kind, "path": path}
    if line is not None:
        finding["line"] = line
    return finding


def path_findings(name: str) -> list[dict]:
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in name:
        return [_finding("unsafe publication path", name)]
    if (
        path.parts[0] in FORBIDDEN_DIRECTORIES
        or path.parts[:2] in {("configs", "generated"), ("configs", "local")}
        or path.name in {"local_secrets.py", ".env", ".netrc", ".ossutilconfig", "credentials"}
        or (path.name.startswith(".env.") and path.name != ".env.example")
        or path.suffix.lower() in {".pem", ".key", ".p12", ".pfx", ".jks"}
        or path.name in {"id_rsa", "id_ed25519", "id_ecdsa"}
    ):
        return [_finding("private file or runtime artifact", name)]
    if path.suffix.lower() in {".csv", ".jsonl", ".parquet", ".sqlite", ".db", ".jsonlines"}:
        if name not in REVIEWED_DATA_FILES:
            return [_finding("unreviewed dataset or trace", name)]
    return []


def _png_findings(name: str, data: bytes) -> list[dict]:
    if name not in REVIEWED_IMAGES or not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [_finding("unreviewed binary file", name)]
    offset = 8
    while offset + 12 <= len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            break
        if (
            chunk == b"eXIf"
            and hashlib.sha256(data[offset + 8 : end - 4]).hexdigest() in REVIEWED_EXIF
        ):
            offset = end
            continue
        if chunk in {b"tEXt", b"zTXt", b"iTXt", b"eXIf"}:
            return [_finding("image metadata requires review", name)]
        if chunk == b"IEND":
            return [] if end == len(data) else [_finding("data appended after image", name)]
        offset = end
    return [_finding("malformed PNG asset", name)]


def content_findings(
    name: str, data: bytes, *, known_secrets: set[bytes] | None = None
) -> list[dict]:
    if any(value in data for value in known_secrets or ()):
        return [_finding("known local credential value", name)]
    if len(data) > MAX_FILE_BYTES:
        return [_finding("oversized file requires review", name)]
    if PurePosixPath(name).suffix.lower() == ".png":
        return _png_findings(name, data)
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return [_finding("unreviewed binary file", name)]
    if "\x00" in text:
        return [_finding("unreviewed binary file", name)]
    findings = []
    for number, line in enumerate(text.splitlines(), 1):
        for kind, pattern in CONTENT_PATTERNS.items():
            if pattern.search(line):
                findings.append(_finding(kind, name, number))
        for match in ASSIGNMENT.finditer(line):
            if name.endswith(".py") and not match.group("quote"):
                continue
            value = match.group("quoted") or match.group("bare") or ""
            if (
                _secret_name(match.group("name"))
                and not _placeholder(value)
                and not _reviewed_test_value(name, match.group("name"), value)
            ):
                findings.append(_finding("literal credential assignment", name, number))
    # Also catch Python multiline assignments/dictionaries without importing local code.
    if name.endswith(".py"):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            tree = None
        for node in ast.walk(tree) if tree else ():
            pairs = []
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                for target in targets:
                    if isinstance(target, ast.Name):
                        pairs.append((target.id, node.value))
                    elif isinstance(target, ast.Attribute):
                        pairs.append((target.attr, node.value))
            elif isinstance(node, ast.Dict):
                pairs = [
                    (key.value, value)
                    for key, value in zip(node.keys, node.values)
                    if isinstance(key, ast.Constant) and isinstance(key.value, str)
                ]
            elif isinstance(node, ast.keyword) and node.arg:
                pairs = [(node.arg, node.value)]
            for key, value_node in pairs:
                if not _secret_name(key):
                    continue
                try:
                    value = ast.literal_eval(value_node)
                except (ValueError, TypeError, SyntaxError):
                    continue
                if (
                    isinstance(value, str)
                    and not _placeholder(value)
                    and not _reviewed_test_value(name, key, value)
                ):
                    findings.append(_finding("literal credential assignment", name, node.lineno))
    return list({tuple(finding.items()): finding for finding in findings}.values())


def _git(root: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    if result.returncode:
        raise ValueError("Git audit command failed; no raw command output is displayed")
    return result.stdout


def _names(data: bytes) -> list[str]:
    return [name.decode("utf-8") for name in data.split(b"\0") if name]


def _allowed(text: str) -> set[str]:
    return {
        line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#")
    } | {".gitignore", "release_manifest.json"}


def _blobs(root: Path, oids: set[str]) -> dict[str, bytes]:
    if not oids:
        return {}
    result = subprocess.run(
        ["git", "-C", str(root), "cat-file", "--batch"],
        input="".join(oid + "\n" for oid in sorted(oids)).encode(),
        capture_output=True,
        check=False,
    )
    if result.returncode:
        raise ValueError("Git blob audit failed; raw output is withheld")
    blobs = {}
    offset = 0
    while offset < len(result.stdout):
        end = result.stdout.index(b"\n", offset)
        parts = result.stdout[offset:end].split()
        if len(parts) != 3 or parts[1] != b"blob":
            raise ValueError("Publication input contains an unavailable or non-blob object")
        oid, _, size = parts
        start = end + 1
        offset = start + int(size) + 1
        blobs[oid.decode()] = result.stdout[start : offset - 1]
    return blobs


def _tree_entries(data: bytes) -> list[tuple[str, str, str, str]]:
    entries = []
    for entry in filter(None, data.split(b"\0")):
        metadata, name = entry.split(b"\t", 1)
        entries.append((*metadata.decode().split(), name.decode()))
    return entries


def read_known_secrets(paths: list[Path]) -> set[bytes]:
    values = set()
    for path in paths:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError, OSError):
            raise ValueError("Cannot parse local key file; raw contents are withheld") from None
        for node in ast.walk(tree):
            if not isinstance(node, (ast.Assign, ast.AnnAssign)):
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if not any(
                isinstance(target, ast.Name) and _secret_name(target.id) for target in targets
            ):
                continue
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError, SyntaxError):
                continue
            if isinstance(value, str) and len(value) >= 8 and not _placeholder(value):
                values.add(value.encode())
    return values


def audit_repository(
    root: Path, *, history: bool = True, known_secrets: set[bytes] | None = None
) -> dict:
    root = root.resolve()
    if Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        raise ValueError("Audit root must be the Git repository root")
    allowed = _allowed((root / "tools/public_files.txt").read_text(encoding="utf-8"))
    index = _names(_git(root, "ls-files", "-z"))
    untracked = _names(_git(root, "ls-files", "--others", "--exclude-standard", "-z"))
    ignored = _names(_git(root, "ls-files", "--others", "--ignored", "--exclude-standard", "-z"))
    index_entries = _tree_entries(_git(root, "ls-files", "--stage", "-z"))
    commits = _git(root, "rev-list", "--all").decode().splitlines() if history else []
    trees = {commit: _tree_entries(_git(root, "ls-tree", "-r", "-z", commit)) for commit in commits}
    oids = {oid for mode, oid, stage, _ in index_entries if mode in {"100644", "100755"}}
    oids.update(oid for entries in trees.values() for _, kind, oid, _ in entries if kind == "blob")
    blob_contents = _blobs(root, oids)
    findings = []
    review_notes = []
    for name in sorted(set(index + untracked)):
        current = path_findings(name)
        if name not in allowed:
            current.append(_finding("file outside publication allowlist", name))
        path = root / name
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            current.append(_finding("unsafe symlink", name))
        elif path.is_file():
            current.extend(content_findings(name, path.read_bytes(), known_secrets=known_secrets))
        else:
            current.append(_finding("missing publication file", name))
        findings.extend({"scope": "worktree", **item} for item in current)
    # The index can differ from both the worktree and HEAD. Audit its exact blobs too.
    for mode, oid, stage, name in index_entries:
        current = path_findings(name)
        if name not in allowed:
            current.append(_finding("file outside publication allowlist", name))
        if mode not in {"100644", "100755"} or stage != "0":
            current.append(_finding("unsafe index entry", name))
        else:
            current.extend(content_findings(name, blob_contents[oid], known_secrets=known_secrets))
        findings.extend({"scope": "index", **item} for item in current)
    blobs = set()
    shallow = _git(root, "rev-parse", "--is-shallow-repository").decode().strip() == "true"
    if history and shallow:
        findings.append({"scope": "history", "kind": "shallow history is incomplete"})
    bootstrap_commit_count = 0
    for commit in commits:
        identities = _git(root, "show", "-s", "--format=%ae%n%ce", commit).decode()
        if CONTENT_PATTERNS["internal service address"].search(identities):
            review_notes.append({"kind": "corporate email in commit identity", "commit": commit})
        message = _git(root, "show", "-s", "--format=%B", commit)
        for item in content_findings("commit-message.txt", message, known_secrets=known_secrets):
            findings.append({"scope": "history", "commit": commit, **item})
        manifest_oid = next(
            (
                oid
                for _, kind, oid, name in trees[commit]
                if kind == "blob" and name == "tools/public_files.txt"
            ),
            None,
        )
        if manifest_oid is None:
            historical_paths = {name for _, _, _, name in trees[commit]}
            # Hosted repositories may start with documentation before importing source.
            if historical_paths and historical_paths <= DOCUMENTATION_BOOTSTRAP_FILES:
                historical_allowed = DOCUMENTATION_BOOTSTRAP_FILES
                bootstrap_commit_count += 1
            else:
                findings.append(
                    {"scope": "history", "commit": commit, "kind": "missing publication allowlist"}
                )
                historical_allowed = set()
        else:
            historical_allowed = _allowed(blob_contents[manifest_oid].decode("utf-8"))
        for mode, kind, oid, name in trees[commit]:
            current = path_findings(name)
            if name not in historical_allowed:
                current.append(_finding("historical file outside publication allowlist", name))
            if kind != "blob" or mode not in {"100644", "100755"}:
                current.append(_finding("unsafe historical entry", name))
            elif (name, oid) not in blobs:
                current.extend(
                    content_findings(name, blob_contents[oid], known_secrets=known_secrets)
                )
                blobs.add((name, oid))
            findings.extend({"scope": "history", "commit": commit, **item} for item in current)
    tag_count = 0
    if history:
        tags = _git(
            root, "for-each-ref", "--format=%(objectname) %(objecttype)", "refs/tags"
        ).decode()
        for tag in tags.splitlines():
            oid, kind = tag.split()
            if kind == "tag":
                tag_count += 1
                for item in content_findings(
                    "annotated-tag.txt",
                    _git(root, "cat-file", "tag", oid),
                    known_secrets=known_secrets,
                ):
                    findings.append({"scope": "history", "object": oid, **item})
    return {
        "schema_version": "cce-publication-security-audit-v1",
        "status": "failed" if findings else "passed",
        "head": _git(root, "rev-parse", "HEAD").decode().strip(),
        "tracked_file_count": len(index),
        "untracked_file_count": len(untracked),
        "ignored_file_count": len(ignored),
        "reachable_commit_count": len(commits),
        "documentation_bootstrap_commit_count": bootstrap_commit_count,
        "history_blob_path_count": len(blobs),
        "annotated_tag_count": tag_count,
        "known_credential_count": len(known_secrets or ()),
        "history_requested": history,
        "shallow": shallow,
        "findings": findings,
        "identity_review_status": "needs_review" if review_notes else "passed",
        "review_notes": review_notes,
        "limitations": [
            "Ignored local files and Git remote configuration are not publication payloads.",
            "Pattern checks cannot certify code ownership or detect every possible secret.",
            "Image pixels and synthetic data contents require separate human review.",
            "No network requests, credential probes or model calls are performed.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[4])
    parser.add_argument("--no-history", action="store_true", help="Audit worktree/index only")
    parser.add_argument(
        "--known-secrets-file",
        type=Path,
        action="append",
        default=[],
        help="Optional local Python key file; parse literals without importing or displaying keys",
    )
    args = parser.parse_args()
    report = audit_repository(
        args.root,
        history=not args.no_history,
        known_secrets=read_known_secrets(args.known_secrets_file),
    )
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
