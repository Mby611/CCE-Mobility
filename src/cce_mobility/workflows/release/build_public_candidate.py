from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path, PurePosixPath

from cce_mobility.workflows.release.security_audit import content_findings, path_findings


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_MANIFEST = PROJECT_ROOT / "tools" / "public_files.txt"
PUBLIC_GITIGNORE = (
    ".venv/\n.uv-cache/\n.pytest_cache/\n.ruff_cache/\n__pycache__/\n*.py[cod]\n"
    "*.egg-info/\nbuild/\ndist/\n"
    ".env\n.env.*\n!.env.example\n"
    "src/cce_mobility/local_secrets.py\n"
    "local_secrets.py\n.aws/\n.azure/\n.kube/\n.ssh/\n.oss/\n.ossutilconfig\n.netrc\n"
    "*.pem\n*.key\n*.p12\n*.pfx\n"
    "configs/generated/\nconfigs/local/\nthird_party/\noutputs/\nruns/\nlogs/\ndata/\n"
)
FORBIDDEN_COMPONENTS = {
    ".git",
    ".venv",
    "amapdata",
    "benchmarks",
    "data",
    "logs",
    "runs",
    "outputs",
    "third_party",
}
FORBIDDEN_FILENAMES = {".env", "local_secrets.py"}
PROJECT_NAMESPACE_PATTERN = re.compile(r"(?<![A-Za-z0-9])[cC][iI][eE](?=$|[^A-Za-z0-9]|[A-Z][a-z])")
CONTENT_RULES = {
    "historical project namespace": PROJECT_NAMESPACE_PATTERN,
    "private filesystem path": re.compile(r"/(?:Users|home)/m(?:by|uboyang)(?:[./]|$)", re.I),
    "internal service address": re.compile(
        r"(?:[a-z0-9.-]+\.)?alibaba-inc\.com|ai-llm-gateway\.amap\.com",
        re.I,
    ),
    "private key block": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "credential assignment": re.compile(
        r"\b(?:[A-Z][A-Z0-9_]*_)?(?:API_KEY|ACCESS_KEY|TOKEN|PASSWORD|SECRET)\b"
        r"\s*[:=]\s*['\"]([^'\"\r\n]+)['\"]",
        re.I,
    ),
}


def _manifests(manifest: Path | list[Path]) -> list[Path]:
    return [manifest] if isinstance(manifest, Path) else manifest


def _listed_files(source_root: Path, manifest: Path | list[Path]) -> list[tuple[str, Path]]:
    root = source_root.resolve()
    files: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for manifest_path in _manifests(manifest):
        local_seen: set[str] = set()
        for line_number, raw in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            name = raw.strip()
            if not name or name.startswith("#"):
                continue
            relative = PurePosixPath(name)
            if PROJECT_NAMESPACE_PATTERN.search(name):
                raise ValueError(f"Historical project namespace in publication path: {name}")
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "." in relative.parts
                or "\\" in name
                or relative.as_posix() != name
            ):
                raise ValueError(f"Invalid manifest path at {manifest_path}:{line_number}: {name}")
            if name in local_seen:
                raise ValueError(f"Duplicate manifest path: {name}")
            local_seen.add(name)
            if relative.parts[0] in FORBIDDEN_COMPONENTS or relative.name in FORBIDDEN_FILENAMES:
                raise ValueError(f"Forbidden publication path: {name}")
            if relative.parts[:2] == ("configs", "generated"):
                raise ValueError(f"Generated config cannot be published: {name}")
            problems = path_findings(name)
            if problems:
                raise ValueError(f"{problems[0]['kind']} in publication path: {name}")
            path = source_root / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                raise ValueError(f"Missing or unsafe manifest file: {name}")
            if name not in seen:
                seen.add(name)
                files.append((name, path))
    if not files:
        raise ValueError("Public candidate manifest is empty")
    return files


def _scan_file(name: str, path: Path) -> None:
    problems = content_findings(name, path.read_bytes())
    if problems:
        first = problems[0]
        location = f"{name}:{first['line']}" if "line" in first else name
        raise ValueError(f"{first['kind']} in {location}")
    if name.startswith("assets/") and path.suffix == ".png":
        if path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"):
            return
        raise ValueError(f"Invalid PNG asset: {name}")
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Non-text file requires separate review: {name}") from exc
    for line_number, line in enumerate(content.splitlines(), 1):
        for label, pattern in CONTENT_RULES.items():
            match = pattern.search(line)
            if not match:
                continue
            if label == "credential assignment" and match.group(1).lower().startswith(
                ("replace-", "your-", "example", "placeholder")
            ):
                continue
            raise ValueError(f"{label} in {name}:{line_number}")


def preflight(source_root: Path, manifest: Path | list[Path]) -> list[tuple[str, Path]]:
    files = _listed_files(source_root, manifest)
    for name, path in files:
        _scan_file(name, path)
    return files


def build_candidate(
    source_root: Path,
    manifest: Path | list[Path],
    destination: Path,
    *,
    dry_run: bool = False,
) -> dict:
    files = preflight(source_root, manifest)
    destination = destination.expanduser().resolve()
    if destination == source_root.resolve() or destination.is_relative_to(source_root.resolve()):
        raise ValueError("Destination must be outside the private repository")
    if destination.exists():
        raise FileExistsError(f"Refusing to replace existing directory: {destination}")

    entries = [
        {
            "path": name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in files
    ]
    entries.append(
        {
            "path": ".gitignore",
            "sha256": hashlib.sha256(PUBLIC_GITIGNORE.encode("utf-8")).hexdigest(),
        }
    )
    report = {
        "schema_version": "cce-release-manifest-v1",
        "scope": "cce-mobility-unified",
        "source_manifests": [path.name for path in _manifests(manifest)],
        "file_count": len(entries),
        "files": entries,
    }
    if dry_run:
        return report

    destination.mkdir(parents=True)
    for name, path in files:
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (destination / ".gitignore").write_text(PUBLIC_GITIGNORE, encoding="utf-8")
    (destination / "release_manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def verify_candidate(destination: Path) -> dict:
    destination = destination.expanduser().resolve()
    report_path = destination / "release_manifest.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for entry in report["files"]:
        path = destination / entry["path"]
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"Candidate file differs from manifest: {entry['path']}")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a clean CCE-Mobility source snapshot without private Git history"
    )
    parser.add_argument("--manifest", type=Path, action="append")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output-dir", type=Path)
    destination.add_argument("--verify-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.verify_dir:
        report = verify_candidate(args.verify_dir)
    else:
        report = build_candidate(
            PROJECT_ROOT,
            args.manifest or DEFAULT_MANIFEST,
            args.output_dir,
            dry_run=args.dry_run,
        )
    print(json.dumps({"file_count": report["file_count"], "scope": report["scope"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
