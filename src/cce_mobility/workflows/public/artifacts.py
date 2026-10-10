from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_locked(path: Path, value: Any) -> None:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Immutable artifact differs; use a new output directory: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256(path), "bytes": path.stat().st_size}


def source_fingerprint() -> str:
    package = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        if path.name == "local_secrets.py":
            continue
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


@contextmanager
def stage_lock(root: Path):
    # One writer per experiment. The operating system releases the lock on exit.
    import fcntl

    root.mkdir(parents=True, exist_ok=True)
    with (root / ".writer.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing this experiment: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
