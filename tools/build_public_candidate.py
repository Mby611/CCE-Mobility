"""Compatibility entry point for the packaged workflow."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
_impl = import_module("cce_mobility.workflows.release.build_public_candidate")
main = _impl.main


def __getattr__(name: str):
    return getattr(_impl, name)


if __name__ == "__main__":
    raise SystemExit(main())
