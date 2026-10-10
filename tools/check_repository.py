"""Offline checks for the public source tree and bilingual documentation."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cce_mobility.workflows.release.build_public_candidate import preflight  # noqa: E402


def numeric_tables(text: str) -> list[list[str]]:
    return [
        [cell.strip().strip("*") for cell in line.strip("| ").split("|")]
        for line in text.splitlines()
        if line.startswith("|") and re.search(r"\d\.\d", line)
    ]


def main() -> int:
    files = preflight(ROOT, ROOT / "tools/public_files.txt")
    english = (ROOT / "README.md").read_text()
    chinese = (ROOT / "README.zh-CN.md").read_text()

    def commands(text: str) -> list[str]:
        return re.findall(r"```bash\n(.*?)```", text, re.S)

    if commands(english) != commands(chinese):
        raise ValueError("English and Chinese README commands differ")
    if numeric_tables(english) != numeric_tables(chinese):
        raise ValueError("English and Chinese README result values differ")
    expected = json.loads((ROOT / "assets/paper_tables_manifest.json").read_text())
    digest = hashlib.sha256(
        json.dumps(numeric_tables(english), ensure_ascii=False).encode()
    ).hexdigest()
    if digest != expected["numeric_tables_sha256"]:
        raise ValueError("README table values differ from the paper transcription manifest")
    published = {name for name, _ in files}
    for name, path in files:
        if re.search(r"(?:^|[_/])v2[1-9]|v2\.[1-9]", name):
            raise ValueError(f"Historical method filename is not public: {name}")
        if path.suffix == ".md":
            for target in re.findall(r"\]\(([^)]+)\)", path.read_text()):
                target = target.split("#", 1)[0]
                if not target or "://" in target or target.startswith("mailto:"):
                    continue
                resolved = (path.parent / target).resolve()
                if (
                    not resolved.is_file()
                    or not resolved.is_relative_to(ROOT)
                    or resolved.relative_to(ROOT).as_posix() not in published
                ):
                    raise ValueError(f"Missing published link in {name}: {target}")
    print(
        json.dumps(
            {
                "status": "passed",
                "listed_files": len(files),
                "bilingual_commands_match": True,
                "paper_tables_match": True,
                "local_documentation_links": "passed",
                "secret_internal_path_scan": "passed",
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
