from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MIGRATION_PLAN = (
    REPOSITORY_ROOT
    / "docs/superpowers/plans/2026-07-17-greenhouse-energy-hub-architecture-migration.md"
)


def _tracked_text_files() -> list[tuple[Path, str]]:
    completed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    files: list[tuple[Path, str]] = []
    for encoded_path in completed.stdout.split(b"\0"):
        if not encoded_path:
            continue
        path = REPOSITORY_ROOT / encoded_path.decode("utf-8")
        try:
            contents = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, IsADirectoryError):
            continue
        files.append((path, contents))
    return files


def test_tracked_text_omits_private_development_metadata() -> None:
    forbidden = (
        (
            "absolute home-directory path",
            re.compile(r"/(?:" + "Users|home" + r")/[A-Za-z0-9._-]+/"),
            False,
        ),
        ("local review draft", re.compile("REVIEW_" + r"RESPONSE\.md"), True),
        (
            "internal agent instruction",
            re.compile("For " + "agentic workers"),
            False,
        ),
        (
            "internal goal instruction",
            re.compile("active " + "Codex goal|goal token " + "usage"),
            False,
        ),
    )

    violations: list[str] = []
    for path, contents in _tracked_text_files():
        relative_path = path.relative_to(REPOSITORY_ROOT)
        for label, pattern, docs_only in forbidden:
            if docs_only and relative_path.parts[0] != "docs":
                continue
            if match := pattern.search(contents):
                line = contents.count("\n", 0, match.start()) + 1
                violations.append(f"{relative_path}:{line}: {label}")

    assert not violations, "private development metadata is tracked:\n" + "\n".join(
        violations
    )


def test_migration_plan_keeps_portable_repository_instructions() -> None:
    contents = MIGRATION_PLAN.read_text(encoding="utf-8")

    assert "Work from the repository root." in contents
    assert "cd - >/dev/null" in contents
