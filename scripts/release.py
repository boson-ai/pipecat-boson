# ruff: noqa: CPY001
"""Release helpers shared by the release PR and the Release workflow.

Works on the repository this script is in, from any directory:

    python scripts/release.py bump patch|minor
        Bumps ``version`` in pyproject.toml and ``__version__``, and dates the
        ``[Unreleased]`` CHANGELOG section as the new version under a new empty
        ``[Unreleased]``. Prints the new version. Changes nothing and exits 1 if
        ``[Unreleased]`` is empty.

    python scripts/release.py notes VERSION
        Prints the CHANGELOG section dated for VERSION, the release notes.
        Exits 1 if there is none.

Standard library only, so it runs without the project installed.
"""

from __future__ import annotations

import datetime
import re
import sys
from pathlib import Path


UNRELEASED = "## [Unreleased]\n"
NEXT_SECTION = "\n## ["

BUMPS = {
    "patch": lambda major, minor, patch: (major, minor, patch + 1),
    "minor": lambda major, minor, patch: (major, minor + 1, 0),
}


def bump_version(version: str, part: str) -> str:
    """``version`` with its ``part`` ("patch" or "minor") bumped."""

    major, minor, patch = (int(number) for number in version.split("."))
    return ".".join(str(number) for number in BUMPS[part](major, minor, patch))


def replace_once(text: str, pattern: str, replacement: str) -> str:
    """``text`` with the single line matching ``pattern`` replaced; raises unless exactly one matches."""

    new_text, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one line matching {pattern!r}, found {count}")
    return new_text


def date_changelog(text: str, version: str, date: datetime.date) -> str:
    """``text`` with the ``[Unreleased]`` notes moved under ``## [version] - date``."""

    head, found, rest = text.partition(UNRELEASED)
    notes, next_section, older = rest.partition(NEXT_SECTION)
    if not found or not notes.strip():
        raise ValueError("CHANGELOG.md has nothing under [Unreleased]")
    return f"{head}{UNRELEASED}\n## [{version}] - {date.isoformat()}\n{notes}{next_section}{older}"


def release_notes(text: str, version: str) -> str:
    """The notes of the CHANGELOG section dated for ``version``."""

    heading = re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\n", text, flags=re.MULTILINE)
    if heading is None:
        raise ValueError(f"CHANGELOG.md has no dated section for {version}")
    notes = text[heading.end() :].partition(NEXT_SECTION)[0].strip()
    if not notes:
        raise ValueError(f"CHANGELOG.md has nothing under {version}")
    return notes + "\n"


def bump(root: Path, part: str, today: datetime.date) -> str:
    """Bump the version and date the CHANGELOG in ``root``; writes only once every file is ready."""

    pyproject = root / "pyproject.toml"
    init = root / "pipecat_boson" / "__init__.py"
    changelog = root / "CHANGELOG.md"

    current = re.search(r'^version = "([^"]+)"$', pyproject.read_text(), flags=re.MULTILINE)
    if current is None:
        raise ValueError("pyproject.toml has no version")
    version = bump_version(current.group(1), part)

    updates = {
        pyproject: replace_once(pyproject.read_text(), r'^version = ".*"$', f'version = "{version}"'),
        init: replace_once(init.read_text(), r'^__version__ = ".*"$', f'__version__ = "{version}"'),
        changelog: date_changelog(changelog.read_text(), version, today),
    }
    for path, text in updates.items():
        path.write_text(text)
    return version


def main(argv: list[str], root: Path) -> int:
    commands = {
        ("bump", 1): lambda part: print(bump(root, part, datetime.date.today())),
        ("notes", 1): lambda version: print(release_notes((root / "CHANGELOG.md").read_text(), version), end=""),
    }
    command = commands.get((argv[0], len(argv) - 1)) if argv else None
    if command is None or (argv[0] == "bump" and argv[1] not in BUMPS):
        print(__doc__, file=sys.stderr)
        return 2
    try:
        command(*argv[1:])
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:], Path(__file__).resolve().parents[1]))
