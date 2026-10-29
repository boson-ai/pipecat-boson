# ruff: noqa: CPY001
from __future__ import annotations

import datetime
import importlib.util
import re
import shutil
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[1]
_spec = importlib.util.spec_from_file_location("release_script", ROOT / "scripts" / "release.py")
release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(release)

TODAY = datetime.date(2026, 11, 5)
CHANGELOG = """# Changelog

## [Unreleased]

### Fixed

- A fix.

## [0.1.4] - 2026-10-22

### Fixed

- An older fix.
"""


@pytest.mark.parametrize(
    ("version", "part", "expected"),
    [("0.1.4", "patch", "0.1.5"), ("0.1.4", "minor", "0.2.0"), ("1.9.9", "patch", "1.9.10")],
)
def test_bump_version(version, part, expected):
    assert release.bump_version(version, part) == expected


def test_date_changelog_moves_unreleased_notes_under_the_new_version():
    dated = release.date_changelog(CHANGELOG, "0.1.5", TODAY)

    assert "## [Unreleased]\n\n## [0.1.5] - 2026-11-05\n\n### Fixed\n\n- A fix.\n\n## [0.1.4]" in dated
    assert release.release_notes(dated, "0.1.5") == "### Fixed\n\n- A fix.\n"


def test_date_changelog_refuses_an_empty_unreleased_section():
    empty = release.date_changelog(CHANGELOG, "0.1.5", TODAY)

    with pytest.raises(ValueError, match="nothing under \\[Unreleased\\]"):
        release.date_changelog(empty, "0.1.6", TODAY)


def test_release_notes_needs_a_dated_section():
    assert release.release_notes(CHANGELOG, "0.1.4") == "### Fixed\n\n- An older fix.\n"
    with pytest.raises(ValueError, match="no dated section for 0.1.5"):
        release.release_notes(CHANGELOG, "0.1.5")


def test_bump_updates_the_three_files_together(tmp_path):
    for name in ("pyproject.toml", "CHANGELOG.md", "pipecat_boson/__init__.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "CHANGELOG.md").write_text(CHANGELOG)
    current = re.search(r'^version = "(.+)"$', (ROOT / "pyproject.toml").read_text(), re.MULTILINE).group(1)

    version = release.bump(tmp_path, "patch", TODAY)

    assert version == release.bump_version(current, "patch")
    assert f'\nversion = "{version}"\n' in (tmp_path / "pyproject.toml").read_text()
    assert f'__version__ = "{version}"' in (tmp_path / "pipecat_boson/__init__.py").read_text()
    assert release.release_notes((tmp_path / "CHANGELOG.md").read_text(), version) == "### Fixed\n\n- A fix.\n"


def test_bump_changes_nothing_when_unreleased_is_empty(tmp_path):
    for name in ("pyproject.toml", "pipecat_boson/__init__.py"):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, tmp_path / name)
    (tmp_path / "CHANGELOG.md").write_text(release.date_changelog(CHANGELOG, "0.1.5", TODAY))
    before = {path: path.read_text() for path in tmp_path.rglob("*") if path.is_file()}

    assert release.main(["bump", "patch"], tmp_path) == 1
    assert {path: path.read_text() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_main_rejects_an_unknown_command(tmp_path):
    assert release.main(["bump", "major"], tmp_path) == 2
    assert release.main([], tmp_path) == 2
