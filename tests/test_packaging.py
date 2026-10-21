# ruff: noqa: CPY001
from __future__ import annotations

import tomllib
from pathlib import Path

from pipecat_boson import __version__


def test_package_version_matches_project_metadata():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert __version__ == pyproject["project"]["version"]


def test_package_depends_on_supported_pipecat_runtime():
    pyproject = tomllib.loads((Path(__file__).parents[1] / "pyproject.toml").read_text())

    assert "pipecat-ai>=1.4.0,<2" in pyproject["project"]["dependencies"]
    assert "pipecat-ai[runner,webrtc]>=1.4.0,<2" in pyproject["project"]["optional-dependencies"]["webrtc"]


def test_manifest_declares_existing_community_integration_files():
    repo_root = Path(__file__).parents[1]
    manifest = (repo_root / "MANIFEST.in").read_text()

    assert "include CHANGELOG.md" in manifest
    assert "recursive-include examples *.py" in manifest

    assert (repo_root / "CHANGELOG.md").is_file()
    assert (repo_root / "examples/pipecat_boson_realtime_agent.py").is_file()
