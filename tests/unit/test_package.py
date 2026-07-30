"""Package metadata tests."""

from importlib import metadata, resources

from pytest import MonkeyPatch

import nneditor
from nneditor import __main__ as package_main


def test_version_is_exposed() -> None:
    assert nneditor.__version__ == "1.1.1"


def test_distribution_metadata_matches_package() -> None:
    distribution = metadata.distribution("nneditor")
    console_scripts = {
        entry.name: entry.value
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    project_urls = distribution.metadata.get_all("Project-URL") or []
    license_files = distribution.metadata.get_all("License-File") or []

    assert distribution.version == nneditor.__version__
    assert distribution.metadata["License-Expression"] == "Apache-2.0"
    assert set(license_files) == {"LICENSE", "NOTICE"}
    assert console_scripts["nneditor"] == "nneditor.ui.app:cli"
    assert "Repository, https://github.com/ibouazizi/nneditor" in project_urls


def test_package_is_marked_as_typed() -> None:
    assert resources.files(nneditor).joinpath("py.typed").is_file()


def test_module_entry_point_delegates_to_app(monkeypatch: MonkeyPatch) -> None:
    calls = 0

    def fake_cli() -> None:
        nonlocal calls
        calls += 1

    monkeypatch.setattr(package_main, "cli", fake_cli)

    package_main.main()

    assert calls == 1
