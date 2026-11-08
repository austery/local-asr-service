"""Tests for the Markdown link check in scripts/check_doc_links.py."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from scripts.check_doc_links import (
    BrokenLink,
    find_broken_links,
    find_broken_links_in_text,
    is_maintained_markdown,
)

EXISTING = {"docs/ADR-002.md", "docs/plans/p.md", "pyproject.toml", "README.md"}


def check(text: str, file: str = "docs/SPEC.md") -> list[BrokenLink]:
    return find_broken_links_in_text(file, text, lambda path: path in EXISTING)


def test_relative_links_that_resolve_pass() -> None:
    text = "[a](ADR-002.md) [b](./plans/p.md#goal) [c](../pyproject.toml) ![i](ADR-002.md)"
    assert check(text) == []


def test_missing_target_is_reported_with_line() -> None:
    assert check("intro\n\n[gone](missing.md)") == [
        BrokenLink("docs/SPEC.md", 3, "missing.md", "missing")
    ]


def test_absolute_and_file_links_fail_even_when_target_exists() -> None:
    reasons = [b.reason for b in check("[a](/abs/README.md) [b](file:///x/README.md)")]
    assert reasons == ["machine-specific", "machine-specific"]


def test_link_leaving_repository_fails() -> None:
    assert [b.reason for b in check("[a](../../outside.md)")] == ["outside-repository"]


def test_urls_and_anchors_are_not_checked() -> None:
    assert check("[a](https://example.com/x) [b](#heading) [c](mailto:me@example.com)") == []


def test_editor_line_suffix_and_encoded_path_resolve() -> None:
    assert check("[a](ADR-002.md:12-20) [b](README.md:3)", file="README.md") == [
        BrokenLink("README.md", 1, "ADR-002.md:12-20", "missing")
    ]
    assert check("[a](docs/ADR%2D002.md)", file="README.md") == []


def test_links_inside_code_are_ignored() -> None:
    text = "`[a](/abs.md)`\n\n```md\n[b](missing.md)\n```\n\n~~~\n[c](/x)\n~~~\n[d](gone.md)"
    assert check(text) == [BrokenLink("docs/SPEC.md", 10, "gone.md", "missing")]


def test_link_text_with_nested_brackets_and_title() -> None:
    assert check('[see [1]](gone.md "Title")') == [
        BrokenLink("docs/SPEC.md", 1, "gone.md", "missing")
    ]


@pytest.mark.parametrize(
    ("path", "maintained"),
    [
        ("README.md", True),
        ("docs/SPEC.md", True),
        ("docs/_archive/README.md", True),
        ("docs/_archive/plans/README.md", True),
        ("docs/_archive/plans/old.md", False),
        ("docs/_archive/old.md", False),
        (".scratch/notes.md", False),
        ("pyproject.toml", False),
    ],
)
def test_maintained_markdown_selection(path: str, maintained: bool) -> None:
    assert is_maintained_markdown(path) is maintained


def test_repository_scan_checks_index_but_skips_archived_files(tmp_path: Path) -> None:
    (tmp_path / "docs/_archive").mkdir(parents=True)
    (tmp_path / "README.md").write_text("[ok](docs/_archive/README.md)\n", encoding="utf-8")
    (tmp_path / "docs/_archive/README.md").write_text("[gone](nope.md)\n", encoding="utf-8")
    (tmp_path / "docs/_archive/old.md").write_text("[gone](/abs.md)\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)

    assert find_broken_links(tmp_path) == [
        BrokenLink("docs/_archive/README.md", 1, "nope.md", "missing")
    ]
