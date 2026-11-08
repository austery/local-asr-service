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


def test_indented_code_block_is_ignored_but_list_content_is_checked() -> None:
    text = "Para.\n\n    [code](gone.md)\n\n- item\n\n    [nested](missing.md)\n"
    assert check(text) == [BrokenLink("docs/SPEC.md", 7, "missing.md", "missing")]


def test_destination_after_line_break_is_checked() -> None:
    assert [b.reason for b in check("[a](\nmissing.md) [b](\n/Users/x.md)")] == [
        "missing",
        "machine-specific",
    ]


def test_stray_or_escaped_backtick_does_not_hide_later_links() -> None:
    for opener in ("Stray ` tick.", "Escaped \\` tick."):
        text = f"{opener}\n\n[b](gone.md)\n\nUse `code` here."
        assert check(text) == [BrokenLink("docs/SPEC.md", 3, "gone.md", "missing")]


def test_image_inside_link_is_checked() -> None:
    assert check("[![badge](gone.png)](ADR-002.md)") == [
        BrokenLink("docs/SPEC.md", 1, "gone.png", "missing")
    ]


def test_home_and_windows_absolute_paths_fail() -> None:
    reasons = [b.reason for b in check("[a](~/x.md) [b](C:/x.md) [c](C:\\x.md)")]
    assert reasons == ["machine-specific"] * 3


def test_commonmark_non_links_are_ignored() -> None:
    text = "<!-- [a](gone.md) -->\n[b](ADR\\-002.md)\n[c](<gone.md\n\nlater >"
    assert check(text) == []


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


def test_repository_scan_uses_tracked_files_and_skips_archived_payloads(tmp_path: Path) -> None:
    (tmp_path / "docs/_archive").mkdir(parents=True)
    (tmp_path / "README.md").write_text("[ok](docs/_archive/README.md)\n", encoding="utf-8")
    (tmp_path / "docs/_archive/README.md").write_text("[gone](nope.md)\n", encoding="utf-8")
    (tmp_path / "docs/_archive/old.md").write_text("[gone](/abs.md)\n", encoding="utf-8")
    (tmp_path / "docs/guide.md").write_text("[dir](_archive/) [new](draft.md)\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    (tmp_path / "docs/draft.md").write_text("untracked\n", encoding="utf-8")

    assert find_broken_links(tmp_path) == [
        BrokenLink("docs/_archive/README.md", 1, "nope.md", "missing"),
        BrokenLink("docs/guide.md", 1, "draft.md", "missing"),
    ]
