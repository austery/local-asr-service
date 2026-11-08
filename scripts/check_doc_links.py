"""Check that relative Markdown links in maintained documents resolve.

Rules (see ~/Documents/global-skills/agents/DOC-LIFECYCLE.md, "Link check"):

- Every relative link or image in a tracked Markdown file resolves to a tracked
  file, a directory that holds one, or the repository root. Checking Git's index
  instead of the disk gives the same result locally and on CI (untracked files,
  letter case).
- Absolute paths (``/x``, ``~/x``, ``C:/x``) and ``file:`` links fail. Write a
  machine-specific path in a code span instead.
- Files under ``docs/_archive/`` are byte-preserved history and are skipped;
  the archive README indexes are checked.
- ``.scratch/`` holds task artifacts, not documents, and is skipped.

Links are found by markdown-it-py's CommonMark parser (plus tables), so code
blocks, code spans and HTML blocks follow CommonMark exactly. Reference-style
links are checked; heading anchors, external URLs and raw HTML links are not.

Usage: ``uv run python scripts/check_doc_links.py``
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

from markdown_it import MarkdownIt
from markdown_it.token import Token

ARCHIVE_DIR = "docs/_archive/"
SCRATCH_DIR = ".scratch/"

Reason = Literal["missing", "outside-repository", "machine-specific"]


@dataclass(frozen=True)
class BrokenLink:
    file: str
    line: int
    target: str
    reason: Reason


_ARCHIVE_INDEX = re.compile(r"^docs/_archive/(?:[^/]+/)?README\.md$")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_WINDOWS_ABSOLUTE = re.compile(r"^[a-z]:[\\/]", re.IGNORECASE)
# A bare filename with an editor line suffix (README.md:12) resembles a scheme.
_BARE_FILE_LINE = re.compile(r"^[^:/]+\.[^:/]+:\d+(?:-\d+)?(?:[?#].*)?$")
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


class _RawLinkMarkdownIt(MarkdownIt):
    """Keep destinations as written.

    The default rejects file: links (which must fail here, not vanish) and
    percent-encodes the rest.
    """

    def validateLink(self, url: str) -> bool:  # noqa: N802 - parent method name
        return True

    def normalizeLink(self, url: str) -> str:  # noqa: N802 - parent method name
        return url


_MD = _RawLinkMarkdownIt("commonmark").enable("table")


def is_maintained_markdown(path: str) -> bool:
    if not path.endswith(".md") or path.startswith(SCRATCH_DIR):
        return False
    if not path.startswith(ARCHIVE_DIR):
        return True
    return _ARCHIVE_INDEX.match(path) is not None


def _destinations(tokens: Sequence[Token]) -> Iterator[tuple[int, int, str]]:
    """Yield (first line, last line, destination) for each link and image."""
    for token in tokens:
        if token.type != "inline" or token.map is None:
            continue
        first, last = token.map
        for child in token.children or []:
            if child.type == "image":
                # An image's children are its alt text; a link there is not rendered.
                target = child.attrGet("src")
            elif child.type == "link_open":
                target = child.attrGet("href")
            else:
                target = None
            if isinstance(target, str):
                yield first, last, target


def _line_of(lines: Sequence[str], first: int, last: int, target: str) -> int:
    """1-based line of the target inside its block; the block start if not found."""
    block = range(first, min(last, len(lines)))
    # Prefer a link destination over plain text that mentions the same path.
    for needle in (f"({target}", f"(<{target}", target):
        for index in block:
            if needle in lines[index]:
                return index + 1
    return first + 1


def _link_path(target: str) -> str:
    without_fragment = target.split("#", 1)[0].split("?", 1)[0]
    return unquote(_LINE_SUFFIX.sub("", without_fragment))


def _is_network_path(target: str) -> bool:
    """A scheme-relative URL such as //example.org/page."""
    return target.startswith("//")


def _is_machine_specific(target: str) -> bool:
    return (
        target.lower().startswith("file:")
        or (target.startswith("/") and not _is_network_path(target))
        or target == "~"
        or target.startswith(("~/", "~\\", "\\"))
        or _WINDOWS_ABSOLUTE.match(target) is not None
    )


def find_broken_links_in_text(
    file: str, text: str, exists: Callable[[str], bool]
) -> list[BrokenLink]:
    """Check one document. `exists` receives a repository-relative POSIX path."""
    lines = text.split("\n")
    broken: list[BrokenLink] = []
    for first, last, target in _destinations(_MD.parse(text)):
        line = _line_of(lines, first, last, target)
        if _is_machine_specific(target):
            broken.append(BrokenLink(file, line, target, "machine-specific"))
            continue
        if (
            target.startswith("#")
            or _is_network_path(target)
            or (_SCHEME.match(target) and not _BARE_FILE_LINE.match(target))
        ):
            continue
        path = _link_path(target)
        if not path:
            continue
        resolved = posixpath.normpath(posixpath.join(posixpath.dirname(file), path))
        if resolved == ".." or resolved.startswith("../"):
            broken.append(BrokenLink(file, line, target, "outside-repository"))
        elif not exists(resolved):
            broken.append(BrokenLink(file, line, target, "missing"))
    return broken


def _tracked_files(root: Path) -> list[str]:
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    return [file for file in listed.split("\0") if file]


def _with_directories(files: Sequence[str]) -> set[str]:
    """The files, every directory that holds one, and the root itself."""
    paths: set[str] = {".", *files}
    for file in files:
        parent = posixpath.dirname(file)
        while parent and parent not in paths:
            paths.add(parent)
            parent = posixpath.dirname(parent)
    return paths


def find_broken_links(root: Path) -> list[BrokenLink]:
    tracked = _tracked_files(root)
    paths = _with_directories(tracked)
    files = sorted(f for f in tracked if is_maintained_markdown(f))
    if not files:
        raise RuntimeError("No maintained Markdown files discovered.")
    broken: list[BrokenLink] = []
    for file in files:
        text = (root / file).read_text(encoding="utf-8")
        broken.extend(find_broken_links_in_text(file, text, paths.__contains__))
    return broken


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    broken = find_broken_links(root)
    for link in broken:
        print(f"{link.file}:{link.line}: {link.reason} link target {link.target}", file=sys.stderr)
    if broken:
        print(f"{len(broken)} broken Markdown link(s).", file=sys.stderr)
        return 1
    print("All relative Markdown links in maintained files resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
