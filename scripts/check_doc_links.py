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
links are checked; heading anchors and external URLs are not.

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


def _parser() -> MarkdownIt:
    md = MarkdownIt("commonmark").enable("table")
    # Keep destinations as written: the default rejects file: links (which must
    # fail here, not vanish) and percent-encodes the rest.
    md.validateLink = lambda _url: True
    md.normalizeLink = lambda url: url
    return md


_MD = _parser()


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
        stack = list(token.children or [])
        while stack:
            child = stack.pop(0)
            attr = "href" if child.type == "link_open" else "src" if child.type == "image" else None
            if attr is not None:
                target = child.attrGet(attr)
                if isinstance(target, str):
                    yield first, last, target
            stack[:0] = child.children or []


def _line_of(lines: Sequence[str], first: int, last: int, target: str) -> int:
    """1-based line of the target inside its block; the block start if not found."""
    for index in range(first, min(last, len(lines))):
        if target in lines[index]:
            return index + 1
    return first + 1


def _link_path(target: str) -> str:
    without_fragment = target.split("#", 1)[0].split("?", 1)[0]
    return unquote(_LINE_SUFFIX.sub("", without_fragment))


def _is_machine_specific(target: str) -> bool:
    return (
        target.lower().startswith("file:")
        or target.startswith(("/", "~", "\\"))
        or _WINDOWS_ABSOLUTE.match(target) is not None
    )


def find_broken_links_in_text(
    file: str, text: str, exists: Callable[[str], bool]
) -> list[BrokenLink]:
    """Check one document. `exists` receives a repository-relative POSIX path."""
    lines = text.splitlines()
    broken: list[BrokenLink] = []
    for first, last, target in _destinations(_MD.parse(text)):
        line = _line_of(lines, first, last, target)
        if _is_machine_specific(target):
            broken.append(BrokenLink(file, line, target, "machine-specific"))
            continue
        if target.startswith("#") or (
            _SCHEME.match(target) and not _BARE_FILE_LINE.match(target)
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


def _tracked_paths(root: Path) -> set[str]:
    """Tracked files, every directory that holds one, and the root itself."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    paths: set[str] = {"."}
    for file in filter(None, listed.split("\0")):
        paths.add(file)
        parent = posixpath.dirname(file)
        while parent and parent not in paths:
            paths.add(parent)
            parent = posixpath.dirname(parent)
    return paths


def find_broken_links(root: Path) -> list[BrokenLink]:
    tracked = _tracked_paths(root)
    files = sorted(f for f in tracked if is_maintained_markdown(f))
    if not files:
        raise RuntimeError("No maintained Markdown files discovered.")
    broken: list[BrokenLink] = []
    for file in files:
        text = (root / file).read_text(encoding="utf-8")
        broken.extend(find_broken_links_in_text(file, text, tracked.__contains__))
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
