"""Check that relative Markdown links in maintained documents resolve.

Rules (see ~/Documents/global-skills/agents/DOC-LIFECYCLE.md, "Link check"):

- Every relative inline link or image in a tracked Markdown file resolves to a
  tracked file, or to a directory that holds one. Checking Git's index instead of
  the disk gives the same result locally and on CI (untracked files, letter case).
- Absolute paths (``/x``, ``~/x``, ``C:/x``) and ``file:`` links fail. Write a
  machine-specific path in a code span instead.
- Files under ``docs/_archive/`` are byte-preserved history and are skipped;
  the archive README indexes are checked.
- ``.scratch/`` holds task artifacts, not documents, and is skipped.

Code blocks (fenced, and indented outside lists), code spans and HTML comments
are ignored. Out of scope: heading anchors, external URLs, reference-style link
definitions, raw HTML links, and link text nested more than one bracket deep.

Usage: ``uv run python scripts/check_doc_links.py``
"""

from __future__ import annotations

import posixpath
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import unquote

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
_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})")
_INDENTED = re.compile(r"^(?: {4}|\t)")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# A code span: an unescaped backtick run, closed by a run of the same length,
# never crossing a blank line (CommonMark: a span stays inside one paragraph).
_CODE_SPAN = re.compile(r"(?<![\\`])(`+)(?!`)(?:[^`\n]|`+(?!\1)|\n(?![ \t]*\n))+?(?<!`)\1(?!`)")
# Link text may hold one level of nested brackets, e.g. [see [1]](path).
_LINK_START = re.compile(r"!?\[((?:[^\[\]\\]|\\.|\[(?:[^\[\]\\]|\\.)*\])*)\]\(")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
_WINDOWS_ABSOLUTE = re.compile(r"^[a-z]:[\\/]", re.IGNORECASE)
# A bare filename with an editor line suffix (README.md:12) resembles a scheme.
_BARE_FILE_LINE = re.compile(r"^[^:/]+\.[^:/]+:\d+(?:-\d+)?(?:[?#].*)?$")
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")
_ESCAPED_PUNCTUATION = re.compile(r"\\([!-/:-@\[-`{-~])")


def is_maintained_markdown(path: str) -> bool:
    if not path.endswith(".md") or path.startswith(SCRATCH_DIR):
        return False
    if not path.startswith(ARCHIVE_DIR):
        return True
    return _ARCHIVE_INDEX.match(path) is not None


def _blank(text: str) -> str:
    """Replace text with spaces, keeping newlines so offsets and lines survive."""
    return re.sub(r"[^\n]", " ", text)


def _mask_blocks(text: str) -> str:
    """Blank fenced code blocks, and indented code blocks that are not list content."""
    out: list[str] = []
    fence: str | None = None
    in_indented = False
    in_list = False
    previous_blank = True
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if fence is not None:
            if stripped.startswith(fence) and set(stripped) == {fence[0]}:
                fence = None
            out.append(_blank(line))
            continue
        match = _FENCE_OPEN.match(line)
        if match:
            fence = match.group(1)
            out.append(_blank(line))
            previous_blank = False
            continue
        is_blank = not stripped
        if is_blank:
            out.append(line)
            previous_blank = True
            continue
        indented = _INDENTED.match(line) is not None
        if indented and not in_list and (previous_blank or in_indented):
            in_indented = True
            out.append(_blank(line))
        else:
            in_indented = False
            if not indented:
                in_list = _LIST_ITEM.match(line) is not None or (in_list and not previous_blank)
            out.append(line)
        previous_blank = False
    return "".join(out)


def _mask_code(text: str) -> str:
    masked = _mask_blocks(text)
    masked = _HTML_COMMENT.sub(lambda m: _blank(m.group(0)), masked)
    return _CODE_SPAN.sub(lambda m: _blank(m.group(0)), masked)


def _read_destination(text: str, start: int) -> str | None:
    """Read a link destination starting just after '('; None if malformed."""
    rest = text[start:]
    rest = rest.lstrip(" \t")
    if rest.startswith("\n"):  # CommonMark allows one line ending before it.
        rest = rest[1:].lstrip(" \t")
    if rest.startswith("<"):
        end = rest.find(">")
        if end < 0 or "\n" in rest[:end]:
            return None
        return rest[1:end]
    depth = 0
    for index, char in enumerate(rest):
        if char == "\\":
            continue
        if char in " \t\n":
            return rest[:index] or None
        if char == "(" and (index == 0 or rest[index - 1] != "\\"):
            depth += 1
        elif char == ")" and (index == 0 or rest[index - 1] != "\\"):
            if depth == 0:
                return rest[:index] or None
            depth -= 1
    return None


def _scan(masked: str, offset: int, end: int) -> Iterator[tuple[int, str]]:
    for match in _LINK_START.finditer(masked, offset, end):
        # A link's text can hold an image, as in a badge: [![alt](img.png)](page.md).
        yield from _scan(masked, match.start(1), match.end(1))
        target = _read_destination(masked, match.end())
        if target is not None:
            yield masked.count("\n", 0, match.start()) + 1, target


def _inline_links(text: str) -> Iterator[tuple[int, str]]:
    masked = _mask_code(text)
    return _scan(masked, 0, len(masked))


def _link_path(target: str) -> str:
    without_fragment = target.split("#", 1)[0].split("?", 1)[0]
    unescaped = _ESCAPED_PUNCTUATION.sub(r"\1", without_fragment)
    return unquote(_LINE_SUFFIX.sub("", unescaped))


def _is_machine_specific(target: str) -> bool:
    return (
        target.lower().startswith("file:")
        or target.startswith(("/", "~"))
        or _WINDOWS_ABSOLUTE.match(target) is not None
    )


def find_broken_links_in_text(
    file: str, text: str, exists: Callable[[str], bool]
) -> list[BrokenLink]:
    """Check one document. `exists` receives a repository-relative POSIX path."""
    broken: list[BrokenLink] = []
    for line, target in sorted(_inline_links(text)):
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
    """Tracked files plus every directory that holds one."""
    listed = subprocess.run(
        ["git", "ls-files", "-z"], cwd=root, check=True, capture_output=True, text=True
    ).stdout
    paths: set[str] = set()
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
