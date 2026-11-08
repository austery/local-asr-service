"""Check that relative Markdown links in maintained documents resolve.

Rules (see ~/Documents/global-skills/agents/DOC-LIFECYCLE.md, "Link check"):

- Every relative inline link or image in a tracked Markdown file resolves to an
  existing path inside the repository.
- Absolute paths and ``file:`` links fail. Write a machine-specific path in a
  code span instead.
- Files under ``docs/_archive/`` are byte-preserved history and are skipped;
  the archive README indexes are checked.
- ``.scratch/`` holds task artifacts, not documents, and is skipped.

Out of scope: heading anchors, external URLs, reference-style link definitions
and raw HTML links. Code blocks and code spans are ignored.

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
_CODE_SPAN = re.compile(r"(`+)(?:(?!\1).)+?\1", re.DOTALL)
# Link text may hold one level of nested brackets, e.g. [see [1]](path).
_LINK_START = re.compile(r"!?\[(?:[^\[\]\\]|\\.|\[[^\[\]]*\])*\]\(")
_SCHEME = re.compile(r"^[a-z][a-z0-9+.-]*:", re.IGNORECASE)
# A bare filename with an editor line suffix (README.md:12) resembles a scheme.
_BARE_FILE_LINE = re.compile(r"^[^:/]+\.[^:/]+:\d+(?:-\d+)?(?:[?#].*)?$")
_LINE_SUFFIX = re.compile(r":\d+(?:-\d+)?$")


def is_maintained_markdown(path: str) -> bool:
    if not path.endswith(".md") or path.startswith(SCRATCH_DIR):
        return False
    if not path.startswith(ARCHIVE_DIR):
        return True
    return _ARCHIVE_INDEX.match(path) is not None


def _blank(text: str) -> str:
    """Replace text with spaces, keeping newlines so offsets and lines survive."""
    return re.sub(r"[^\n]", " ", text)


def _mask_code(text: str) -> str:
    """Blank fenced code blocks and code spans so links inside them are ignored."""
    out: list[str] = []
    fence: str | None = None
    for line in text.splitlines(keepends=True):
        match = _FENCE_OPEN.match(line)
        if fence is None and match:
            fence = match.group(1)
            out.append(_blank(line))
        elif fence is not None:
            stripped = line.strip()
            if stripped.startswith(fence[0] * len(fence)) and set(stripped) <= {fence[0]}:
                fence = None
            out.append(_blank(line))
        else:
            out.append(line)
    return _CODE_SPAN.sub(lambda m: _blank(m.group(0)), "".join(out))


def _read_destination(text: str, start: int) -> str | None:
    """Read a link destination starting just after '('; None if malformed."""
    rest = text[start:].lstrip(" \t")
    if rest.startswith("<"):
        end = rest.find(">")
        return None if end < 0 else rest[1:end]
    depth = 0
    for index, char in enumerate(rest):
        if char in " \t\n":
            return rest[:index]
        if char == "(":
            depth += 1
        elif char == ")":
            if depth == 0:
                return rest[:index]
            depth -= 1
    return None


def _inline_links(text: str) -> Iterator[tuple[int, str]]:
    masked = _mask_code(text)
    for match in _LINK_START.finditer(masked):
        target = _read_destination(masked, match.end())
        if target is not None:
            yield masked.count("\n", 0, match.start()) + 1, target


def _link_path(target: str) -> str:
    without_fragment = target.split("#", 1)[0].split("?", 1)[0]
    return unquote(_LINE_SUFFIX.sub("", without_fragment))


def find_broken_links_in_text(
    file: str, text: str, exists: Callable[[str], bool]
) -> list[BrokenLink]:
    """Check one document. `exists` receives a repository-relative POSIX path."""
    broken: list[BrokenLink] = []
    for line, target in _inline_links(text):
        if target.lower().startswith("file:") or target.startswith("/"):
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


def find_broken_links(root: Path) -> list[BrokenLink]:
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    files = [f for f in listed.split("\0") if f and is_maintained_markdown(f)]
    if not files:
        raise RuntimeError("No maintained Markdown files discovered.")
    broken: list[BrokenLink] = []
    for file in files:
        text = (root / file).read_text(encoding="utf-8")
        broken.extend(find_broken_links_in_text(file, text, lambda p: (root / p).exists()))
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
