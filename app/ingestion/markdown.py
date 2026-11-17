"""Deterministic Markdown extraction without changing evidence formatting."""

import hashlib
import json
import os
import re
import stat
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import yaml
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNKING_VERSION = "markdown-v1:600:80"


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def json_value(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("Frontmatter keys must be strings")
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_value(item) for item in value]
    return value


@dataclass
class ExtractedDocument:
    document_id: str
    source: str
    source_digest: str
    chunks: list[dict]


def discover(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("Document root must be a directory, not a symlink")

    def fail(error):
        raise error

    paths = []
    # os.walk's default silently ignores unreadable directories, unsafe for pruning.
    for directory, dirs, files in os.walk(root, followlinks=False, onerror=fail):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            path = Path(directory) / name
            if path.suffix.lower() == ".md" and stat.S_ISREG(path.lstat().st_mode):
                paths.append(path)
    return sorted(paths)


def sections(body: str):
    lines = body.splitlines(keepends=True)
    headings: list[tuple[int, str]] = []
    content: list[str] = []
    fence = None
    i = 0
    while i < len(lines):
        line = lines[i]
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            content.append(line)
            if re.match(rf"^ {{0,3}}{re.escape(fence[0])}{{{fence[1]},}}\s*$", line):
                fence = None
            i += 1
            continue
        if marker and not (marker[1][0] == "`" and "`" in marker[2]):
            fence = (marker[1][0], len(marker[1]))
            content.append(line)
            i += 1
            continue
        atx = re.match(r"^ {0,3}(#{1,6})(?:[ \t]+(.*)|\s*)$", line.rstrip("\r\n"))
        underline = (
            re.match(r"^ {0,3}(=+|-+)[ \t]*$", lines[i + 1].rstrip("\r\n"))
            if i + 1 < len(lines)
            and line.strip()
            and not line.startswith(("    ", "\t"))
            else None
        )
        if atx or underline:
            if content:
                yield [name for _, name in headings], "".join(content)
            level = len(atx[1]) if atx else (1 if underline[1][0] == "=" else 2)
            name = re.sub(r"[ \t]+#+[ \t]*$", "", atx[2] or "") if atx else line.strip()
            headings = [(depth, title) for depth, title in headings if depth < level]
            headings.append((level, name))
            content = [line] if atx else [line, lines[i + 1]]
            i += 1 if atx else 2
        else:
            content.append(line)
            i += 1
    if content:
        yield [name for _, name in headings], "".join(content)


def extract(path: Path, root: Path) -> ExtractedDocument:
    relative = path.relative_to(root)
    if any(
        (root / Path(*relative.parts[:i])).is_symlink()
        for i in range(1, len(relative.parts) + 1)
    ):
        raise ValueError("Symlink documents are unsupported")
    raw = path.read_bytes().decode("utf-8-sig")
    body = raw
    frontmatter = {}
    lines = raw.splitlines(keepends=True)
    if lines and lines[0].strip() == "---":
        end = next(
            (i for i in range(1, len(lines)) if lines[i].strip() in {"---", "..."}),
            None,
        )
        if end is None:
            raise ValueError("Unterminated frontmatter")
        frontmatter = yaml.safe_load("".join(lines[1:end]))
        if frontmatter is None:
            frontmatter = {}
        if not isinstance(frontmatter, dict):
            raise ValueError("Frontmatter must be a mapping")
        frontmatter = json_value(frontmatter)
        json.dumps(frontmatter, allow_nan=False)
        body = "".join(lines[end + 1 :])
    source = relative.as_posix()
    document_id = digest(source)
    chunks = []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=80, strip_whitespace=False
    )
    for section_index, (headings, text) in enumerate(sections(body)):
        for section_chunk, passage in enumerate(splitter.split_text(text)):
            if not passage.strip():
                continue
            chunk_id = digest(
                json.dumps(
                    [
                        document_id,
                        CHUNKING_VERSION,
                        section_index,
                        section_chunk,
                        passage,
                    ]
                )
            )
            metadata = {
                "source": source,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "headings": headings,
                "chunk_index": len(chunks),
                "chunking_version": CHUNKING_VERSION,
                "frontmatter": frontmatter,
            }
            chunks.append({"id": chunk_id, "text": passage, "metadata": metadata})
    return ExtractedDocument(document_id, source, digest(raw), chunks)
