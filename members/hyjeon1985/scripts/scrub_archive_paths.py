#!/usr/bin/env python3
"""
Scrub privacy-sensitive absolute paths inside archived text files.

Policy:
- Replace repository absolute root path with "<REPO_ROOT>".
- Do not touch binary artifacts (npz/pt/etc).

This is intended to run after copying outputs into members/<id>/archive.
"""

from __future__ import annotations

import argparse
from pathlib import Path


TEXT_EXTS = {
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".csv",
    ".md",
    ".txt",
    ".log",
}


def _scrub_text(text: str, repo_root_abs: str) -> str:
    # Keep relative structure but remove the machine-specific absolute prefix.
    if repo_root_abs:
        text = text.replace(repo_root_abs, "<REPO_ROOT>")
        if not repo_root_abs.endswith("/"):
            text = text.replace(repo_root_abs + "/", "<REPO_ROOT>/")
    return text


def _try_read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Archive root directory to scrub in-place.")
    ap.add_argument(
        "--repo-root",
        default="",
        help="Absolute repository root to scrub. If omitted, it is resolved from --root.",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"archive root not found or not a directory: {root}")

    repo_root_abs = args.repo_root.strip()
    if not repo_root_abs:
        # Best-effort: walk up until a ".git" directory is found.
        cur = root.resolve()
        while True:
            if (cur / ".git").exists():
                repo_root_abs = str(cur)
                break
            if cur.parent == cur:
                repo_root_abs = ""
                break
            cur = cur.parent
    if repo_root_abs:
        repo_root_abs = str(Path(repo_root_abs).resolve())

    scanned = 0
    modified = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in TEXT_EXTS:
            continue

        scanned += 1
        raw = _try_read_text(p)
        if raw is None:
            continue

        out = _scrub_text(raw, repo_root_abs=repo_root_abs)
        if out == raw:
            continue

        p.write_text(out, encoding="utf-8")
        modified += 1

    print(f"[scrub] root={root} scanned_files={scanned} modified_files={modified}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
