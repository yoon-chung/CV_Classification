"""T7 forbidden scan - static analysis for v2 violations"""

from __future__ import annotations

import fnmatch
import re
import sys
from pathlib import Path
from typing import Iterator, Tuple

SCAN_ROOTS = [
    "configs",
    "src",
    "scripts",
    "docs",
]

EXCLUDE_PATTERNS = [
    ".sisyphus/**",
    "src/experiment/ops/qa/**",
    "**/__pycache__/**",
    "**/*.pyc",
]

# Forbidden patterns by file type
FORBIDDEN_PATTERNS = {
    "yaml": [
        (r"^\s*stage\s*:", "YAML key 'stage:' (use pipeline.step/stop_after)"),
        (r"upload\.dry_run", "upload.dry_run (deprecated, use S3_DRY_RUN env)"),
        (r"experiment\.scenario\s*:\s*s3\b", "scenario: s3 (use cloud)"),
    ],
    "script": [
        (r"\bstage=", "CLI override stage= (use pipeline.stop_after)"),
        (r"\bupload\.dry_run\b", "upload.dry_run override"),
        (r"scenario=s3\b", "scenario=s3 (use scenario=cloud)"),
    ],
    "all": [
        (r"\bdgx\b", "'dgx' string (use hpc)"),
        (
            r"runner\.profile\s*=\s*(explore|tune|solve)",
            "profile=explore/tune/solve in wrong place",
        ),
    ],
}


def should_exclude(path: Path, member_root: Path) -> bool:
    rel_path = path.relative_to(member_root)
    path_str = rel_path.as_posix()

    for pattern in EXCLUDE_PATTERNS:
        if fnmatch.fnmatch(path_str, pattern):
            return True

    return False


def scan_file(path: Path, member_root: Path) -> Iterator[Tuple[Path, int, str, str]]:
    if should_exclude(path, member_root):
        return

    try:
        content = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return

    lines = content.split("\n")
    ext = path.suffix.lower()

    patterns: list[tuple[str, str]] = []
    if ext in [".yaml", ".yml"]:
        patterns.extend(FORBIDDEN_PATTERNS["yaml"])
    elif ext in [".sh", ".md", ".txt"]:
        patterns.extend(FORBIDDEN_PATTERNS["script"])

    patterns.extend(FORBIDDEN_PATTERNS["all"])

    for line_no, line in enumerate(lines, 1):
        for pattern, message in patterns:
            if re.search(pattern, line, re.IGNORECASE):
                yield (path.relative_to(member_root), line_no, line.strip(), message)


def run_forbidden_scan(member_root: Path | None = None) -> int:
    if member_root is None:
        member_root = Path(__file__).resolve().parents[4]

    violations: list[tuple[Path, int, str, str]] = []

    for root_name in SCAN_ROOTS:
        root_path = member_root / root_name
        if not root_path.exists():
            continue

        for path in root_path.rglob("*"):
            if not path.is_file():
                continue
            for violation in scan_file(path, member_root):
                violations.append(violation)

    if violations:
        print("FORBIDDEN TOKENS FOUND:", file=sys.stderr)
        for path, line_no, line, message in violations:
            print(f"  {path}:{line_no}: {message}", file=sys.stderr)
            print(f"    {line[:80]}", file=sys.stderr)
        return 1

    print("Forbidden scan: PASSED (no violations)")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_forbidden_scan())
