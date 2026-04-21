"""Layer 2: File-matching heuristic.

Checks if changed source files have corresponding test files.
Classifies each file as PASS (test modified), WARNING (test exists, not modified),
FAIL (no test found), or SKIP (excluded).
"""

from __future__ import annotations

import fnmatch
from pathlib import PurePosixPath

from src.models import FileVerdict, LayerResult, Verdict


def _is_excluded(filepath: str, exclude_patterns: list[str]) -> bool:
    """Check if a file matches any exclusion pattern."""
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filepath, pattern):
            return True
        # Also check just the filename for extension patterns
        if fnmatch.fnmatch(PurePosixPath(filepath).name, pattern):
            return True
    return False


def _is_test_file(filepath: str, patterns: dict[str, dict[str, str]]) -> bool:
    """Check if a file is itself a test file."""
    name = PurePosixPath(filepath).stem
    for _lang, mapping in patterns.items():
        template = mapping["test_template"]
        if "test_{name}" in template and name.startswith("test_"):
            return True
        if "{name}Test" in template and name.endswith("Test"):
            return True
        if "{name}Tests" in template and name.endswith("Tests"):
            return True
        if "{name}.test" in template and ".test." in filepath:
            return True
        if "{name}.spec" in template and ".spec." in filepath:
            return True
        if "{name}_test" in template and name.endswith("_test"):
            return True
        if "{name}_spec" in template and name.endswith("_spec"):
            return True
        if "{name}Spec" in template and name.endswith("Spec"):
            return True
        if "__tests__/" in template and "__tests__/" in filepath:
            return True
    return False


def _match_test_file(
    source_file: str,
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
) -> str | None:
    """Find a matching test file for a source file.

    Returns the test file path if found, None otherwise.
    """
    if _is_test_file(source_file, patterns):
        return None  # Don't match test files against themselves

    source_path = PurePosixPath(source_file)
    source_name = source_path.stem

    for _lang, mapping in patterns.items():
        src_pattern = mapping["src_pattern"]
        test_template = mapping["test_template"]

        # Check if this source file matches the language pattern
        if not fnmatch.fnmatch(source_file, src_pattern):
            continue

        # Build possible test file names from template
        test_name = test_template.replace("{name}", source_name)

        # Search repo files for a match
        for repo_file in all_repo_files:
            if fnmatch.fnmatch(repo_file, test_name):
                return repo_file

    return None


def run_layer2(
    changed_files: list[str],
    all_repo_files: list[str],
    patterns: dict[str, dict[str, str]],
    exclude_patterns: list[str],
) -> LayerResult:
    """Execute Layer 2 analysis.

    Args:
        changed_files: Files changed in the PR.
        all_repo_files: All files in the repo (for test lookup).
        patterns: Language→pattern mappings for source-to-test matching.
        exclude_patterns: Glob patterns to exclude from analysis.

    Returns:
        LayerResult with per-file verdicts.
    """
    file_verdicts: list[FileVerdict] = []
    changed_set = set(changed_files)

    for filepath in changed_files:
        # Skip excluded files
        if _is_excluded(filepath, exclude_patterns):
            continue

        # Skip test files themselves
        if _is_test_file(filepath, patterns):
            continue

        test_file = _match_test_file(filepath, all_repo_files, patterns)

        if test_file is None:
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.FAIL,
                    reason="No matching test file found",
                    layer="layer2",
                )
            )
        elif test_file in changed_set:
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.PASS,
                    reason=f"Test file modified in PR: {test_file}",
                    layer="layer2",
                )
            )
        else:
            # Test exists but wasn't modified — ambiguous
            file_verdicts.append(
                FileVerdict(
                    file=filepath,
                    verdict=Verdict.WARNING,
                    reason=f"Test file exists ({test_file}) but was not modified in this PR",
                    layer="layer2",
                )
            )

    # Determine overall verdict
    verdicts = [fv.verdict for fv in file_verdicts]
    if not verdicts:
        overall = Verdict.PASS
    elif Verdict.FAIL in verdicts:
        overall = Verdict.FAIL
    elif Verdict.WARNING in verdicts:
        overall = Verdict.WARNING
    else:
        overall = Verdict.PASS

    details_parts: list[str] = []
    for v in [Verdict.PASS, Verdict.WARNING, Verdict.FAIL]:
        count = verdicts.count(v)
        if count:
            details_parts.append(f"{count} {v.value}")
    details = (
        f"File matching: {', '.join(details_parts)}"
        if details_parts
        else "No source files to check"
    )

    return LayerResult(
        layer="layer2",
        verdict=overall,
        details=details,
        file_verdicts=file_verdicts,
        short_circuit=(overall == Verdict.PASS),
    )
