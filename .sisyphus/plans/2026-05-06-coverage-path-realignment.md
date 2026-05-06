# Coverage Path Realignment — Factory + Auto-Detection

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix Layer 1 producing false FAILs ("not in coverage report") when coverage XML was generated inside a Docker container with different filesystem paths than the GitHub Actions runner.

**Architecture:** A `CoveragePathNormalizer` factory detects whether each coverage XML is Cobertura or Clover format, then normalizes paths before passing to diff-cover. For Clover, it auto-detects the container path prefix by comparing `<file path="...">` values against the PR's git-relative paths, then rewrites the XML with stripped prefixes. For Cobertura, it verifies/fixes the `<source>` element so diff-cover's built-in prefix stripping works correctly.

**Tech Stack:** Python 3.12, xml.etree.ElementTree, tempfile, pathlib

---

## Problem Statement

When tests run in Docker (e.g., Matecat's PHPUnit runs at `/var/www/app/`), coverage XMLs record container-internal absolute paths:

- **Clover:** `<file path="/var/www/app/lib/Controller/AuthCookie.php">`
- **Cobertura:** `<source>/var/www/app</source>` + `<class filename="lib/Controller/AuthCookie.php">`

diff-cover's Clover path matching computes `os.path.relpath("/var/www/app/lib/...", "/home/runner/work/Repo/Repo/")` → `../../../../var/www/app/lib/...` which never matches the git-relative path `lib/Controller/AuthCookie.php`. Result: `src_stats = {}`, all files marked as "not in coverage report".

For Cobertura, diff-cover stores the bare `filename` attribute in its lookup cache AND a `source + filename` entry. The git-relative path lookup (`src_rel_path`) SHOULD match the bare filename — but only if `filename` is relative. If the coverage tool writes absolute filenames (some do), Cobertura breaks the same way.

## Design Decision

**Zero-config auto-detection** — no new `action.yml` inputs required. The normalizer inspects the XML, compares recorded paths against the list of `diff_files` (already available), and auto-determines the prefix to strip.

Fallback: if auto-detection fails (no `diff_files` match after prefix stripping), pass the XML unchanged to diff-cover (current behavior — no regression).

---

## File Structure

| File | Responsibility |
|------|---------------|
| `src/coverage_normalizer.py` (CREATE) | Format detection, prefix auto-detection, XML rewriting |
| `src/layer1_coverage.py` (MODIFY) | Call normalizer before invoking diff-cover |
| `tests/test_coverage_normalizer.py` (CREATE) | Unit tests for the normalizer |
| `tests/test_layer1_coverage.py` (MODIFY) | Integration test verifying the full flow with Docker paths |

---

### Task 1: Format Detection Factory

**Files:**
- Create: `src/coverage_normalizer.py`
- Test: `tests/test_coverage_normalizer.py`

The factory must detect: Clover, Cobertura, JaCoCo, or Unknown (LCOV/non-XML).

Detection rules (same as diff-cover's internal logic):
- Has `<coverage clover="...">` or root tag is `<coverage>` with `<project>` child → Clover
- Has `<coverage>` root with `<packages>` child → Cobertura
- Has `<!DOCTYPE report ...>` or `<report>` root with `<package>` children containing `<sourcefile>` → JaCoCo
- Non-XML file → LCOV (no normalization possible)

- [ ] **Step 1: Write failing tests for format detection**

```python
# tests/test_coverage_normalizer.py

import tempfile
from pathlib import Path

from src.coverage_normalizer import CoverageFormat, detect_format


class TestDetectFormat:
    def test_detects_clover_format(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        assert detect_format(str(xml)) == CoverageFormat.CLOVER

    def test_detects_cobertura_format(self, tmp_path: Path):
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" timestamp="1700000000" lines-valid="100" lines-covered="80" line-rate="0.8">
  <sources><source>/var/www/app</source></sources>
  <packages>
    <package name="lib" line-rate="0.8">
      <classes>
        <class name="AuthCookie" filename="lib/AuthCookie.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        assert detect_format(str(xml)) == CoverageFormat.COBERTURA

    def test_detects_jacoco_format(self, tmp_path: Path):
        xml = tmp_path / "jacoco.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE report PUBLIC "-//JACOCO//DTD Report 1.1//EN" "report.dtd">
<report name="test">
  <package name="com/example">
    <sourcefile name="App.java">
      <line nr="1" mi="0" ci="1"/>
    </sourcefile>
  </package>
</report>""")
        assert detect_format(str(xml)) == CoverageFormat.JACOCO

    def test_non_xml_returns_unknown(self, tmp_path: Path):
        lcov = tmp_path / "coverage.info"
        lcov.write_text("SF:/var/www/app/lib/AuthCookie.php\nDA:1,1\nend_of_record\n")
        assert detect_format(str(lcov)) == CoverageFormat.UNKNOWN

    def test_malformed_xml_returns_unknown(self, tmp_path: Path):
        xml = tmp_path / "broken.xml"
        xml.write_text("not xml at all {{{")
        assert detect_format(str(xml)) == CoverageFormat.UNKNOWN
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.coverage_normalizer'`

- [ ] **Step 3: Implement format detection**

```python
# src/coverage_normalizer.py
"""Coverage XML path normalizer.

Detects coverage format (Clover/Cobertura/JaCoCo) and normalizes Docker-internal
paths so diff-cover can match them against git-relative file paths.
"""

from __future__ import annotations

import os
import tempfile
from enum import Enum
from pathlib import Path
from xml.etree import ElementTree as ET


class CoverageFormat(Enum):
    CLOVER = "clover"
    COBERTURA = "cobertura"
    JACOCO = "jacoco"
    UNKNOWN = "unknown"


def detect_format(filepath: str) -> CoverageFormat:
    """Detect the coverage XML format by inspecting root structure.

    Returns CoverageFormat.UNKNOWN for non-XML files or unrecognized formats.
    """
    if not filepath.endswith(".xml"):
        return CoverageFormat.UNKNOWN

    try:
        tree = ET.parse(filepath)  # noqa: S314
    except ET.ParseError:
        return CoverageFormat.UNKNOWN

    root = tree.getroot()

    # Clover: <coverage> root with <project> child
    if root.tag == "coverage" and root.find("project") is not None:
        return CoverageFormat.CLOVER

    # JaCoCo: <report> root with <package>/<sourcefile> structure
    if root.tag == "report":
        pkg = root.find(".//package")
        if pkg is not None and pkg.find("sourcefile") is not None:
            return CoverageFormat.JACOCO

    # Cobertura: <coverage> root with <packages> child
    if root.tag == "coverage" and root.find("packages") is not None:
        return CoverageFormat.COBERTURA

    return CoverageFormat.UNKNOWN
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestDetectFormat -v`
Expected: All 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/coverage_normalizer.py tests/test_coverage_normalizer.py
git commit -m "feat(layer1): add coverage format detection factory"
```

---

### Task 2: Clover Path Prefix Auto-Detection + Rewriting

**Files:**
- Modify: `src/coverage_normalizer.py`
- Test: `tests/test_coverage_normalizer.py`

The Clover normalizer must:
1. Extract all `<file name="...">` or `<file path="...">` paths from the XML
2. Compare against `diff_files` (git-relative paths from the PR)
3. Find the longest common prefix that, when stripped, makes XML paths match git paths
4. Rewrite the XML to a temp file with stripped paths

- [ ] **Step 1: Write failing tests for Clover prefix detection**

```python
# tests/test_coverage_normalizer.py (append to file)

from src.coverage_normalizer import detect_path_prefix


class TestDetectPathPrefix:
    def test_finds_docker_prefix_from_diff_files(self):
        xml_paths = [
            "/var/www/app/lib/Controller/AuthCookie.php",
            "/var/www/app/lib/Model/UserDao.php",
            "/var/www/app/src/Service/AuthService.php",
        ]
        diff_files = [
            "lib/Controller/AuthCookie.php",
            "lib/Model/UserDao.php",
            "src/other.py",  # not in XML — shouldn't affect detection
        ]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == "/var/www/app/"

    def test_finds_prefix_with_trailing_slash_normalization(self):
        xml_paths = ["/app/src/auth.py", "/app/src/billing.py"]
        diff_files = ["src/auth.py", "src/billing.py"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == "/app/"

    def test_returns_empty_when_no_match(self):
        xml_paths = ["/completely/different/path.py"]
        diff_files = ["src/auth.py"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == ""

    def test_returns_empty_when_paths_already_relative(self):
        xml_paths = ["lib/Controller/AuthCookie.php", "lib/Model/UserDao.php"]
        diff_files = ["lib/Controller/AuthCookie.php"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == ""

    def test_requires_minimum_match_ratio(self):
        """Prefix detection requires at least 1 match to confirm."""
        xml_paths = [
            "/var/www/app/lib/AuthCookie.php",
            "/other/path/something.php",
        ]
        diff_files = ["lib/AuthCookie.php"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == "/var/www/app/"

    def test_handles_windows_style_paths(self):
        xml_paths = ["C:\\Users\\app\\src\\auth.py"]
        diff_files = ["src/auth.py"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == "C:\\Users\\app\\"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestDetectPathPrefix -v`
Expected: FAIL with `ImportError: cannot import name 'detect_path_prefix'`

- [ ] **Step 3: Implement prefix detection**

```python
# Add to src/coverage_normalizer.py

def detect_path_prefix(xml_paths: list[str], diff_files: list[str]) -> str:
    """Auto-detect the path prefix in coverage XML that maps to git-relative paths.

    Compares absolute paths from coverage XML against git-relative diff_files.
    When stripping a candidate prefix from an XML path produces a match in diff_files,
    that prefix is confirmed.

    Returns the prefix string (including trailing separator) or "" if no prefix detected.
    """
    if not xml_paths or not diff_files:
        return ""

    diff_set = set(diff_files)

    # Only consider absolute paths (those starting with / or drive letter)
    absolute_xml_paths = [
        p for p in xml_paths
        if p.startswith("/") or (len(p) >= 3 and p[1] == ":" and p[2] in ("/", "\\"))
    ]
    if not absolute_xml_paths:
        return ""

    # Strategy: for each XML path, check if removing a prefix yields a diff_file match.
    # The prefix is the portion before the matching suffix.
    best_prefix = ""
    match_count = 0

    for xml_path in absolute_xml_paths:
        # Normalize separators for comparison
        normalized = xml_path.replace("\\", "/")
        for diff_file in diff_set:
            if normalized.endswith("/" + diff_file) or normalized.endswith("\\" + diff_file):
                candidate = xml_path[: len(xml_path) - len(diff_file)]
                if not best_prefix:
                    best_prefix = candidate
                    match_count = 1
                elif candidate == best_prefix:
                    match_count += 1
                break

    # Require at least 1 confirmed match
    if match_count >= 1:
        return best_prefix
    return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestDetectPathPrefix -v`
Expected: All 6 PASS

- [ ] **Step 5: Write failing tests for Clover XML rewriting**

```python
# tests/test_coverage_normalizer.py (append to file)

from src.coverage_normalizer import normalize_coverage_file


class TestNormalizeClover:
    def test_rewrites_clover_file_paths(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/Controller/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
      <line num="15" type="stmt" count="0"/>
    </file>
    <file name="/var/www/app/lib/Model/UserDao.php">
      <line num="1" type="stmt" count="5"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/Controller/AuthCookie.php", "lib/Model/UserDao.php"]
        result = normalize_coverage_file(str(xml), diff_files)

        # Should return a new temp file path (not the original)
        assert result != str(xml)
        assert Path(result).exists()

        # Parse the rewritten XML and verify paths are now relative
        tree = ET.parse(result)
        files = tree.findall(".//file")
        names = [f.get("name") for f in files]
        assert "lib/Controller/AuthCookie.php" in names
        assert "lib/Model/UserDao.php" in names
        # No absolute paths remain
        assert not any(n.startswith("/") for n in names if n)

    def test_preserves_line_coverage_data(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/app/src/auth.py">
      <line num="10" type="stmt" count="3"/>
      <line num="20" type="cond" count="0"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["src/auth.py"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)
        lines = tree.findall(".//line")
        assert len(lines) == 2
        assert lines[0].get("num") == "10"
        assert lines[0].get("count") == "3"

    def test_returns_original_when_no_prefix_detected(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="lib/AuthCookie.php">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        # No rewriting needed — returns original path
        assert result == str(xml)

    def test_returns_original_for_non_xml(self, tmp_path: Path):
        lcov = tmp_path / "coverage.info"
        lcov.write_text("SF:/app/src/auth.py\nDA:1,1\nend_of_record\n")
        result = normalize_coverage_file(str(lcov), ["src/auth.py"])
        assert result == str(lcov)
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestNormalizeClover -v`
Expected: FAIL with `ImportError: cannot import name 'normalize_coverage_file'`

- [ ] **Step 7: Implement Clover normalization**

```python
# Add to src/coverage_normalizer.py

def _extract_clover_paths(tree: ET.ElementTree) -> list[str]:
    """Extract all file paths from a Clover XML tree."""
    paths: list[str] = []
    for file_elem in tree.findall(".//file"):
        # Clover uses 'name' attribute (some variants use 'path')
        name = file_elem.get("name") or file_elem.get("path") or ""
        if name:
            paths.append(name)
    return paths


def _rewrite_clover(filepath: str, prefix: str) -> str:
    """Rewrite Clover XML stripping `prefix` from all <file> name/path attributes.

    Returns path to a temp file with rewritten content.
    """
    tree = ET.parse(filepath)  # noqa: S314
    for file_elem in tree.findall(".//file"):
        for attr in ("name", "path"):
            val = file_elem.get(attr)
            if val and val.startswith(prefix):
                file_elem.set(attr, val[len(prefix):])

    # Write to a named temp file that persists until process exit
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="tg_clover_")
    os.close(fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    return tmp_path


def normalize_coverage_file(filepath: str, diff_files: list[str]) -> str:
    """Normalize a coverage file's paths for diff-cover compatibility.

    Detects the format, auto-detects Docker path prefixes, and rewrites
    if necessary. Returns the path to use (original or a temp rewritten copy).

    For Cobertura: verifies <source> element enables correct resolution.
    For Clover: rewrites <file name="..."> attributes with stripped prefix.
    For JaCoCo/LCOV/Unknown: returns original (no normalization needed/possible).
    """
    fmt = detect_format(filepath)

    if fmt == CoverageFormat.CLOVER:
        try:
            tree = ET.parse(filepath)  # noqa: S314
        except ET.ParseError:
            return filepath
        xml_paths = _extract_clover_paths(tree)
        prefix = detect_path_prefix(xml_paths, diff_files)
        if prefix:
            return _rewrite_clover(filepath, prefix)
        return filepath

    if fmt == CoverageFormat.COBERTURA:
        return _normalize_cobertura(filepath, diff_files)

    # JaCoCo uses --src-roots (handled by diff-cover), LCOV can't be normalized
    return filepath
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestNormalizeClover -v`
Expected: FAIL — `_normalize_cobertura` not yet defined. Add a stub:

```python
def _normalize_cobertura(filepath: str, diff_files: list[str]) -> str:
    """Normalize Cobertura XML. Implemented in Task 3."""
    return filepath
```

Then re-run. Expected: All 4 PASS

- [ ] **Step 9: Commit**

```bash
git add src/coverage_normalizer.py tests/test_coverage_normalizer.py
git commit -m "feat(layer1): add Clover path prefix auto-detection and XML rewriting"
```

---

### Task 3: Cobertura Source Element Verification + Fix

**Files:**
- Modify: `src/coverage_normalizer.py`
- Test: `tests/test_coverage_normalizer.py`

For Cobertura, diff-cover stores two keys per class in its cache:
1. The bare `filename` attribute (e.g., `lib/AuthCookie.php`)
2. `source + "/" + filename` (e.g., `/var/www/app/lib/AuthCookie.php`)

The git-relative path lookup (`src_rel_path`) matches the bare `filename` — so Cobertura SHOULD work when `filename` is already relative. But if the coverage tool writes absolute filenames (rare but possible) OR if there are no `<source>` elements, it breaks.

The normalizer must:
1. Check if `<class filename="...">` values are absolute → rewrite to relative
2. Verify `<source>` element exists — if missing and filenames are absolute, add it

- [ ] **Step 1: Write failing tests for Cobertura normalization**

```python
# tests/test_coverage_normalizer.py (append to file)

class TestNormalizeCobertura:
    def test_returns_original_when_filenames_relative(self, tmp_path: Path):
        """Well-formed Cobertura with relative filenames needs no rewriting."""
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" line-rate="0.8">
  <sources><source>/var/www/app</source></sources>
  <packages>
    <package name="lib">
      <classes>
        <class name="AuthCookie" filename="lib/AuthCookie.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        # No rewriting needed
        assert result == str(xml)

    def test_rewrites_absolute_filenames_to_relative(self, tmp_path: Path):
        """Cobertura with absolute filename attributes gets rewritten."""
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class name="AuthCookie" filename="/var/www/app/lib/AuthCookie.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)
        tree = ET.parse(result)
        classes = tree.findall(".//class")
        assert classes[0].get("filename") == "lib/AuthCookie.php"

    def test_adds_source_element_when_missing(self, tmp_path: Path):
        """If no <source> and filenames are absolute, add <source> after rewriting."""
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class name="AuthCookie" filename="/docker/app/lib/AuthCookie.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)
        tree = ET.parse(result)
        # Filenames should be relative now
        classes = tree.findall(".//class")
        assert classes[0].get("filename") == "lib/AuthCookie.php"

    def test_preserves_coverage_data(self, tmp_path: Path):
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class name="Auth" filename="/app/lib/Auth.php" line-rate="0.9">
          <lines>
            <line number="1" hits="5"/>
            <line number="2" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/Auth.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)
        lines = tree.findall(".//line")
        assert len(lines) == 2
        assert lines[0].get("hits") == "5"
        assert lines[1].get("hits") == "0"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestNormalizeCobertura -v`
Expected: First test passes (returns original), others FAIL

- [ ] **Step 3: Implement Cobertura normalization**

Replace the `_normalize_cobertura` stub in `src/coverage_normalizer.py`:

```python
def _normalize_cobertura(filepath: str, diff_files: list[str]) -> str:
    """Normalize Cobertura XML paths for diff-cover compatibility.

    If <class filename="..."> values are already relative, no rewriting is needed
    (diff-cover handles this natively via its cache lookup).

    If filenames are absolute (Docker paths), detect the prefix and rewrite
    filenames to relative paths.
    """
    try:
        tree = ET.parse(filepath)  # noqa: S314
    except ET.ParseError:
        return filepath

    root = tree.getroot()
    classes = root.findall(".//class")
    if not classes:
        return filepath

    # Check if filenames are absolute
    filenames = [c.get("filename", "") for c in classes]
    has_absolute = any(
        f.startswith("/") or (len(f) >= 3 and f[1] == ":" and f[2] in ("/", "\\"))
        for f in filenames
        if f
    )
    if not has_absolute:
        return filepath  # Already relative — diff-cover handles it natively

    # Auto-detect prefix from absolute filenames
    prefix = detect_path_prefix(filenames, diff_files)
    if not prefix:
        return filepath  # Can't determine prefix — pass through unchanged

    # Rewrite filenames to relative
    for clazz in classes:
        fname = clazz.get("filename", "")
        if fname.startswith(prefix):
            clazz.set("filename", fname[len(prefix):])

    # Write to temp file
    fd, tmp_path = tempfile.mkstemp(suffix=".xml", prefix="tg_cobertura_")
    os.close(fd)
    tree.write(tmp_path, encoding="unicode", xml_declaration=True)
    return tmp_path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestNormalizeCobertura -v`
Expected: All 4 PASS

- [ ] **Step 5: Commit**

```bash
git add src/coverage_normalizer.py tests/test_coverage_normalizer.py
git commit -m "feat(layer1): add Cobertura path normalization for absolute filenames"
```

---

### Task 4: Integrate Normalizer into Layer 1

**Files:**
- Modify: `src/layer1_coverage.py`
- Modify: `tests/test_layer1_coverage.py`

The normalizer must run BEFORE `_compute_diff_coverage()` is called, replacing the `valid_files` list with normalized paths. Temp files are cleaned up after diff-cover finishes.

- [ ] **Step 1: Write failing integration test**

```python
# tests/test_layer1_coverage.py (append new class)

from xml.etree import ElementTree as ET


class TestDockerPathNormalization:
    """Integration tests verifying Layer 1 handles Docker-internal coverage paths."""

    @patch("src.layer1_coverage.subprocess.run")
    def test_clover_docker_paths_produce_correct_diff_cover_input(self, mock_run, tmp_path):
        """When Clover XML has Docker paths, the normalizer should rewrite them
        so diff-cover can match files. We verify by checking that diff-cover is
        called with a rewritten file (not the original)."""
        cov = tmp_path / "php-coverage.xml"
        cov.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/Controller/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
    <file name="/var/www/app/lib/Model/UserDao.php">
      <line num="5" type="stmt" count="3"/>
    </file>
  </project>
</coverage>""")
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 90.0, "src_stats": {"lib/Controller/AuthCookie.php": {"percent_covered": 90.0}}}',
            stderr="",
        )
        diff_files = ["lib/Controller/AuthCookie.php", "lib/Model/UserDao.php"]
        result = run_layer1([str(cov)], threshold=80, diff_files=diff_files)

        # The key assertion: diff-cover was called with a DIFFERENT file (normalized)
        cmd = mock_run.call_args[0][0]
        # The coverage file in the command should NOT be the original Docker-path file
        assert cmd[1] != str(cov)  # normalized temp file used instead
        # And the result should show the matched file
        assert any(
            fv.file == "lib/Controller/AuthCookie.php" and fv.verdict == Verdict.PASS
            for fv in result.file_verdicts
        )

    @patch("src.layer1_coverage.subprocess.run")
    def test_already_relative_paths_use_original_file(self, mock_run, tmp_path):
        """When paths are already relative, no temp file is created."""
        cov = tmp_path / "coverage.xml"
        cov.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        mock_run.return_value = subprocess.CompletedProcess(
            args=["diff-cover"],
            returncode=0,
            stdout='{"total_percent_covered": 90.0, "src_stats": {"lib/AuthCookie.php": {"percent_covered": 90.0}}}',
            stderr="",
        )
        result = run_layer1([str(cov)], threshold=80, diff_files=["lib/AuthCookie.php"])
        cmd = mock_run.call_args[0][0]
        # Should use original file (no rewriting)
        assert cmd[1] == str(cov)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source .venv/bin/activate && pytest tests/test_layer1_coverage.py::TestDockerPathNormalization -v`
Expected: FAIL — normalizer not yet integrated

- [ ] **Step 3: Integrate normalizer into `_compute_diff_coverage` call site**

Modify `src/layer1_coverage.py` — add import and call normalization before diff-cover:

```python
# Add import at top of file
from src.coverage_normalizer import normalize_coverage_file

# In run_layer1(), after the valid_files check (line 117), add normalization:
# Replace the line:
#     total_pct, per_file, error_reason = _compute_diff_coverage(valid_files)
# With:
    normalized_files = [normalize_coverage_file(f, diff_files) for f in valid_files]
    try:
        total_pct, per_file, error_reason = _compute_diff_coverage(normalized_files)
    finally:
        # Clean up temp files created by normalization
        for nf, vf in zip(normalized_files, valid_files):
            if nf != vf:
                Path(nf).unlink(missing_ok=True)
```

- [ ] **Step 4: Run all Layer 1 tests to verify nothing breaks**

Run: `source .venv/bin/activate && pytest tests/test_layer1_coverage.py -v`
Expected: All tests PASS (existing tests use mocked `_compute_diff_coverage` so normalization is transparent)

- [ ] **Step 5: Run the new integration tests**

Run: `source .venv/bin/activate && pytest tests/test_layer1_coverage.py::TestDockerPathNormalization -v`
Expected: All 2 PASS

- [ ] **Step 6: Commit**

```bash
git add src/layer1_coverage.py tests/test_layer1_coverage.py
git commit -m "feat(layer1): integrate coverage path normalizer into diff-cover pipeline"
```

---

### Task 5: Edge Cases + Robustness

**Files:**
- Modify: `tests/test_coverage_normalizer.py`
- Modify: `src/coverage_normalizer.py` (if any edge case fixes needed)

- [ ] **Step 1: Write edge case tests**

```python
# tests/test_coverage_normalizer.py (append to file)

class TestEdgeCases:
    def test_normalize_handles_empty_diff_files(self, tmp_path: Path):
        """Should not crash when diff_files is empty."""
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/app/lib/Auth.php">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        result = normalize_coverage_file(str(xml), [])
        assert result == str(xml)  # No diff_files → can't detect prefix → original

    def test_normalize_handles_large_prefix_ratio(self, tmp_path: Path):
        """Even if only 1 file matches, prefix should be detected."""
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/docker/workspace/src/a.php">
      <line num="1" type="stmt" count="1"/>
    </file>
    <file name="/docker/workspace/src/b.php">
      <line num="1" type="stmt" count="1"/>
    </file>
    <file name="/docker/workspace/src/c.php">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        # Only 1 of 3 XML files is in the diff
        diff_files = ["src/a.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)  # Should still rewrite
        tree = ET.parse(result)
        names = [f.get("name") for f in tree.findall(".//file")]
        assert "src/a.php" in names
        assert "src/b.php" in names
        assert "src/c.php" in names

    def test_normalize_preserves_xml_structure(self, tmp_path: Path):
        """Rewriting should preserve metrics, packages, and other elements."""
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <metrics statements="100" coveredstatements="80"/>
    <package name="Controller">
      <file name="/app/lib/Controller/Auth.php">
        <class name="Auth" namespace="Controller">
          <metrics methods="5" coveredmethods="4"/>
        </class>
        <line num="10" type="stmt" count="1"/>
      </file>
    </package>
  </project>
</coverage>""")
        diff_files = ["lib/Controller/Auth.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)
        # Metrics preserved
        assert tree.find(".//metrics[@statements]") is not None
        # Package preserved
        assert tree.find(".//package[@name='Controller']") is not None
        # Class preserved
        assert tree.find(".//class[@name='Auth']") is not None

    def test_multiple_coverage_files_normalized_independently(self, tmp_path: Path):
        """Each file is normalized on its own — no cross-contamination."""
        php_cov = tmp_path / "php-coverage.xml"
        php_cov.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/Auth.php">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        js_cov = tmp_path / "js-coverage.xml"
        js_cov.write_text("""<?xml version="1.0" ?>
<coverage line-rate="0.9">
  <packages>
    <package name="src">
      <classes>
        <class filename="public/js/app.js" line-rate="0.9">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/Auth.php", "public/js/app.js"]

        php_result = normalize_coverage_file(str(php_cov), diff_files)
        js_result = normalize_coverage_file(str(js_cov), diff_files)

        # PHP (Clover with Docker paths) should be rewritten
        assert php_result != str(php_cov)
        # JS (Cobertura with relative paths) should be unchanged
        assert js_result == str(js_cov)
```

- [ ] **Step 2: Run edge case tests**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestEdgeCases -v`
Expected: All PASS (if implementation is correct from Tasks 2-3). Fix any failures.

- [ ] **Step 3: Run full test suite**

Run: `source .venv/bin/activate && pytest -v`
Expected: All tests PASS, no regressions

- [ ] **Step 4: Run linting**

Run: `source .venv/bin/activate && ruff check src/coverage_normalizer.py tests/test_coverage_normalizer.py`
Expected: No errors

- [ ] **Step 5: Commit**

```bash
git add src/coverage_normalizer.py tests/test_coverage_normalizer.py
git commit -m "test(layer1): add edge case coverage for path normalizer"
```

---

### Task 6: Diagnostic Warning When Normalization Activates

**Files:**
- Modify: `src/coverage_normalizer.py`
- Modify: `tests/test_coverage_normalizer.py`

When normalization detects and fixes a path mismatch, emit a GitHub Actions `::warning::` annotation so users know it happened. This aids debugging without breaking the pipeline.

- [ ] **Step 1: Write failing test**

```python
# tests/test_coverage_normalizer.py (append)

import io
import sys


class TestDiagnosticOutput:
    def test_emits_warning_when_clover_rewritten(self, tmp_path: Path, capsys):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/docker/app/src/auth.py">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        normalize_coverage_file(str(xml), ["src/auth.py"])
        captured = capsys.readouterr()
        assert "::warning::" in captured.out
        assert "/docker/app/" in captured.out

    def test_no_warning_when_paths_already_correct(self, tmp_path: Path, capsys):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="src/auth.py">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        normalize_coverage_file(str(xml), ["src/auth.py"])
        captured = capsys.readouterr()
        assert "::warning::" not in captured.out
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestDiagnosticOutput -v`
Expected: First test FAIL (no warning emitted yet)

- [ ] **Step 3: Add diagnostic print to `normalize_coverage_file`**

In `src/coverage_normalizer.py`, in the Clover branch of `normalize_coverage_file()`, after detecting prefix and before returning the rewritten file:

```python
        if prefix:
            print(
                f"::warning::Coverage path normalization: stripped prefix '{prefix}' "
                f"from {len(xml_paths)} file path(s) in {Path(filepath).name} "
                f"(Docker/container paths detected)"
            )
            return _rewrite_clover(filepath, prefix)
```

Similarly for Cobertura in `_normalize_cobertura()`, after detecting prefix:

```python
    if not prefix:
        return filepath

    print(
        f"::warning::Coverage path normalization: stripped prefix '{prefix}' "
        f"from {len([f for f in filenames if f.startswith(prefix)])} filename(s) "
        f"in {Path(filepath).name} (Docker/container paths detected)"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `source .venv/bin/activate && pytest tests/test_coverage_normalizer.py::TestDiagnosticOutput -v`
Expected: All 2 PASS

- [ ] **Step 5: Run full suite one final time**

Run: `source .venv/bin/activate && pytest -v && ruff check src/ tests/`
Expected: All green

- [ ] **Step 6: Commit**

```bash
git add src/coverage_normalizer.py tests/test_coverage_normalizer.py
git commit -m "feat(layer1): emit diagnostic warning when Docker paths are auto-fixed"
```

---

## Summary of Changes

| Component | Change |
|-----------|--------|
| `src/coverage_normalizer.py` | NEW — format detection + Clover rewriter + Cobertura fixer |
| `src/layer1_coverage.py` | 3-line change — call normalizer before diff-cover |
| `tests/test_coverage_normalizer.py` | NEW — ~25 tests covering detection, rewriting, edge cases |
| `tests/test_layer1_coverage.py` | 2 new integration tests for Docker path scenario |

**No new action.yml inputs required.** The fix is fully automatic.

**Backwards compatible.** When paths are already correct (no Docker), normalizer returns the original file unchanged. No performance impact.
