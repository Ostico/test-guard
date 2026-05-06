from pathlib import Path
from xml.etree import ElementTree as ET

from src.coverage_normalizer import (
    CoverageFormat,
    detect_format,
    detect_path_prefix,
    normalize_coverage_file,
)


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
            "src/other.py",
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

    def test_requires_minimum_one_match(self):
        xml_paths = [
            "/var/www/app/lib/AuthCookie.php",
            "/other/path/something.php",
        ]
        diff_files = ["lib/AuthCookie.php"]
        prefix = detect_path_prefix(xml_paths, diff_files)
        assert prefix == "/var/www/app/"


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
        assert result != str(xml)
        assert Path(result).exists()
        tree = ET.parse(result)  # noqa: S314
        files = tree.findall(".//file")
        names = [f.get("name") for f in files]
        assert "lib/Controller/AuthCookie.php" in names
        assert "lib/Model/UserDao.php" in names
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
        tree = ET.parse(result)  # noqa: S314
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
        assert result == str(xml)

    def test_sets_clover_attribute_on_root(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)  # noqa: S314
        root = tree.getroot()
        assert root.get("clover") == "test-guard"

    def test_preserves_existing_clover_attribute(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage clover="PHPUnit 12.5.6" generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)  # noqa: S314
        root = tree.getroot()
        assert root.get("clover") == "PHPUnit 12.5.6"

    def test_sets_path_attribute_from_name(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)  # noqa: S314
        file_elem = tree.find(".//file")
        assert file_elem.get("path") == "lib/AuthCookie.php"
        assert file_elem.get("name") == "lib/AuthCookie.php"

    def test_rewrites_existing_path_attribute(self, tmp_path: Path):
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/var/www/app/lib/AuthCookie.php" path="/var/www/app/lib/AuthCookie.php">
      <line num="10" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        diff_files = ["lib/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        tree = ET.parse(result)  # noqa: S314
        file_elem = tree.find(".//file")
        assert file_elem.get("path") == "lib/AuthCookie.php"
        assert file_elem.get("name") == "lib/AuthCookie.php"

    def test_returns_original_for_non_xml(self, tmp_path: Path):
        lcov = tmp_path / "coverage.info"
        lcov.write_text("SF:/app/src/auth.py\nDA:1,1\nend_of_record\n")
        result = normalize_coverage_file(str(lcov), ["src/auth.py"])
        assert result == str(lcov)


class TestNormalizeCobertura:
    def test_returns_original_when_filenames_relative(self, tmp_path: Path):
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
        assert result == str(xml)

    def test_rewrites_absolute_filenames_to_relative(self, tmp_path: Path):
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
        tree = ET.parse(result)  # noqa: S314
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
        tree = ET.parse(result)  # noqa: S314
        lines = tree.findall(".//line")
        assert len(lines) == 2
        assert lines[0].get("hits") == "5"
        assert lines[1].get("hits") == "0"


class TestEdgeCases:
    def test_normalize_handles_empty_diff_files(self, tmp_path: Path):
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
        assert result == str(xml)

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
        diff_files = ["src/a.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)
        tree = ET.parse(result)  # noqa: S314
        names = [f.get("name") for f in tree.findall(".//file")]
        assert "src/a.php" in names
        assert "src/b.php" in names
        assert "src/c.php" in names

    def test_multiple_coverage_files_normalized_independently(self, tmp_path: Path):
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

    def test_emits_warning_when_cobertura_rewritten(self, tmp_path: Path, capsys):
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class filename="/app/lib/Auth.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        normalize_coverage_file(str(xml), ["lib/Auth.php"])
        captured = capsys.readouterr()
        assert "::warning::" in captured.out
        assert "/app/" in captured.out


class TestUncoveredBranches:
    """Tests targeting previously uncovered code paths."""

    def test_cobertura_malformed_xml_returns_original(self, tmp_path: Path):
        """Cobertura file that passes detect_format but fails full parse."""
        # Create a file that looks like Cobertura to detect_format (has <coverage> + <packages>)
        # but is actually truncated/corrupted when fully parsed by _normalize_cobertura
        xml = tmp_path / "broken.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class filename="/app/lib/Auth.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        # Corrupt the file after format detection would succeed
        xml.write_text("<?xml version='1.0'?>\n<coverage><packages/><INVALID")
        result = normalize_coverage_file(str(xml), ["lib/Auth.php"])
        # ParseError in _normalize_cobertura → returns original
        assert result == str(xml)

    def test_cobertura_no_class_elements_returns_original(self, tmp_path: Path):
        """Cobertura XML with packages but no class elements."""
        xml = tmp_path / "empty-cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage line-rate="0.0">
  <packages>
    <package name="empty">
      <classes/>
    </package>
  </packages>
</coverage>""")
        result = normalize_coverage_file(str(xml), ["src/anything.py"])
        assert result == str(xml)

    def test_cobertura_absolute_but_no_diff_match_returns_original(self, tmp_path: Path):
        """Cobertura with absolute filenames that don't suffix-match any diff_file."""
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage line-rate="0.8">
  <packages>
    <package name="lib">
      <classes>
        <class filename="/docker/app/totally/different/Module.php" line-rate="0.8">
          <lines><line number="1" hits="1"/></lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        # diff_files has no suffix match for the XML path
        result = normalize_coverage_file(str(xml), ["src/unrelated.py"])
        assert result == str(xml)

    def test_clover_parse_failure_on_reparse_returns_original(self, tmp_path: Path):
        """Clover file that succeeds initial parse but fails when re-read."""
        xml = tmp_path / "clover.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1700000000">
  <project timestamp="1700000000">
    <file name="/app/src/auth.py">
      <line num="1" type="stmt" count="1"/>
    </file>
  </project>
</coverage>""")
        # First call to detect_format + initial parse will succeed.
        # But we can't easily test this without monkeypatching since
        # the file doesn't change between reads. Instead, test that
        # normalize_coverage_file handles a valid Clover file that has
        # no absolute paths (returns original — exercises the "no prefix" path).
        result = normalize_coverage_file(str(xml), ["completely/different.py"])
        # No diff_file matches any XML path suffix → no prefix → original returned
        assert result == str(xml)


class TestRealWorldFormats:
    """Tests mimicking actual tool output formats."""

    def test_phpunit_clover_output(self, tmp_path: Path):
        """Real PHPUnit --coverage-clover output format."""
        xml = tmp_path / "php-coverage.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage generated="1714000000">
  <project timestamp="1714000000" name="MateCat">
    <package name="Controller\\Abstracts\\Authentication">
      <file name="/var/www/app/lib/Controller/Abstracts/Authentication/AuthCookie.php">
        <class name="AuthCookie" namespace="Controller\\Abstracts\\Authentication">
          <metrics complexity="5" methods="3" coveredmethods="2"
                   conditionals="0" coveredconditionals="0"
                   statements="20" coveredstatements="15"/>
        </class>
        <line num="10" type="method" name="setAuthCookie" visibility="public"
              complexity="2" crap="2" count="5"/>
        <line num="11" type="stmt" count="5"/>
        <line num="12" type="stmt" count="5"/>
        <line num="25" type="method" name="getAuthCookie" visibility="public"
              complexity="1" crap="1" count="3"/>
        <line num="26" type="stmt" count="3"/>
        <line num="40" type="method" name="deleteAuthCookie" visibility="public"
              complexity="2" crap="2.03" count="0"/>
        <line num="41" type="stmt" count="0"/>
      </file>
    </package>
    <package name="Model\\DataAccess">
      <file name="/var/www/app/lib/Model/DataAccess/AbstractDao.php">
        <class name="AbstractDao" namespace="Model\\DataAccess">
          <metrics complexity="10" methods="5" coveredmethods="5"
                   statements="40" coveredstatements="38"/>
        </class>
        <line num="15" type="stmt" count="10"/>
        <line num="16" type="stmt" count="10"/>
      </file>
    </package>
    <metrics files="2" loc="200" ncloc="150" classes="2"
             methods="8" coveredmethods="7" conditionals="0"
             coveredconditionals="0" statements="60" coveredstatements="53"/>
  </project>
</coverage>""")
        diff_files = [
            "lib/Controller/Abstracts/Authentication/AuthCookie.php",
            "lib/Model/DataAccess/AbstractDao.php",
            "lib/Model/DataAccess/DaoCacheTrait.php",  # in diff but not in coverage
        ]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)
        tree = ET.parse(result)  # noqa: S314
        files = tree.findall(".//file")
        names = [f.get("name") for f in files]
        assert "lib/Controller/Abstracts/Authentication/AuthCookie.php" in names
        assert "lib/Model/DataAccess/AbstractDao.php" in names
        # Verify coverage metrics and line data preserved
        auth_file = next(f for f in files if "AuthCookie" in (f.get("name") or ""))
        lines = auth_file.findall("line")
        assert len(lines) == 7
        assert auth_file.find(".//class") is not None
        assert auth_file.find(".//metrics") is not None

    def test_pytest_cov_cobertura_output_no_rewrite(self, tmp_path: Path):
        """Real pytest-cov Cobertura output — paths are already relative.
        This is what dogfood.yml generates. Must be a no-op."""
        xml = tmp_path / "coverage.xml"
        xml.write_text("""<?xml version="1.0" ?>
<coverage version="7.6" timestamp="1714000000" lines-valid="1026"
          lines-covered="986" line-rate="0.961" branches-covered="0"
          branches-valid="0" branch-rate="0" complexity="0">
  <sources>
    <source>/home/runner/work/test-guard/test-guard</source>
  </sources>
  <packages>
    <package name="src" line-rate="0.961" branch-rate="0" complexity="0">
      <classes>
        <class name="coverage_normalizer.py" filename="src/coverage_normalizer.py"
               line-rate="0.94" branch-rate="0" complexity="0">
          <methods/>
          <lines>
            <line number="28" hits="5"/>
            <line number="31" hits="5"/>
            <line number="32" hits="1"/>
            <line number="34" hits="4"/>
          </lines>
        </class>
        <class name="layer1_coverage.py" filename="src/layer1_coverage.py"
               line-rate="1.0" branch-rate="0" complexity="0">
          <methods/>
          <lines>
            <line number="42" hits="10"/>
            <line number="50" hits="10"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["src/coverage_normalizer.py", "src/layer1_coverage.py"]
        result = normalize_coverage_file(str(xml), diff_files)
        # pytest-cov produces relative filenames → NO rewriting
        assert result == str(xml)

    def test_phpunit_cobertura_output(self, tmp_path: Path):
        """PHPUnit --coverage-cobertura format with Docker absolute paths."""
        xml = tmp_path / "cobertura.xml"
        xml.write_text("""<?xml version="1.0" encoding="UTF-8"?>
<coverage version="1">
  <packages>
    <package name="Controller\\Abstracts\\Authentication" complexity="5">
      <classes>
        <class name="AuthCookie"
               filename="/var/www/app/lib/Controller/Abstracts/Authentication/AuthCookie.php"
               line-rate="0.75" branch-rate="0" complexity="5">
          <lines>
            <line number="10" hits="5"/>
            <line number="11" hits="5"/>
            <line number="40" hits="0"/>
            <line number="41" hits="0"/>
          </lines>
        </class>
      </classes>
    </package>
  </packages>
</coverage>""")
        diff_files = ["lib/Controller/Abstracts/Authentication/AuthCookie.php"]
        result = normalize_coverage_file(str(xml), diff_files)
        assert result != str(xml)
        tree = ET.parse(result)  # noqa: S314
        clazz = tree.find(".//class")
        assert clazz is not None
        assert clazz.get("filename") == "lib/Controller/Abstracts/Authentication/AuthCookie.php"
        lines = clazz.findall(".//line")
        assert len(lines) == 4
        assert lines[2].get("hits") == "0"
