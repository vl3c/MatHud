"""Browser libraries must be served from static/vendor/ so MatHud works offline.

Covers the HTML templates (no external script/stylesheet/font URLs, every
referenced static file exists) and the pinned manifest in
scripts/vendor_js_libs.py (hash verification in --check mode).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import sys
import tempfile
import unittest
from html.parser import HTMLParser
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
TEMPLATES_DIR = REPO_ROOT / "templates"
STATIC_DIR = REPO_ROOT / "static"
SCRIPT_PATH = REPO_ROOT / "scripts" / "vendor_js_libs.py"

# Jinja expression used by the templates to reference static files.
STATIC_URL_FOR = re.compile(r"""url_for\(\s*['"]static['"]\s*,\s*filename\s*=\s*['"]([^'"]+)['"]\s*\)""")
EXTERNAL_URL = re.compile(r"^\s*(?:https?:)?//", re.IGNORECASE)
CSS_EXTERNAL_URL = re.compile(r"""(?:url\(|@import)\s*['"]?\s*(?:https?:)?//""", re.IGNORECASE)


def _load_vendor_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("vendor_js_libs", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    return module


vendor_js_libs = _load_vendor_script()


class _ResourceCollector(HTMLParser):
    """Collect (tag, attribute, value) for every attribute that makes the browser fetch something."""

    FETCHING_ATTRS = {"script": ("src",), "link": ("href",), "img": ("src",), "iframe": ("src",)}

    def __init__(self) -> None:
        super().__init__()
        self.resources: list[tuple[str, str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        for name, value in attrs:
            if value is not None and name in self.FETCHING_ATTRS.get(tag, ()):
                self.resources.append((tag, name, value))


def _templates() -> list[Path]:
    return sorted(TEMPLATES_DIR.glob("*.html"))


def _resources(template: Path) -> list[tuple[str, str, str]]:
    collector = _ResourceCollector()
    collector.feed(template.read_text(encoding="utf-8"))
    return collector.resources


class TestTemplatesUseLocalAssets(unittest.TestCase):
    def test_templates_exist(self) -> None:
        names = {t.name for t in _templates()}
        self.assertIn("index.html", names)
        self.assertIn("login.html", names)

    def test_no_external_script_link_or_image_urls(self) -> None:
        for template in _templates():
            for tag, attr, value in _resources(template):
                with self.subTest(template=template.name, tag=tag, value=value):
                    self.assertIsNone(EXTERNAL_URL.match(value), f"<{tag} {attr}> loads an external URL")

    def test_no_external_urls_in_inline_css(self) -> None:
        for template in _templates():
            with self.subTest(template=template.name):
                self.assertIsNone(CSS_EXTERNAL_URL.search(template.read_text(encoding="utf-8")))

    def test_every_static_reference_exists(self) -> None:
        for template in _templates():
            for filename in STATIC_URL_FOR.findall(template.read_text(encoding="utf-8")):
                with self.subTest(template=template.name, filename=filename):
                    # Directories are allowed too (Brython's pythonpath points at static/client).
                    self.assertTrue((STATIC_DIR / filename).exists(), f"static/{filename} is missing")

    def test_index_loads_every_library_from_vendor(self) -> None:
        referenced = set(STATIC_URL_FOR.findall((TEMPLATES_DIR / "index.html").read_text(encoding="utf-8")))
        for path in (
            "vendor/inter/5.3.0/inter.css",
            "vendor/mathjax/3.2.2/es5/tex-mml-chtml.js",
            "vendor/mathjs/14.5.2/math.min.js",
            "vendor/nerdamer/1.1.13/nerdamer.core.js",
            "vendor/nerdamer/1.1.13/Algebra.js",
            "vendor/nerdamer/1.1.13/Calculus.js",
            "vendor/nerdamer/1.1.13/Solve.js",
            "vendor/nerdamer/1.1.13/Extra.js",
            "vendor/brython/3.12.5/brython.min.js",
            "vendor/brython/3.12.5/brython_stdlib.min.js",
        ):
            with self.subTest(path=path):
                self.assertIn(path, referenced)

    def test_vendored_script_references_are_pinned_in_manifest(self) -> None:
        pinned = {f.dest for f in vendor_js_libs.iter_files()} | set(vendor_js_libs.LOCAL_FILES)
        for template in _templates():
            for filename in STATIC_URL_FOR.findall(template.read_text(encoding="utf-8")):
                if filename.startswith("vendor/"):
                    with self.subTest(template=template.name, filename=filename):
                        self.assertIn(filename.removeprefix("vendor/"), pinned)

    def test_inter_css_references_only_pinned_font_files(self) -> None:
        css_path = STATIC_DIR / "vendor" / "inter" / "5.3.0" / "inter.css"
        urls = re.findall(r"url\(([^)]+)\)", css_path.read_text(encoding="utf-8"))
        self.assertEqual(len(urls), 7)
        pinned = {f.dest for f in vendor_js_libs.iter_files()}
        for url in urls:
            with self.subTest(url=url):
                self.assertIn("inter/5.3.0/" + url.strip("'\"").removeprefix("./"), pinned)


class TestVendorManifest(unittest.TestCase):
    def test_urls_come_from_allowed_cdns(self) -> None:
        for f in vendor_js_libs.iter_files():
            with self.subTest(url=f.url):
                self.assertTrue(f.url.startswith(vendor_js_libs.ALLOWED_URL_PREFIXES))

    def test_destinations_are_unique_and_versioned(self) -> None:
        dests = [f.dest for f in vendor_js_libs.iter_files()]
        self.assertEqual(len(dests), len(set(dests)))
        for lib in vendor_js_libs.LIBRARIES:
            for f in lib.files:
                with self.subTest(dest=f.dest):
                    self.assertIn(f"/{lib.version}/", f"/{f.dest}")
                    self.assertNotIn("..", f.dest)
                    self.assertRegex(f.sha256, r"^[0-9a-f]{64}$")

    def test_every_library_ships_a_license_file(self) -> None:
        for lib in vendor_js_libs.LIBRARIES:
            with self.subTest(library=lib.name):
                self.assertTrue(any(re.search(r"licen[cs]e", f.dest, re.IGNORECASE) for f in lib.files))

    def test_check_passes_for_committed_vendor_dir(self) -> None:
        self.assertEqual(vendor_js_libs.check(), [])

    def test_check_mode_cli_succeeds_without_network(self) -> None:
        self.assertEqual(vendor_js_libs.main(["--check"]), 0)

    def test_check_reports_missing_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            problems = vendor_js_libs.check(Path(tmp))
        expected = len(vendor_js_libs.iter_files()) + len(vendor_js_libs.LOCAL_FILES)
        self.assertEqual(len(problems), expected)
        self.assertTrue(all(p.startswith("missing") for p in problems))

    def test_check_reports_modified_and_unexpected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vendor_copy = Path(tmp) / "vendor"
            shutil.copytree(vendor_js_libs.VENDOR_DIR, vendor_copy)
            (vendor_copy / "nerdamer" / "1.1.13" / "license.txt").write_bytes(b"tampered")
            (vendor_copy / "stray.js").write_text("// not pinned", encoding="utf-8")
            problems = vendor_js_libs.check(vendor_copy)
        self.assertEqual(
            problems,
            ["hash mismatch: nerdamer/1.1.13/license.txt", "unexpected file: stray.js"],
        )

    def test_check_ignores_os_junk_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            vendor_copy = Path(tmp) / "vendor"
            shutil.copytree(vendor_js_libs.VENDOR_DIR, vendor_copy)
            (vendor_copy / ".DS_Store").write_bytes(b"\x00")
            (vendor_copy / "mathjax" / "Thumbs.db").write_bytes(b"\x00")
            (vendor_copy / "brython" / "3.12.5" / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
            problems = vendor_js_libs.check(vendor_copy)
        self.assertEqual(problems, [])


class TestVendorCaching(unittest.TestCase):
    """Vendored files live under versioned paths, so browsers may cache them for long."""

    def setUp(self) -> None:
        from static.app_manager import MatHudFlask

        self.client = MatHudFlask("static.app_manager", static_folder=str(STATIC_DIR)).test_client()

    def _cache_control(self, path: str) -> str:
        response = self.client.get(path)
        try:
            self.assertEqual(response.status_code, 200, path)
            return str(response.headers.get("Cache-Control", ""))
        finally:
            response.close()

    def test_vendor_files_get_a_long_max_age(self) -> None:
        cache_control = self._cache_control("/static/vendor/brython/3.12.5/brython.min.js")

        match = re.search(r"max-age=(\d+)", cache_control)
        self.assertIsNotNone(match, cache_control)
        assert match is not None
        self.assertGreaterEqual(int(match.group(1)), 30 * 24 * 3600)
        self.assertNotIn("no-cache", cache_control)

    def test_other_static_files_keep_revalidating(self) -> None:
        cache_control = self._cache_control("/static/style.css")

        self.assertIn("no-cache", cache_control)
        self.assertNotIn("max-age", cache_control)


if __name__ == "__main__":
    unittest.main()
