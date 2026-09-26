"""Security settings in templates/index.html."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "templates" / "index.html"
STATIC_DIR = INDEX_HTML.parent.parent / "static"


def _mathjax_config() -> str:
    """The inline <script> block that configures MathJax."""
    template = INDEX_HTML.read_text(encoding="utf-8")
    config_at = template.index("window.MathJax = {")
    start = template.rindex("<script>", 0, config_at)
    return template[start : template.index("</script>", config_at)]


class TestMathJaxConfig(unittest.TestCase):
    """MathJax's html extension (\\href, \\class, \\style, \\cssId) must stay off.

    With it, ``\\href{javascript:...}{...}`` in a chat message (live or loaded from a
    workspace) typesets to a clickable ``javascript:`` link.
    """

    def test_html_extension_is_not_autoloaded(self) -> None:
        config = _mathjax_config()
        self.assertRegex(config, r"tex:\s*\{")
        self.assertIsNotNone(re.search(r"autoload:\s*\{\s*html:\s*\[\s*\]\s*\}", config), config)

    def test_safe_extension_is_loaded_from_vendor(self) -> None:
        """ui/safe filters href/style/class/id that base TeX (\\mmlToken, \\bbox) can still set."""
        config = _mathjax_config()
        self.assertIsNotNone(re.search(r"loader:\s*\{\s*load:\s*\[\s*'ui/safe'\s*\]\s*\}", config), config)
        # [mathjax] resolves to the folder of tex-mml-chtml.js, so ui/safe must sit next to it.
        vendored = INDEX_HTML.parent.parent / "static" / "vendor" / "mathjax" / "3.2.2" / "es5"
        self.assertTrue((vendored / "tex-mml-chtml.js").is_file())
        self.assertTrue((vendored / "ui" / "safe.js").is_file())
        self.assertNotRegex(config, r"paths\s*:", "a custom path would stop ui/safe resolving to the vendor folder")

    def test_require_package_is_removed(self) -> None:
        """Without this, \\require{html} loads the extension despite the autoload setting."""
        config = _mathjax_config()
        self.assertIsNotNone(
            re.search(r"packages:\s*\{\s*'\[-\]':\s*\[\s*'require'\s*\]\s*\}", config),
            config,
        )

    def test_safe_options_allow_no_urls_and_no_cursor(self) -> None:
        """ui/safe's protocol check misses a tab inside "javascript:"; no URLs at all is safe."""
        config = _mathjax_config()
        self.assertIsNotNone(re.search(r"allow:\s*\{\s*URLs:\s*'none'\s*\}", config), config)
        self.assertIsNotNone(re.search(r"safeStyles:\s*\{\s*cursor:\s*false\s*\}", config), config)

    def test_fontfamily_is_filtered(self) -> None:
        """fontfamily (\\mmlToken, \\unicode's font) is pasted into the style unless filtered."""
        config = _mathjax_config()
        self.assertIn("safe.filterAttributes.set('fontfamily', 'filterFontFamily')", config)
        self.assertIn("var MATHUD_SAFE_FONT_FAMILY = /^[\\w\\s,'\"-]*$/;", config)
        # The filter is installed after the default startup has created the document.
        self.assertLess(config.index("MathJax.startup.defaultReady()"), config.index("filterAttributes.set"))

    def test_page_is_not_typeset_outside_the_sanitised_chat_path(self) -> None:
        self.assertRegex(_mathjax_config(), r"typeset:\s*false")

    def test_output_sanitizer_is_loaded(self) -> None:
        template = INDEX_HTML.read_text(encoding="utf-8")
        self.assertIn("url_for('static', filename='math_output_sanitizer.js')", template)
        self.assertTrue((STATIC_DIR / "math_output_sanitizer.js").is_file())


if __name__ == "__main__":
    unittest.main()
