"""Security settings in templates/index.html."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "templates" / "index.html"


def _mathjax_config() -> str:
    template = INDEX_HTML.read_text(encoding="utf-8")
    return template[template.index("window.MathJax = {") : template.index("</script>")]


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


if __name__ == "__main__":
    unittest.main()
