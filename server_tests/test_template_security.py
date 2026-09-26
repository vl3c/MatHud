"""Security settings in templates/index.html."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

INDEX_HTML = Path(__file__).resolve().parent.parent / "templates" / "index.html"


class TestMathJaxConfig(unittest.TestCase):
    def test_html_extension_is_not_autoloaded(self) -> None:
        """MathJax's html extension (\\href, \\class, \\style, \\cssId) must stay off.

        With it, ``\\href{javascript:...}{...}`` in a chat message (live or loaded from a
        workspace) typesets to a clickable ``javascript:`` link.
        """
        template = INDEX_HTML.read_text(encoding="utf-8")
        config = template[template.index("window.MathJax = {") : template.index("</script>")]
        self.assertRegex(config, r"tex:\s*\{")
        self.assertIsNotNone(re.search(r"autoload:\s*\{\s*html:\s*\[\s*\]\s*\}", config), config)


if __name__ == "__main__":
    unittest.main()
